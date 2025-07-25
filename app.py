import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
# Keep this import for decoding
from mltu.utils.text_utils import ctc_decoder as mltu_ctc_decoder 
import Levenshtein
import pandas as pd
import tempfile
from pathlib import Path

# --- Import configurations directly from your scripts/config.py ---
# Ensure your 'config.py' is correctly set up as a module
# If your app.py is outside scripts folder, you might need 'from scripts.config import ...'
# Assuming app.py can directly see 'config' if scripts is in PYTHONPATH or app.py is in root.
# For consistency with other scripts, let's use:
try:
    from scripts.config import image_height, image_width, characters, model_path
    # Also load target_medicines if you want to use it for initial medicine name list
    from scripts.config import target_medicines
except ImportError:
    # Fallback if scripts.config isn't directly importable (e.g., if app.py is in root but scripts isn't a package)
    print("Warning: Could not import from scripts.config. Attempting direct import (requires path setup).")
    from config import image_height, image_width, characters, model_path, target_medicines

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- Global variables for Model ---
ocr_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_medicine_names = [] # This will be populated with a list of known medicine names

# --- CRNN Model Definition (Copied from model_utils.py, aligned for 1-channel grayscale) ---
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            # First CNN block
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output H = H/2

            # Second CNN block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output H = H/4
        )
        
        # Calculate input size for RNN
        # For image_height=32, after two 2x2 max pools, height becomes 32 // 2 // 2 = 8
        # Input channels to RNN is 128 (from last CNN layer) * 8 (final height) = 1024
        self.rnn_input_size = 128 * (image_height // 4) 

        # RNN layers (Bidirectional LSTMs)
        # batch_first=True because we will reshape CNN output to [B, W, C*H]
        self.rnn1 = nn.LSTM(self.rnn_input_size, 256, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True) # 512 because bidirectional (256*2)

        # Output classifier (Linear layer)
        self.fc = nn.Linear(512, num_classes) # 512 from bidirectional output of rnn2

    def forward(self, input_tensor):
        # Pass through CNN
        conv_features = self.cnn(input_tensor) # [B, C, H, W]

        # Reshape CNN output for RNN input
        # Target shape for batch_first LSTM: [Batch_Size, Sequence_Length (W), Feature_Size (C*H)]
        batch_size, num_channels, H, W = conv_features.size()
        
        # Permute to get [B, W, C, H], then flatten C and H into Feature_Size
        conv_features = conv_features.permute(0, 3, 1, 2) # [B, W, C, H]
        conv_features = conv_features.contiguous().view(batch_size, W, num_channels * H) # [B, W, C*H]

        # Pass through RNN layers
        rnn_features, _ = self.rnn1(conv_features) # Output: [B, W, 2 * hidden_size] = [B, W, 512]
        rnn_features, _ = self.rnn2(rnn_features) # Output: [B, W, 512]

        # Pass through final classifier
        output = self.fc(rnn_features) # Output: [B, W, num_classes]

        # Permute output for CTC Loss and ctc_decoder: [Sequence_Length (T), Batch_Size (B), Num_Classes (C)]
        output = output.permute(1, 0, 2) # [W, B, num_classes]

        return output

# --- Preprocessing Function (Aligned for 1-channel grayscale) ---
def preprocess_image(image_input):
    """
    Preprocesses an image for the model.
    Accepts either a file path (str or Path) or a PIL Image object.
    Ensures output is 1-channel (Grayscale) and normalized to [-1, 1].
    Uses image_height and image_width from config.
    """
    if isinstance(image_input, (str, Path)):
        img_pil = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img_pil = image_input
    else:
        raise ValueError(
            f"Unsupported image input type: {type(image_input)}. "
            "Must be a file path (str/Path) or a PIL Image object."
        )

    img_pil = img_pil.convert('L')
    img_np = np.array(img_pil, dtype=np.float32)

    # Use image_width and image_height from config.py
    img_resized = cv2.resize(img_np, (image_width, image_height))

    img_normalized = img_resized / 255.0
    img_normalized = (img_normalized - 0.5) / 0.5

    image_tensor = torch.from_numpy(img_normalized).unsqueeze(0) 
    image_tensor = image_tensor.unsqueeze(0) # Add batch dimension

    return image_tensor

# --- Closest Match Function (No change) ---
def find_closest_match(predicted_word, medicine_list):
    """
    Finds the closest matching medicine name from a list using Levenshtein distance.
    """
    if not medicine_list:
        return "N/A (Medicine list empty)"
    
    predicted_word_lower = predicted_word.strip().lower()
    
    closest_match = None
    min_distance = float('inf')

    for medicine_name in medicine_list:
        current_distance = Levenshtein.distance(predicted_word_lower, medicine_name.strip().lower())
        if current_distance < min_distance:
            min_distance = current_distance
            closest_match = medicine_name
    
    # Optional: Set a threshold for "closeness"
    # If min_distance is too high, it might mean no good match
    if min_distance > len(predicted_word_lower) / 2 and len(predicted_word_lower) > 2: # Example threshold
        return f"{closest_match} (Weak match)"

    return closest_match


# --- Model and Resources Loading (Modified) ---
def load_resources():
    """
    Loads the trained OCR model and the list of medicine names.
    """
    global ocr_model, all_medicine_names

    # --- Load OCR Model (Directly from models/crnn_model.pth) ---
    full_model_path = Path(model_path) # model_path is from config.py
    
    if not full_model_path.exists():
        print(f"Error: Trained model not found at {full_model_path}. "
              "Please ensure your 'crnn_model.pth' exists in the 'models' folder and config.py is correct.")
        ocr_model = None
        return

    # Initialize the CRNN model architecture with 1 input channel
    ocr_model = CRNN(num_classes=len(characters) + 1).to(device)

    try:
        ocr_model.load_state_dict(torch.load(full_model_path, map_location=device))
        ocr_model.eval() # Set model to evaluation mode
        print(f"OCR Model loaded successfully from {full_model_path} on {device}")
    except Exception as e:
        print(f"Error loading model state dictionary from {full_model_path}: {e}")
        ocr_model = None
        return # Exit if model loading fails

    # --- Load All Medicine Names (for closest match) ---
    # Assuming your training_labels.csv or a similar file contains all valid medicine names
    # Adjust this path if your CSV is elsewhere
    medicine_names_csv_path = Path("dataset/doctors-handwritten-prescription-bd-dataset/Filtered/filtered_training_labels.csv")
    
    if not medicine_names_csv_path.exists():
        print(f"Warning: Medicine names CSV not found at {medicine_names_csv_path}. Closest match will not work correctly.")
        all_medicine_names = []
    else:
        try:
            df_medicines = pd.read_csv(medicine_names_csv_path)
            # Use a set for unique names, then convert to list and sort for consistency
            all_medicine_names = sorted(list(df_medicines["MEDICINE_NAME"].unique()))
            print(f"Loaded {len(all_medicine_names)} unique medicine names.")
        except Exception as e:
            print(f"Error loading medicine names from {medicine_names_csv_path}: {e}")
            all_medicine_names = []


# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if ocr_model is None: # Changed 'model' to 'ocr_model' for consistency
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500
    # Allow prediction even if medicine names list is empty, but warn
    if not all_medicine_names:
        print("Warning: Medicine names list is empty. Closest match will not be effective.")


    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img_np is None:
            return jsonify({"error": "Could not decode image"}), 400

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            cv2.imwrite(temp_file.name, img_np)
            temp_image_path = Path(temp_file.name)

        from scripts.pipeline_extract import segment_words
        word_images = segment_words(temp_image_path)
        temp_image_path.unlink(missing_ok=True)

        results = []
        if not word_images:
            return jsonify({"message": "No words detected in the image."})

        for i, word_img_np in enumerate(word_images):
            try:
                pil_word_img = Image.fromarray(word_img_np).convert("L")
                
                image_tensor = preprocess_image(pil_word_img)
                image_tensor = image_tensor.to(device)

                with torch.no_grad():
                    output = ocr_model(image_tensor)

                log_probs = output.permute(1, 0, 2).log_softmax(2)
                
                # Use 'characters' directly (from config.py) for decoding
                characters_list = list(characters) 
                predicted_word = mltu_ctc_decoder(log_probs.cpu().numpy(), characters_list)[0]
                
                closest_match = find_closest_match(predicted_word, all_medicine_names)
                
                results.append({
                    "original_word_idx": i,
                    "predicted_text": predicted_word,
                    "closest_match": closest_match
                })

            except Exception as e:
                print(f"Error processing word image {i}: {e}")
                results.append({
                    "original_word_idx": i,
                    "predicted_text": f"Error: {e}",
                    "closest_match": "N/A"
                })

        return jsonify({"results": results})

    except Exception as e:
        print(f"Overall error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# --- Main Application Runner ---
if __name__ == '__main__':
    print("Starting Flask application...")
    # Call the resource loading function directly here
    load_resources()

    app.run(debug=True, port=5000)