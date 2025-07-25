import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
import cv2

# Use relative import for config
from .config import image_height, image_width, characters, model_path

# --- CRNN Model Definition (Aligned with 1-channel Grayscale input) ---
class CRNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CRNNModel, self).__init__()
        # CNN layers
        # IMPORTANT: input_channel is now 1 for Grayscale (L) images
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 16 -> 8

            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,1), padding=(0,1)), # 8 -> 4 (height), width is pooled with stride 1

            nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,1), padding=(0,1)), # 4 -> 2 (height), width is pooled with stride 1

            nn.Conv2d(512, 512, kernel_size=(2,2), stride=(1,1)), # 2 -> 1 (height)
            nn.ReLU(True)
        )
        
        # RNN (LSTM) layers
        # Input features to LSTM will be 512 (channels) * 1 (final feature map height)
        self.rnn = nn.Sequential(
            nn.LSTM(512 * 1, 256, bidirectional=True, num_layers=2), # Input to LSTM is (seq_len, batch_size, feature_size)
            nn.Linear(256 * 2, num_classes) # Output of bidirectional LSTM is 2*hidden_size
        )

    def forward(self, input):
        # CNN features
        conv_features = self.cnn(input) # Output shape: (B, C, H, W) e.g., (B, 512, 1, W_prime)
        
        # Reshape for RNN: (W_prime, B, C*H)
        # Squeeze the height dimension (H), which should be 1 after CNN
        conv_features = conv_features.squeeze(2) # Output shape: (B, C, W_prime)
        
        # Permute to (sequence_length, batch_size, input_size) for LSTM
        conv_features = conv_features.permute(2, 0, 1) 

        # RNN features
        output = self.rnn(conv_features) # Output shape: (sequence_length, batch_size, num_classes)
        return output

# --- Preprocessing Function (Aligned with 1-channel Grayscale input) ---
def preprocess_image(image_input):
    """
    Preprocesses an image for the model.
    Accepts either a file path (str or Path) or a PIL Image object.
    Ensures output is 1-channel (Grayscale) and normalized to [-1, 1].
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

    # IMPORTANT: Convert to Grayscale (L mode)
    img_pil = img_pil.convert('L')

    # Convert PIL Image to NumPy array
    img_np = np.array(img_pil, dtype=np.float32)

    # Resize using OpenCV (more control over interpolation)
    # cv2.resize expects (width, height)
    img_resized = cv2.resize(img_np, (image_width, image_height))

    # Normalize pixel values to [0, 1]
    img_normalized = img_resized / 255.0

    # Normalize to [-1, 1] for typical CNN input
    img_normalized = (img_normalized - 0.5) / 0.5

    # Convert NumPy array to PyTorch tensor
    # For grayscale, the shape is (H, W), unsqueeze to (1, H, W) for channel dim
    image_tensor = torch.from_numpy(img_normalized).unsqueeze(0) 

    # Add a batch dimension (1, 1, H, W) for single image inference
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

# --- CTC Greedy Decode Function ---
def ctc_decode(output_tensor: torch.Tensor, characters: list) -> list:
    """
    Decodes the CTC output tensor to a list of predicted texts.

    Args:
        output_tensor (torch.Tensor): Model's output tensor (batch_size, sequence_length, num_classes).
        characters (list): List of characters (vocabulary) used in training.

    Returns:
        list: List of decoded strings.
    """
    # Permute for CTC decoding: (sequence_length, batch_size, num_classes)
    log_probs = output_tensor.permute(1, 0, 2).log_softmax(2)

    # Use mltu's ctc_decoder for robust decoding (expects numpy array for log_probs)
    from mltu.utils.text_utils import ctc_decoder as mltu_ctc_decoder
    decoded_texts = mltu_ctc_decoder(log_probs.cpu().numpy(), characters)
    
    return decoded_texts

# --- Load Model Function ---
def load_model():
    """
    Loads the trained PyTorch model for inference.
    Uses global variables from config.py for model parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the CRNNModel architecture with 1 input channel
    model = CRNNModel(num_classes=len(characters) + 1).to(device)

    # Load the trained weights
    try:
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval() # Set model to evaluation mode
            print(f"Model loaded successfully from {model_path} on {device}")
        else:
            print(f"Warning: Model file not found at {model_path}. Initializing with random weights.")
            print("Please ensure your pre-trained 'crnn_model.pth' is in the 'models' folder.")
            model = None # Indicate model not loaded if file missing
    except Exception as e:
        print(f"Error loading model state dictionary from {model_path}: {e}")
        model = None # Indicate model not loaded if error occurs
    
    return model