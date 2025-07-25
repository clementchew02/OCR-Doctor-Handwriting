# scripts/ocr_predict.py

import torch
import pandas as pd
from pathlib import Path
from model_utils import load_model, preprocess_image, ctc_decode
from config import characters

# Load filtered test set
csv_path = Path("dataset/doctors-handwritten-prescription-bd-dataset/Filtered/filtered_testing_labels.csv")
image_folder = csv_path.parent.parent / "Testing/testing_words"

# Load model
model = load_model()
model.eval()

# Read dataset
df = pd.read_csv(csv_path)
df["predicted"] = ""

correct = 0

for idx, row in df.iterrows():
    image_path = image_folder / row["IMAGE"]
    label = row["MEDICINE_NAME"]  # 🔁 make sure this matches your CSV header exactly

    if not image_path.exists():
        print(f"Missing file: {image_path}")
        continue

    try:
        image_tensor = preprocess_image(image_path)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        continue

    with torch.no_grad():
        output = model(image_tensor)
        prediction = ctc_decode(output, characters)[0]

    df.at[idx, "predicted"] = prediction

    if prediction.strip().lower() == label.strip().lower():
        correct += 1

accuracy = correct / len(df) * 100
print(f"\n✅ Prediction complete — Accuracy: {accuracy:.2f}% on {len(df)} samples")

# Save results
output_path = Path("output/predictions.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"📄 Results saved to {output_path}")
