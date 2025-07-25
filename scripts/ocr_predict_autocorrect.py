import os
import csv
import torch
from pathlib import Path
from model_utils import load_model, preprocess_image, ctc_decode
from config import characters, model_path
from rapidfuzz import process

# ✅ Path to label CSV
label_csv_path = Path("dataset/brand_generic_lookup.csv")  # change if needed

# ✅ Load image labels into dictionary {filename: brand}
label_dict = {}
with open(label_csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label_dict[row["image"].strip()] = row["brand"].strip()

# ✅ Known brand names (from CSV or manual list)
known_brands = sorted(set(label_dict.values()))

# ✅ Load model
model = load_model()

# ✅ Prediction input and output
image_dir = Path("dataset/doctors-handwritten-prescription-bd-dataset/Testing/testing_words")
output_file = Path("output/predictions.csv")
output_file.parent.mkdir(exist_ok=True, parents=True)

# ✅ Prepare output
rows = [("image", "predicted", "corrected", "label")]
correct = 0
total = 0

# ✅ Iterate over images
for image_path in sorted(image_dir.glob("*.png")):
    try:
        image_name = image_path.name
        label = label_dict.get(image_name, "").strip()

        if not label:
            print(f"⚠️ Skipping {image_name}: No label found in CSV.")
            continue

        image_tensor = preprocess_image(image_path)
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))  # [1, T, C]
        prediction = ctc_decode(output, characters)[0]

        # ✅ Fuzzy match
        corrected_prediction, score, _ = process.extractOne(prediction, known_brands, score_cutoff=60)
        corrected_prediction = corrected_prediction if corrected_prediction else prediction

        # ✅ Compare and store
        if corrected_prediction.lower() == label.lower():
            correct += 1
        total += 1

        print(f"{image_name}: Raw = {prediction}, Corrected = {corrected_prediction}, Label = {label}")
        rows.append((image_name, prediction, corrected_prediction, label))

    except Exception as e:
        print(f"⚠️ Error processing {image_path.name}: {e}")

# ✅ Save predictions
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

accuracy = (correct / total) * 100 if total else 0
print(f"\n✅ Prediction complete — Accuracy: {accuracy:.2f}% on {total} samples")
print(f"📄 Results saved to {output_file}")
