# scripts/train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from config import image_height, image_width, characters, model_path
from model_utils import CRNNModel

# 🧠 Dataset for medicine word images
class MedicineDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.char_to_idx = {c: i + 1 for i, c in enumerate(characters)}  # +1 because CTC blank is 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.image_dir / row["IMAGE"]
        label = row["MEDICINE_NAME"]

        # Open image and preprocess
        img = Image.open(img_path).convert("L").resize((image_width, image_height))
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        img = torch.tensor(img).unsqueeze(0)  # [1, H, W]

        # Encode label
        label_indices = torch.tensor([self.char_to_idx[c] for c in label if c in self.char_to_idx], dtype=torch.long)
        return img, label_indices

# 🔁 Collate function for CTC
def collate_fn(batch):
    images, labels = zip(*batch)
    image_batch = torch.stack(images)
    label_batch = torch.cat(labels)
    label_lengths = torch.tensor([len(l) for l in labels])
    return image_batch, label_batch, label_lengths

# 🏋️‍♂️ Train function
def train():
    print("🧪 Loading dataset...")
    data_root = Path("dataset/doctors-handwritten-prescription-bd-dataset")
    train_csv = data_root / "Filtered/filtered_training_labels.csv"
    val_csv = data_root / "Filtered/filtered_validation_labels.csv"
    train_img_dir = data_root / "Training/training_words"
    val_img_dir = data_root / "Validation/validation_words"

    # Load datasets
    train_ds = MedicineDataset(train_csv, train_img_dir)
    val_ds = MedicineDataset(val_csv, val_img_dir)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Init model + loss + optimizer
    model = CRNNModel(num_classes=len(characters) + 1)  # +1 for CTC blank
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 3000
    print("🚀 Starting training...\n")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, labels, label_lengths in train_loader:
            optimizer.zero_grad()
            outputs = model(images)  # [T, B, C]
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"📘 Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.4f}")

    # Save model
    model_out = Path(model_path)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"\n✅ Model saved to {model_out}")

if __name__ == "__main__":
    train()
