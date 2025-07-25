import pandas as pd
from pathlib import Path

# Set paths
base_path = Path("dataset/doctors-handwritten-prescription-bd-dataset")
target_medicines = [
    'Aceta',
    'Fixal',
    'Disopan',
    'Telfast',
    'Sergel'
]

def filter_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    filtered = df[df["MEDICINE_NAME"].isin(target_medicines)].copy()
    filtered.to_csv(output_csv, index=False)
    print(f"{output_csv.name} → {len(filtered)} rows")

filter_csv(base_path / "Training/training_labels.csv", base_path / "Filtered/filtered_training_labels.csv")
filter_csv(base_path / "Validation/validation_labels.csv", base_path / "Filtered/filtered_validation_labels.csv")
filter_csv(base_path / "Testing/testing_labels.csv", base_path / "Filtered/filtered_testing_labels.csv")
