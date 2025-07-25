# MediWrite: Transforming Doctors' Handwriting to Text - OCR for Medicine Names

## Project Overview

**MediWrite** is an Optical Character Recognition (OCR) system specifically designed to interpret handwritten medicine names from prescription images. Leveraging deep learning techniques, this project aims to address the common challenge of illegible handwriting in medical prescriptions by converting handwritten text into digital, searchable data. The system employs a Convolutional Recurrent Neural Network (CRNN) model, integrated with robust image preprocessing and a post-OCR closest-match algorithm for enhanced accuracy in identifying specific medicine names.

This repository contains the necessary scripts for dataset preparation, model training, evaluation, and deploying the OCR system as a user-friendly Flask web application.

## Features

* **Handwritten OCR:** Utilizes a CRNN model to recognize handwritten text from images.
* **Word Segmentation:** Automatically segments full prescription images into individual word images for precise OCR.
* **Medicine Name Matching:** Implements a Levenshtein distance-based algorithm to find the closest match to predicted OCR text from a predefined list of medicine names.
* **Web Application Interface:** A simple Flask web interface for uploading prescription images and receiving OCR predictions.
* **Modular Design:** Separate scripts for data filtering, model definition, training, prediction, and web deployment.

## Technologies Used

* **Python 3.x**
* **PyTorch:** Deep learning framework for model definition, training, and inference.
* **torchvision:** For image transformations and dataset handling.
* **OpenCV (`cv2`):** For image processing tasks like word segmentation and resizing.
* **Pillow (`PIL`):** For image manipulation.
* **Pandas:** For data handling and CSV operations.
* **Levenshtein:** For calculating string similarity (closest match).
* **Flask:** Web framework for the application's backend.
* **HTML/CSS/JavaScript:** For the web application's frontend.

## Project Structure
.
app.py                      # Flask web application for OCR inference
models/                     # Directory to store the trained CRNN model (.pth)
    -crnn_model.pth          # The trained OCR model
scripts/
    -config.py               # Global configurations (image dimensions, characters, model path, target medicines)
    -filter_dataset.py       # Script to filter the raw dataset for specific medicine names
    -model_utils.py          # Defines the CRNNModel architecture and utility functions (preprocess, decode)
    -ocr_predict.py          # Script for evaluating the model on a test dataset and generating predictions.csv
    -pipeline_extract.py     # Contains functions for segmenting words from full images and predicting them
    -rain.py                # Script for training the CRNN model
dataset/
    -doctors-handwritten-prescription-bd-dataset/
        -Training/           # Original training images and labels
        -Validation/         # Original validation images and labels
        -Testing/            # Original testing images and labels
        -Filtered/           # Filtered CSVs of labels (created by filter_dataset.py)
            -filtered_training_labels.csv
            -filtered_validation_labels.csv
            -filtered_testing_labels.csv
templates/
    -index.html              # Frontend HTML for the Flask app
requirements.txt            # List of Python dependencies
README.md                   # This file
## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd MediWrite-OCR
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Create `requirements.txt` if you don't have one, by running `pip freeze > requirements.txt` after installing all your libraries like `torch`, `pandas`, `opencv-python`, `Pillow`, `Levenshtein`, `Flask`).

4.  **Download the Dataset:**
    * Obtain the "Doctors' Handwritten Prescription BD Dataset" or a similar dataset of handwritten medicine names.
    * Place it in the `dataset/` directory as shown in the `Project Structure` above. The expected structure is `dataset/doctors-handwritten-prescription-bd-dataset/Training`, `Validation`, `Testing`.

## Usage

### 1. Prepare the Dataset

First, filter the dataset to include only the `target_medicines` defined in `config.py`.

```bash
python scripts/filter_dataset.py
This will create filtered_training_labels.csv, filtered_validation_labels.csv, and filtered_testing_labels.csv in the dataset/doctors-handwritten-prescription-bd-dataset/Filtered/ directory.

2. Train the Model
Train the CRNN model using the prepared training data.

Bash

python scripts/train.py
The trained model (crnn_model.pth) will be saved to the models/ directory. Adjust num_epochs in train.py for longer training.

3. Evaluate the Model (Optional)
Run the ocr_predict.py script to evaluate the trained model's performance on the test set and generate a predictions.csv file.

Bash

python scripts/ocr_predict.py
4. Run the Web Application
Start the Flask development server:

Bash

python app.py
The application will typically run on http://127.0.0.1:5000/. Open this URL in your web browser.

Using the Web App:
Upload an image of a handwritten prescription.

Click "Predict OCR" to get the segmented words, their OCR predictions, and the closest matching medicine names from your predefined list.

5. Test Word Segmentation (Optional)
You can test the word segmentation pipeline on a single image:

Bash

python scripts/pipeline_extract.py --image <path_to_your_prescription_image.png>
This script will output predictions for words segmented from the provided image.

Customization
config.py: Modify image_height, image_width, and target_medicines to suit your dataset or specific OCR requirements. characters are automatically generated from target_medicines.

Dataset Paths: Ensure paths in filter_dataset.py, train.py, ocr_predict.py, and app.py correctly point to your dataset location.

Model Architecture: The CRNNModel in scripts/model_utils.py can be adjusted for more complex CNN/RNN layers if needed, but ensure consistency between training and inference models.

Contributing
Feel free to fork this repository, submit issues, or propose pull requests.

License
[Specify your license here, e.g., MIT, Apache 2.0, etc.]

Acknowledgements
"Doctors' Handwritten Prescription BD Dataset" (if publicly available and used)

PyTorch community for excellent deep learning framework.

OpenCV for image processing capabilities.