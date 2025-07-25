# scripts/pipeline_extract.py

import cv2
import torch
from pathlib import Path
from .model_utils import load_model, preprocess_image, ctc_decode
from .config import target_medicines, characters

def segment_words(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))

    # Threshold + dilation to segment words
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_images = []

    for i, cnt in enumerate(sorted(contours, key=lambda c: cv2.boundingRect(c)[1])):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 10:
            word_img = img[y:y+h, x:x+w]
            word_images.append(word_img)
    return word_images

def predict_word(model, img_array):
    # Convert numpy image to tensor
    from PIL import Image
    import numpy as np

    pil_img = Image.fromarray(img_array)
    tensor = preprocess_image(pil_img)

    with torch.no_grad():
        output = model(tensor)
        prediction = ctc_decode(output, characters)[0]
    return prediction

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to full prescription image")
    args = parser.parse_args()

    model = load_model()
    words = segment_words(Path(args.image))

    found = []
    for img in words:
        text = predict_word(model, img).strip()
        for med in target_medicines:
            if med.lower() in text.lower():
                found.append(med)
                break

    if found:
        print("✅ Medicines found:")
        for med in set(found):
            print("-", med)
    else:
        print("❌ No target medicines detected.")
