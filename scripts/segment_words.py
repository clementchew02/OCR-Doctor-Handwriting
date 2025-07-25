# scripts/1_segment_words.py

import cv2
import os
import argparse
import numpy as np
from pathlib import Path

def segment_image(image_path, output_dir):
    # Load and preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (800, int(image.shape[0] * (800 / image.shape[1]))))  # scale width

    # Threshold to binary image (invert: white text on black background)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological dilation to connect characters into words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find contours (each contour is a word-ish segment)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  # sort by y-axis (top to bottom)

    os.makedirs(output_dir, exist_ok=True)
    word_images = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        if w < 30 or h < 10:
            continue  # Skip small noise

        word_img = image[y:y+h, x:x+w]
        filename = f"{Path(image_path).stem}_word_{i}.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, word_img)
        word_images.append(save_path)

    print(f"Extracted {len(word_images)} words from {image_path}")
    return word_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to full prescription image")
    parser.add_argument("--output", default="output/words", help="Directory to save word crops")
    args = parser.parse_args()

    segment_image(args.input, args.output)
