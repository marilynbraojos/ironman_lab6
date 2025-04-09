#!/usr/bin/env python3
import os
import argparse
import csv
import cv2
import numpy as np

def crop_center(img, cropx=100, cropy=100):
    y, x, _ = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx]

def preprocess_and_flatten(img, size=(25, 33)):
    cropped = crop_center(img, 100, 100)
    resized = cv2.resize(cropped, size)
    denoised = cv2.bilateralFilter(resized, 9, 75, 75)
    normalized = denoised.astype(np.float32) / 255.0
    return normalized.flatten().reshape(1, -1)

def initialize_model(model_path=None):
    if model_path and os.path.exists(model_path):
        model = cv2.ml.KNearest_create()
        model = model.load(model_path)
        return model
    raise ValueError("❌ Model file not found or not specified!")

def predict(model, image):
    if image is None:
        raise ValueError("Input image is None")
    processed = preprocess_and_flatten(image)
    ret, _, _, _ = model.findNearest(processed, k=3)
    return int(ret)

def load_validation_data(data_path):
    labels_file = os.path.join(data_path, "labels.txt")
    data = []
    with open(labels_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            image_file = os.path.join(data_path, row[0] + ".png")
            data.append((image_file, int(row[1])))
    return data

def evaluate_model(model, validation_data):
    num_classes = 6
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    correct = 0

    for image_path, true_label in validation_data:
        image = cv2.imread(image_path)
        if image is None:
            print("⚠️ Could not load image:", image_path)
            continue
        pred_label = predict(model, image)
        if pred_label == true_label:
            correct += 1
        confusion_matrix[true_label][pred_label] += 1
        print(f"{os.path.basename(image_path)} - True: {true_label}, Pred: {pred_label}")

    accuracy = correct / len(validation_data)
    print("\n✅ Accuracy:", accuracy)
    print("📊 Confusion Matrix:")
    print(confusion_matrix)

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained KNN model")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    data = load_validation_data(args.data_path)
    model = initialize_model(args.model_path)
    evaluate_model(model, data)

if __name__ == "__main__":
    main()
