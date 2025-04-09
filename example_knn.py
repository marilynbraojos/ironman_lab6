#!/usr/bin/env python3
import cv2
import argparse
import csv
import math
import numpy as np
import random
import os

def check_split_value_range(val):
    try:
        float_val = float(val)
        if float_val < 0 or float_val > 1:
            raise argparse.ArgumentTypeError("Received data split ratio of %s which is an invalid value. The input ratio must be in range [0, 1]!" % float_val)
        return float_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Received '{val}' which is not a valid float!")

def check_k_value(val):
    try:
        int_val = int(val)
        if float(val) != int_val:
            raise argparse.ArgumentTypeError(f"Received '{val}' which is a float not an integer.")
        if int_val % 2 == 0 or int_val < 1:
            raise argparse.ArgumentTypeError(f"Received '{val}' which is not a positive, odd integer.")
        return int_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Received '{val}' which is not a valid integer!")

def load_and_split_data(data_path, split_ratio):
    with open(os.path.join(data_path, 'labels.txt'), 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
    random.shuffle(lines)
    split = math.floor(len(lines) * split_ratio)
    return lines[:split], lines[split:]

def crop_center(img, cropx=160, cropy=100):
    y, x, _ = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx]

def preprocess_and_flatten(img, size=(25, 33)):
    cropped = crop_center(img, 100, 100)
    resized = cv2.resize(cropped, size)
    denoised = cv2.bilateralFilter(resized, 9, 75, 75)
    normalized = denoised.astype(np.float32) / 255.0
    return normalized.flatten()

def train_model(data_path, train_lines, image_type, model_filename, save_model):
    train_data = []
    for img_name, _ in train_lines:
        img = cv2.imread(os.path.join(data_path, img_name + image_type))
        if img is not None:
            train_data.append(preprocess_and_flatten(img))
    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array([int(label) for _, label in train_lines], dtype=np.int32)

    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    print("✅ KNN model trained!")

    if save_model:
        knn.save(model_filename + '.xml')
        print(f"📁 Model saved to {model_filename}.xml")
    return knn

def test_model(data_path, test_lines, image_type, knn_model, knn_value, show_img):
    if show_img:
        cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Processed", cv2.WINDOW_AUTOSIZE)

    correct = 0
    confusion_matrix = np.zeros((6, 6), dtype=np.int32)

    for img_name, label in test_lines:
        img_path = os.path.join(data_path, img_name + image_type)
        img = cv2.imread(img_path)
        if img is None:
            continue
        processed = preprocess_and_flatten(img).reshape(1, -1)
        true_label = int(label)
        ret, _, _, _ = knn_model.findNearest(processed, knn_value)
        pred_label = int(ret)

        if pred_label == true_label:
            correct += 1
        confusion_matrix[true_label][pred_label] += 1

        if show_img:
            cv2.imshow("Original", img)
            img_show = (processed.reshape(33, 25, 3) * 255).astype(np.uint8)
            cv2.imshow("Processed", img_show)
            if cv2.waitKey(0) == 27:
                break

        print(f"{img_name}: True={true_label}, Pred={pred_label}")

    print("\n📊 Accuracy:", correct / len(test_lines))
    print("Confusion Matrix:")
    print(confusion_matrix)

def main():
    parser = argparse.ArgumentParser(description="Train and test a KNN classifier for image signs")
    parser.add_argument("-p", "--data_path", required=True)
    parser.add_argument("-r", "--data_split_ratio", type=check_split_value_range, default=0.5)
    parser.add_argument("-k", "--knn-value", type=check_k_value, default=3)
    parser.add_argument("-i", "--image_type", default=".png")
    parser.add_argument("-s", "--save_model_bool", action="store_true")
    parser.add_argument("-n", "--model_filename", default="knn_model")
    parser.add_argument("-t", "--dont_test_model_bool", action="store_false")
    parser.add_argument("-d", "--show_img", action="store_true")
    args = parser.parse_args()

    train_lines, test_lines = load_and_split_data(args.data_path, args.data_split_ratio)
    knn_model = train_model(args.data_path, train_lines, args.image_type, args.model_filename, args.save_model_bool)
    if args.dont_test_model_bool:
        test_model(args.data_path, test_lines, args.image_type, knn_model, args.knn_value, args.show_img)

if __name__ == "__main__":
    main()
