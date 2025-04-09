#!/usr/bin/env python3
import cv2
import argparse
import csv
import math
import numpy as np
import random

def check_split_value_range(val):
    try:
        float_val = float(val)
        if float_val < 0 or float_val > 1:
            raise argparse.ArgumentTypeError(f"Invalid split ratio: {val}. Must be in [0, 1]!")
        return float_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float: '{val}'")

def check_k_value(val):
    try:
        int_val = int(val)
        if float(val) != int_val:
            raise argparse.ArgumentTypeError(f"K must be an integer, got float: '{val}'")
        if int_val % 2 == 0 or int_val < 1:
            raise argparse.ArgumentTypeError(f"K must be a positive odd integer, got: '{val}'")
        return int_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer: '{val}'")

def load_and_split_data(data_path, split_ratio):
    with open(data_path + 'labels.txt', 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    random.shuffle(lines)
    train_lines = lines[:math.floor(len(lines)*split_ratio)]
    test_lines = lines[math.floor(len(lines)*split_ratio):]
    return train_lines, test_lines

def preprocess_image(img, size=(25, 33)):
    """
    Preprocesses the image:
    - Resize to fixed size
    - Histogram equalization on each channel
    - Flatten to 1D float array
    """
    resized = cv2.resize(img, size)
    b, g, r = cv2.split(resized)
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    equalized = cv2.merge([b_eq, g_eq, r_eq])
    return equalized.flatten().astype(np.float32)

def train_model(data_path, train_lines, image_type, model_filename, save_model):
    train_data = []
    for img_name, _ in train_lines:
        img = cv2.imread(data_path + img_name + image_type)
        processed = preprocess_image(img)
        train_data.append(processed)
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
        cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Preprocessed", cv2.WINDOW_AUTOSIZE)

    correct = 0.0
    confusion_matrix = np.zeros((6, 6))

    for img_name, label in test_lines:
        orig_img = cv2.imread(data_path + img_name + image_type)
        processed = preprocess_image(orig_img).reshape(1, -1)

        if show_img:
            cv2.imshow("Original Image", orig_img)
            reshaped = processed.reshape(33, 25, 3).astype(np.uint8)
            cv2.imshow("Preprocessed", reshaped)
            key = cv2.waitKey()
            if key == 27:  # ESC to quit
                break

        true_label = int(label)
        ret, results, neighbours, dist = knn_model.findNearest(processed, knn_value)
        pred_label = int(ret)

        if true_label == pred_label:
            print(f"{img_name} ✅ Correct, Predicted: {pred_label}")
            correct += 1
        else:
            print(f"{img_name} ❌ Wrong, True: {true_label}, Predicted: {pred_label}")
            print(f"\tNeighbors: {neighbours.flatten()}")
            print(f"\tDistances: {dist.flatten()}")

        confusion_matrix[true_label][pred_label] += 1

    print("\n📊 Total Accuracy:", correct / len(test_lines))
    print("🔢 Confusion Matrix:")
    print(confusion_matrix)

def main():
    parser = argparse.ArgumentParser(description="Train & test KNN image classifier")
    parser.add_argument("-p", "--data_path", type=str, required=True,
                        help="Path to dataset (must include labels.txt and images)")
    parser.add_argument("-r", "--data_split_ratio", type=check_split_value_range,
                        default=0.5, help="Train/test split ratio (0-1)")
    parser.add_argument("-k", "--knn-value", type=check_k_value,
                        default=3, help="K value for KNN (odd int)")
    parser.add_argument("-i", "--image_type", type=str,
                        default=".png", help="Image file extension")
    parser.add_argument("-s", "--save_model_bool", action='store_true',
                        help="Save trained KNN model as XML")
    parser.add_argument("-n", "--model_filename", type=str,
                        default="knn_model", help="Filename for saved model")
    parser.add_argument("-t", "--dont_test_model_bool", action='store_false',
                        help="Skip model testing (train only)")
    parser.add_argument("-d", "--show_img", action='store_true',
                        help="Show images during classification")

    args = parser.parse_args()

    train_lines, test_lines = load_and_split_data(args.data_path, args.data_split_ratio)
    knn_model = train_model(args.data_path, train_lines, args.image_type, args.model_filename, args.save_model_bool)
    if args.dont_test_model_bool:
        test_model(args.data_path, test_lines, args.image_type, knn_model, args.knn_value, args.show_img)

if __name__ == "__main__":
    main()
