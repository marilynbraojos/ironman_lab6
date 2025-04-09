#!/usr/bin/env python3
import cv2 # opencv for img processing
import argparse # command-line args
import csv # reads txt and csv files 
import math # math ops 
import pickle # saving/loading 
import numpy as np 
import random

def check_split_value_range(val): # validating command line inputs 
    try: 
        float_val = float(val) # convert value to float 
        if float_val < 0 or float_val > 1: # if below 0 or above 1 
            raise argparse.ArgumentTypeError("Received data split ratio of %s which is an invalid value. The input ratio must be in range [0, 1]!" % float_val) # received incorrect ratio for split
        return float_val # return the float value 
    except ValueError: # if input wasn't a float, return error
        raise argparse.ArgumentTypeError(f"Received '{val}' which is not a valid float!")

def check_k_value(val): # validate that the k NN is + and odd
    try:
        int_val = int(val) # data type = integer 
        if float(val) != int_val:
            raise argparse.ArgumentTypeError(f"Received '{val}' which is a float not an integer. The KNN value input must be an integer!")
        if int_val % 2 == 0 or int_val < 1: # if even or less than 1 
            raise argparse.ArgumentTypeError(f"Received '{val}' which not a positive, odd integer. The KNN value input must be a postive, odd integer!")
        return int_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Received '{val}' which is not a valid integer!")

def load_and_split_data(data_path, split_ratio):
    """
    Uses the provided labels.txt file to split the data into training and testing sets.

    Args:
        data_path (str): Path to the dataset. 
        split_ratio (float): must be a float between 0 and 1. Split ratio will be used to split the data into training and testing sets. 
                             split_ratio of the data will be used for training and (1-split_ratio) will be used for testing. 
                             For example if split ratio was 0.7, 70% of the data will be used for training and the remaining 30% will be used for testing.

    Returns:
        list of tuples for testing and training (image_path, true_label)
    """

    with open(data_path + 'labels.txt', 'r') as f: # open labels file 
        reader = csv.reader(f) # create CSV reader
        lines = list(reader) # convert file to list of [image, labels]

    #Randomly choose train and test data (50/50 split).
    random.shuffle(lines) # shuffle data randomly
    train_lines = lines[:math.floor(len(lines)*split_ratio)][:] # use first N lines for training
    test_lines = lines[math.floor(len(lines)*split_ratio):][:] # use remanining lines for testing 

    return train_lines, test_lines # return train/test

def train_model(data_path, train_lines, image_type, model_filename, save_model):
    """
    Loads the images from the training set and uses them to create a KNN model.
    The images and labels must be in the given directoy.

    Args:
        data_path (str): Path to the dataset.
        train_lines (tuple): Tuple of the training data containing (image_number, true_label)
        image_type (str): Image extension to load (e.g. .png, .jpg, .jpeg)

    Returns:
        knn (knn_model_object): The KNN model.
    """

    #This line reads in all images listed in the file in color, and resizes them to 25x33 pixels
    train = np.array([np.array(cv2.resize(cv2.imread(data_path+train_lines[i][0]+image_type),(25,33))) for i in range(len(train_lines))])

    #Here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants), note the *3 is due to 3 channels of color.
    train_data = train.flatten().reshape(len(train_lines), 33*25*3)
    train_data = train_data.astype(np.float32)

    #Read in training labels
    train_labels = np.array([np.int32(train_lines[i][1]) for i in range(len(train_lines))])

    ### Train classifier
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    print("KNN model created!")

    if(save_model):
        # Save the trained model
        knn.save(model_filename + '.xml')

        print(f"KNN model saved to {model_filename}.xml")
    
    return knn

def test_model(data_path, test_lines, image_type, knn_model, knn_value, show_img):
    """
    Loads the images and tests the provided KNN model prediction with the dataset label.
    The images and labels must be in the given directoy.

    Args:
        data_path (str): Path to the dataset.
        test_lines (tuple): Tuple of the training data containing (image_number, true_label)
        image_type (str): Image extension to load (e.g. .png, .jpg, .jpeg)
        knn_model (model object): The knn model
        knn_value (int): The number of KNN neighbors to consider when classifying
        show_img: A boolean whether to show images as they are processed or not

    Returns:
        knn (knn_model_object): The KNN model.
    """

    if(show_img):
        Title_images = 'Original Image'
        Title_resized = 'Image Resized'
        cv2.namedWindow( Title_images, cv2.WINDOW_AUTOSIZE )

    correct = 0.0
    confusion_matrix = np.zeros((6,6))

    k = knn_value

    for i in range(len(test_lines)):  # for each test image 
        original_img = cv2.imread(data_path+test_lines[i][0]+image_type) # load the full image
        test_img = np.array(cv2.resize(cv2.imread(data_path+test_lines[i][0]+image_type),(25,33))) # resize image 
        if(show_img):
            cv2.imshow(Title_images, original_img)
            cv2.imshow(Title_resized, test_img)
            key = cv2.waitKey()
            if key==27:    # Esc key to stop
                break
        test_img = test_img.flatten().reshape(1, 33*25*3) # flatten
        test_img = test_img.astype(np.float32) 

        test_label = np.int32(test_lines[i][1])

        ret, results, neighbours, dist = knn_model.findNearest(test_img, k) # use trained model to classify 

        if test_label == ret:
            print(str(test_lines[i][0]) + " Correct, " + str(ret))
            correct += 1
            confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
        else:
            confusion_matrix[test_label][np.int32(ret)] += 1
            
            print(str(test_lines[i][0]) + " Wrong, " + str(test_label) + " classified as " + str(ret))
            print("\tneighbours: " + str(neighbours))
            print("\tdistances: " + str(dist))

    print("\n\nTotal accuracy: " + str(correct/len(test_lines)))
    print(confusion_matrix)

def main():
    parser = argparse.ArgumentParser(description="Example Model Trainer and Tester with Basic KNN for 7785 Lab 6!")
    parser.add_argument("-p","--data_path", type=str, required=True, help="Path to the valid dataset directory (must contain labels.txt and images)")
    parser.add_argument("-r","--data_split_ratio", type=check_split_value_range, required=False, default=0.5, help="Ratio of the train, test split. Must be a float between 0 and 1. The number entered is the percentage of data used for training, the remaining is used for testing!")
    parser.add_argument("-k","--knn-value", type=check_k_value, required=False, default=3, help="KNN value. Must be an odd integer greater than zero.")
    parser.add_argument("-i","--image_type", type=str, required=False, default=".png", help="Extension of the image files (e.g. .png, .jpg)")
    parser.add_argument("-s","--save_model_bool", action='store_true', required=False, help="Boolean flag to save the KNN model as an XML file for later use.")
    parser.add_argument("-n","--model_filename", type=str, required=False, default="knn_model", help="Filename of the saved KNN model.")
    parser.add_argument("-t","--dont_test_model_bool", action='store_false', required=False, help="Boolean flag to not test the created KNN model on split testing set (training only).")
    parser.add_argument("-d","--show_img", action='store_true', required=False, help="Boolean flag to show the tested images as they are classified.")


    args = parser.parse_args()

    #Path to dataset directory from command line argument.
    dataset_path = args.data_path

    #Ratio of datasplit from command line argument.
    data_split_ratio = args.data_split_ratio

    #Image type from command line argument.
    image_type = args.image_type

    #Boolean if true will save the KNN model as a XML file from command line argument.
    save_model_bool = args.save_model_bool

    #Filename for the saved KNN model from command line argument.
    model_filename = args.model_filename

    #Boolean if true will test the model on the split testing set based on command line argument.
    test_model_bool = args.dont_test_model_bool

    #Number of neighbors to consider for KNN.
    knn_value = args.knn_value

    #Boolean if true will show the images as they are tested.
    show_img= args.show_img

    train_lines, test_lines = load_and_split_data(dataset_path, data_split_ratio)
    knn_model = train_model(dataset_path, train_lines, image_type, model_filename, save_model_bool)
    if(test_model_bool):
        test_model(dataset_path, test_lines, image_type, knn_model, knn_value, show_img)

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# import cv2
# import argparse
# import csv
# import math
# import numpy as np
# import random

# def check_split_value_range(val):
#     try:
#         float_val = float(val)
#         if float_val < 0 or float_val > 1:
#             raise argparse.ArgumentTypeError(f"Invalid split ratio: {val}. Must be in [0, 1]!")
#         return float_val
#     except ValueError:
#         raise argparse.ArgumentTypeError(f"Invalid float: '{val}'")

# def check_k_value(val):
#     try:
#         int_val = int(val)
#         if float(val) != int_val:
#             raise argparse.ArgumentTypeError(f"K must be an integer, got float: '{val}'")
#         if int_val % 2 == 0 or int_val < 1:
#             raise argparse.ArgumentTypeError(f"K must be a positive odd integer, got: '{val}'")
#         return int_val
#     except ValueError:
#         raise argparse.ArgumentTypeError(f"Invalid integer: '{val}'")

# def load_and_split_data(data_path, split_ratio):
#     with open(data_path + 'labels.txt', 'r') as f:
#         reader = csv.reader(f)
#         lines = list(reader)

#     random.shuffle(lines)
#     train_lines = lines[:math.floor(len(lines)*split_ratio)]
#     test_lines = lines[math.floor(len(lines)*split_ratio):]
#     return train_lines, test_lines

# def preprocess_image(img, size=(25, 33)):
#     """
#     Preprocesses the image:
#     - Resize to fixed size
#     - Histogram equalization on each channel
#     - Flatten to 1D float array
#     """
#     resized = cv2.resize(img, size)
#     b, g, r = cv2.split(resized)
#     b_eq = cv2.equalizeHist(b)
#     g_eq = cv2.equalizeHist(g)
#     r_eq = cv2.equalizeHist(r)
#     equalized = cv2.merge([b_eq, g_eq, r_eq])
#     return equalized.flatten().astype(np.float32)

# def train_model(data_path, train_lines, image_type, model_filename, save_model):
#     train_data = []
#     for img_name, _ in train_lines:
#         img = cv2.imread(data_path + img_name + image_type)
#         processed = preprocess_image(img)
#         train_data.append(processed)
#     train_data = np.array(train_data, dtype=np.float32)

#     train_labels = np.array([int(label) for _, label in train_lines], dtype=np.int32)

#     knn = cv2.ml.KNearest_create()
#     knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

#     print("✅ KNN model trained!")

#     if save_model:
#         knn.save(model_filename + '.xml')
#         print(f"📁 Model saved to {model_filename}.xml")

#     return knn

# def test_model(data_path, test_lines, image_type, knn_model, knn_value, show_img):
#     if show_img:
#         cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
#         cv2.namedWindow("Preprocessed", cv2.WINDOW_AUTOSIZE)

#     correct = 0.0
#     confusion_matrix = np.zeros((6, 6))

#     for img_name, label in test_lines:
#         orig_img = cv2.imread(data_path + img_name + image_type)
#         processed = preprocess_image(orig_img).reshape(1, -1)

#         if show_img:
#             cv2.imshow("Original Image", orig_img)
#             reshaped = processed.reshape(33, 25, 3).astype(np.uint8)
#             cv2.imshow("Preprocessed", reshaped)
#             key = cv2.waitKey()
#             if key == 27:  # ESC to quit
#                 break

#         true_label = int(label)
#         ret, results, neighbours, dist = knn_model.findNearest(processed, knn_value)
#         pred_label = int(ret)

#         if true_label == pred_label:
#             print(f"{img_name} ✅ Correct, Predicted: {pred_label}")
#             correct += 1
#         else:
#             print(f"{img_name} ❌ Wrong, True: {true_label}, Predicted: {pred_label}")
#             print(f"\tNeighbors: {neighbours.flatten()}")
#             print(f"\tDistances: {dist.flatten()}")

#         confusion_matrix[true_label][pred_label] += 1

#     print("\n📊 Total Accuracy:", correct / len(test_lines))
#     print("🔢 Confusion Matrix:")
#     print(confusion_matrix)

# def main():
#     parser = argparse.ArgumentParser(description="Train & test KNN image classifier")
#     parser.add_argument("-p", "--data_path", type=str, required=True,
#                         help="Path to dataset (must include labels.txt and images)")
#     parser.add_argument("-r", "--data_split_ratio", type=check_split_value_range,
#                         default=0.5, help="Train/test split ratio (0-1)")
#     parser.add_argument("-k", "--knn-value", type=check_k_value,
#                         default=3, help="K value for KNN (odd int)")
#     parser.add_argument("-i", "--image_type", type=str,
#                         default=".png", help="Image file extension")
#     parser.add_argument("-s", "--save_model_bool", action='store_true',
#                         help="Save trained KNN model as XML")
#     parser.add_argument("-n", "--model_filename", type=str,
#                         default="knn_model", help="Filename for saved model")
#     parser.add_argument("-t", "--dont_test_model_bool", action='store_false',
#                         help="Skip model testing (train only)")
#     parser.add_argument("-d", "--show_img", action='store_true',
#                         help="Show images during classification")

#     args = parser.parse_args()

#     train_lines, test_lines = load_and_split_data(args.data_path, args.data_split_ratio)
#     knn_model = train_model(args.data_path, train_lines, args.image_type, args.model_filename, args.save_model_bool)
#     if args.dont_test_model_bool:
#         test_model(args.data_path, test_lines, args.image_type, knn_model, args.knn_value, args.show_img)

# if __name__ == "__main__":
#     main()
