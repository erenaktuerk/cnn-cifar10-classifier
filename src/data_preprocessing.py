"""
data_preprocessing.py

This module handles data loading, preprocessing, and saving of CIFAR-10 images to disk.
It downloads the CIFAR-10 dataset, normalizes the image pixel values,
and organizes the images into class-based directories for both training and test sets.
"""

import os
import numpy as np
from tensorflow.keras.datasets import cifar10
from PIL import Image

# Dictionary mapping CIFAR-10 numeric labels to human-readable class names.
CLASS_NAMES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

def load_and_preprocess_data():
    """
    Loads the CIFAR-10 dataset from Keras and preprocesses it by normalizing pixel values.
    
    Returns:
        tuple: Two tuples containing:
            - Training data (x_train, y_train)
            - Test data (x_test, y_test)
    """
    # Load the CIFAR-10 dataset (download if not already present)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize the image pixel values to the range [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    return (x_train, y_train), (x_test, y_test)

def save_cifar10_to_disk():
    """
    Downloads the CIFAR-10 dataset and saves the images to disk in a structured directory format.
    
    Images are saved in the following structure:
        data/cifar10/train/<class_name>/<index>.png
        data/cifar10/test/<class_name>/<index>.png
    
    This function is useful for visual inspection and further processing of the dataset.
    """
    # Load the CIFAR-10 dataset (images are in uint8 format before normalization)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Define the base directory for storing images
    base_dir = os.path.join(os.getcwd(), "data", "cifar10")
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    # Create the base directories if they do not exist
    for directory in [train_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)

    # For each class, create a subdirectory in both the train and test folders
    for label, class_name in CLASS_NAMES.items():
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Save training images
    for idx, (img, label) in enumerate(zip(x_train, y_train)):
        class_name = CLASS_NAMES[int(label)]
        file_path = os.path.join(train_dir, class_name, f"{idx}.png")
        # Convert normalized image (0-1) back to 0-255 uint8 format
        img_uint8 = (img * 255).astype(np.uint8)
        im = Image.fromarray(img_uint8)
        im.save(file_path)

    # Save test images
    for idx, (img, label) in enumerate(zip(x_test, y_test)):
        class_name = CLASS_NAMES[int(label)]
        file_path = os.path.join(test_dir, class_name, f"{idx}.png")
        img_uint8 = (img * 255).astype(np.uint8)
        im = Image.fromarray(img_uint8)
        im.save(file_path)

    print("CIFAR-10 images saved to disk in the 'data/cifar10/' directory.")