import os
import tarfile
import urllib.request
from tqdm import tqdm
import shutil
import random
import numpy as np
from PIL import Image


def download_food101(download_dir):
    """Download and extract the Food-101 dataset."""
    os.makedirs(download_dir, exist_ok=True)

    # URL for Food-101 dataset
    url = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    filename = os.path.join(download_dir, "food-101.tar.gz")

    # Download if the file doesn't exist
    if not os.path.exists(filename):
        print(f"Downloading Food-101 dataset to {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print(f"Found existing download at {filename}")

    # Extract the tarfile
    extract_dir = os.path.join(download_dir, "food-101")
    if not os.path.exists(extract_dir):
        print("Extracting Food-101 dataset...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(download_dir)
        print("Extraction complete.")
    else:
        print(f"Found existing extraction at {extract_dir}")

    return extract_dir


def create_train_val_test_split(food101_dir, output_dir, val_split=0.1, seed=42):
    """
    Create train/val/test directories for Food-101.

    Args:
        food101_dir: Path to extracted Food-101 dataset
        output_dir: Directory to create the split datasets
        val_split: Percentage of training data to use for validation
        seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Path to metadata
    meta_dir = os.path.join(food101_dir, "meta")
    train_file = os.path.join(meta_dir, "train.txt")
    test_file = os.path.join(meta_dir, "test.txt")

    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Read class names
    classes = []
    with open(train_file, 'r') as f:
        for line in f:
            class_name = line.strip().split('/')[0]
            if class_name not in classes:
                classes.append(class_name)

    print(f"Found {len(classes)} classes")

    # Create class directories
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Read train and test file lists
    train_images = []
    with open(train_file, 'r') as f:
        for line in f:
            train_images.append(line.strip())

    test_images = []
    with open(test_file, 'r') as f:
        for line in f:
            test_images.append(line.strip())

    # Split train into train and val
    train_val_images = {}
    for class_name in classes:
        class_images = [img for img in train_images if img.startswith(class_name + '/')]
        random.shuffle(class_images)

        val_count = int(len(class_images) * val_split)
        train_val_images[class_name] = {
            'train': class_images[val_count:],
            'val': class_images[:val_count]
        }

    # Copy images to appropriate directories
    images_dir = os.path.join(food101_dir, "images")

    print("Copying training images...")
    for class_name in tqdm(classes):
        for img_path in train_val_images[class_name]['train']:
            src = os.path.join(images_dir, img_path + '.jpg')
            dst = os.path.join(train_dir, img_path + '.jpg')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

    print("Copying validation images...")
    for class_name in tqdm(classes):
        for img_path in train_val_images[class_name]['val']:
            src = os.path.join(images_dir, img_path + '.jpg')
            dst = os.path.join(val_dir, img_path + '.jpg')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

    print("Copying test images...")
    for img_path in tqdm(test_images):
        src = os.path.join(images_dir, img_path + '.jpg')
        dst = os.path.join(test_dir, img_path + '.jpg')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    print("Dataset preparation complete.")
    print(f"Total classes: {len(classes)}")
    return output_dir


if __name__ == "__main__":
    # Download and extract Food-101
    food101_dir = download_food101("datasets")

    # Create train/val/test split
    output_dir = create_train_val_test_split(food101_dir, "datasets/food101_split")

    print(f"Food-101 dataset prepared at {output_dir}")
