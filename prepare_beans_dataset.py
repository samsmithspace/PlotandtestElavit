import os
import zipfile
import urllib.request
from tqdm import tqdm
import shutil
import random
import numpy as np
from PIL import Image
import requests
import json


def download_beans_dataset(download_dir):
    """
    Download the Beans dataset from Kaggle.

    Args:
        download_dir: Directory to save the downloaded dataset

    Returns:
        Path to the downloaded dataset
    """
    os.makedirs(download_dir, exist_ok=True)

    # URL for the Beans dataset (direct from GitHub)
    url = "https://github.com/AI-Lab-Makerere/ibean/archive/refs/heads/master.zip"
    filename = os.path.join(download_dir, "beans.zip")

    # Download if the file doesn't exist
    if not os.path.exists(filename):
        print(f"Downloading Beans dataset to {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print(f"Found existing download at {filename}")

    # Extract the zipfile
    extract_dir = os.path.join(download_dir, "beans")
    if not os.path.exists(extract_dir):
        print("Extracting Beans dataset...")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        print("Extraction complete.")
    else:
        print(f"Found existing extraction at {extract_dir}")

    # Find the dataset directory
    dataset_dir = os.path.join(extract_dir, "ibean-master")
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Could not find extracted dataset at {dataset_dir}")

    return dataset_dir


def create_train_val_test_split(beans_dir, output_dir, val_split=0.15, test_split=0.15, seed=42):
    """
    Create train/val/test directories for Beans dataset.

    Args:
        beans_dir: Path to extracted Beans dataset
        output_dir: Directory to create the split datasets
        val_split: Percentage of data to use for validation
        test_split: Percentage of data to use for testing
        seed: Random seed for reproducibility

    Returns:
        Path to the output directory
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Define the class names (folders in the dataset)
    class_names = ['angular_leaf_spot', 'bean_rust', 'healthy']

    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(output_dir, exist_ok=True)

    # Create class directories
    for class_name in class_names:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Create class to label mapping for documentation
    class_to_label = {class_name: i for i, class_name in enumerate(class_names)}
    with open(os.path.join(output_dir, "class_mapping.json"), 'w') as f:
        json.dump(class_to_label, f, indent=2)

    # Process each class
    class_counts = {}

    # Function to process images for a class
    def process_class(class_name):
        # Source directory for this class
        source_dir = os.path.join(beans_dir, class_name)
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory not found: {source_dir}")
            return 0

        # Get all image files
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(image_files)

        # Calculate split sizes
        total = len(image_files)
        val_size = int(total * val_split)
        test_size = int(total * test_split)
        train_size = total - val_size - test_size

        # Split the files
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]

        # Copy files to respective directories
        for files, target_dir in [
            (train_files, os.path.join(train_dir, class_name)),
            (val_files, os.path.join(val_dir, class_name)),
            (test_files, os.path.join(test_dir, class_name))
        ]:
            for file in files:
                src = os.path.join(source_dir, file)
                dst = os.path.join(target_dir, file)
                shutil.copy(src, dst)

        return {
            'total': total,
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }

    # Look for different directory structures (original dataset or training data)
    train_dataset_dir = os.path.join(beans_dir, "train")
    if os.path.exists(train_dataset_dir):
        # The dataset has a train directory which contains class folders
        print("Found train directory in dataset")

        # Process each class in the train directory
        for class_name in class_names:
            print(f"Processing class: {class_name}")
            counts = process_class(os.path.join("train", class_name))
            class_counts[class_name] = counts
    else:
        # Try finding the class directories directly in the dataset root
        for class_name in class_names:
            class_dir = os.path.join(beans_dir, class_name)
            if os.path.exists(class_dir):
                print(f"Processing class: {class_name}")
                counts = process_class(class_name)
                class_counts[class_name] = counts

    # Print statistics
    print("\nDataset Statistics:")
    total_train = 0
    total_val = 0
    total_test = 0

    for class_name, counts in class_counts.items():
        print(f"  {class_name}:")
        print(f"    - Total: {counts['total']}")
        print(f"    - Train: {counts['train']}")
        print(f"    - Val: {counts['val']}")
        print(f"    - Test: {counts['test']}")

        total_train += counts['train']
        total_val += counts['val']
        total_test += counts['test']

    print("\nOverall:")
    print(f"  - Total Train: {total_train}")
    print(f"  - Total Val: {total_val}")
    print(f"  - Total Test: {total_test}")
    print(f"  - Total: {total_train + total_val + total_test}")

    return output_dir


def check_image_integrity(output_dir):
    """
    Check if all images are valid and can be opened.

    Args:
        output_dir: Path to the dataset directory
    """
    print("\nChecking image integrity...")
    corrupted_files = []

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)

        if not os.path.exists(split_dir):
            continue

        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)

            if not os.path.isdir(class_dir):
                continue

            for filename in os.listdir(class_dir):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                file_path = os.path.join(class_dir, filename)

                try:
                    # Try to open the image
                    with Image.open(file_path) as img:
                        img.verify()  # Verify it's a valid image
                except Exception as e:
                    corrupted_files.append((file_path, str(e)))

    if corrupted_files:
        print(f"Found {len(corrupted_files)} corrupted images:")
        for file_path, error in corrupted_files:
            print(f"  - {file_path}: {error}")
    else:
        print("All images are valid.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and prepare the Beans dataset")
    parser.add_argument("--download_dir", type=str, default="datasets",
                        help="Directory to download the dataset")
    parser.add_argument("--output_dir", type=str, default="datasets/beans_split",
                        help="Directory to create the split datasets")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Percentage of data to use for validation")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="Percentage of data to use for testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Download and extract Beans dataset
    beans_dir = download_beans_dataset(args.download_dir)

    # Create train/val/test split
    output_dir = create_train_val_test_split(
        beans_dir,
        args.output_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )

    # Check image integrity
    check_image_integrity(output_dir)

    print(f"\nBeans dataset prepared at {output_dir}")
    print("Directory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── angular_leaf_spot/")
    print(f"  │   ├── bean_rust/")
    print(f"  │   └── healthy/")
    print(f"  ├── val/")
    print(f"  │   ├── angular_leaf_spot/")
    print(f"  │   ├── bean_rust/")
    print(f"  │   └── healthy/")
    print(f"  └── test/")
    print(f"      ├── angular_leaf_spot/")
    print(f"      ├── bean_rust/")
    print(f"      └── healthy/")

    print("\nUsage with HN-Freeze:")
    print(f"python main.py --data_dir {args.output_dir} --dataset beans --model_size base")