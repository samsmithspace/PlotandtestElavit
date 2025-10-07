import os
import pickle
import urllib.request
from tqdm import tqdm
import shutil
import random
import numpy as np
from PIL import Image
import json
import tarfile


def download_cifar100_dataset(download_dir):
    """
    Download the CIFAR100 dataset.

    Args:
        download_dir: Directory to save the downloaded dataset

    Returns:
        Path to the downloaded dataset
    """
    os.makedirs(download_dir, exist_ok=True)

    # URL for the CIFAR100 dataset
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = os.path.join(download_dir, "cifar-100-python.tar.gz")

    # Download if the file doesn't exist
    if not os.path.exists(filename):
        print(f"Downloading CIFAR100 dataset to {filename}...")
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as t:
            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    t.total = total_size
                t.update(block_size)

            urllib.request.urlretrieve(url, filename, reporthook=reporthook)
        print("Download complete.")
    else:
        print(f"Found existing download at {filename}")

    # Extract the tarfile
    extract_dir = os.path.join(download_dir, "cifar-100-python")
    if not os.path.exists(extract_dir):
        print("Extracting CIFAR100 dataset...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(path=download_dir)
        print("Extraction complete.")
    else:
        print(f"Found existing extraction at {extract_dir}")

    return extract_dir


def unpickle(file):
    """
    Unpickle the CIFAR100 dataset files.

    Args:
        file: Path to the pickled file

    Returns:
        Dictionary with the unpickled data
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def create_directory_structure(cifar_dir, output_dir, val_split=0.15, seed=42):
    """
    Create train/val/test directories for CIFAR100 dataset.

    Args:
        cifar_dir: Path to extracted CIFAR100 dataset
        output_dir: Directory to create the split datasets
        val_split: Percentage of original training data to use for validation
        seed: Random seed for reproducibility

    Returns:
        Path to the output directory
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Load meta file to get class names
    meta = unpickle(os.path.join(cifar_dir, 'meta'))
    fine_labels = [label.decode('utf-8') for label in meta[b'fine_label_names']]

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Create mapping dictionaries for documentation
    class_to_label = {fine_labels[i]: i for i in range(len(fine_labels))}
    with open(os.path.join(output_dir, "class_mapping.json"), 'w') as f:
        json.dump(class_to_label, f, indent=2)

    # Create split directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    # Create class directories
    for class_name in fine_labels:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Load training data
    train_data = unpickle(os.path.join(cifar_dir, 'train'))
    train_images = train_data[b'data']
    train_fine_labels = train_data[b'fine_labels']

    # Reshape images from flat (3072,) arrays to (32, 32, 3) arrays
    train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Group images by class
    class_indices = {}
    for i in range(len(train_fine_labels)):
        label = train_fine_labels[i]
        class_name = fine_labels[label]
        if class_name not in class_indices:
            class_indices[class_name] = []
        class_indices[class_name].append(i)

    # Split training data into train and validation sets
    class_counts = {}
    for class_name, indices in class_indices.items():
        random.shuffle(indices)

        val_size = int(len(indices) * val_split)
        train_size = len(indices) - val_size

        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        # Save train images
        for i, idx in enumerate(train_indices):
            img = Image.fromarray(train_images[idx])
            img.save(os.path.join(train_dir, class_name, f"{class_name}_{i}.png"))

        # Save validation images
        for i, idx in enumerate(val_indices):
            img = Image.fromarray(train_images[idx])
            img.save(os.path.join(val_dir, class_name, f"{class_name}_{i}.png"))

        class_counts[class_name] = {
            'train': len(train_indices),
            'val': len(val_indices)
        }

    # Load test data
    test_data = unpickle(os.path.join(cifar_dir, 'test'))
    test_images = test_data[b'data']
    test_fine_labels = test_data[b'fine_labels']

    # Reshape test images
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Save test images
    for i in range(len(test_fine_labels)):
        label = test_fine_labels[i]
        class_name = fine_labels[label]

        if class_name not in class_counts:
            class_counts[class_name] = {'test': 0}
        elif 'test' not in class_counts[class_name]:
            class_counts[class_name]['test'] = 0

        img = Image.fromarray(test_images[i])
        img.save(os.path.join(test_dir, class_name, f"{class_name}_{class_counts[class_name].get('test', 0)}.png"))

        class_counts[class_name]['test'] = class_counts[class_name].get('test', 0) + 1

    # Print statistics
    print("\nDataset Statistics:")
    total_train = 0
    total_val = 0
    total_test = 0

    for class_name, counts in class_counts.items():
        total_train += counts.get('train', 0)
        total_val += counts.get('val', 0)
        total_test += counts.get('test', 0)

        print(f"  {class_name}:")
        print(f"    - Train: {counts.get('train', 0)}")
        print(f"    - Val: {counts.get('val', 0)}")
        print(f"    - Test: {counts.get('test', 0)}")
        print(f"    - Total: {counts.get('train', 0) + counts.get('val', 0) + counts.get('test', 0)}")

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

    parser = argparse.ArgumentParser(description="Download and prepare the CIFAR100 dataset")
    parser.add_argument("--download_dir", type=str, default="datasets",
                        help="Directory to download the dataset")
    parser.add_argument("--output_dir", type=str, default="datasets/cifar100_split",
                        help="Directory to create the split datasets")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Percentage of training data to use for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Download and extract CIFAR100 dataset
    cifar_dir = download_cifar100_dataset(args.download_dir)

    # Create train/val/test split
    output_dir = create_directory_structure(
        cifar_dir,
        args.output_dir,
        val_split=args.val_split,
        seed=args.seed
    )

    # Check image integrity
    check_image_integrity(output_dir)

    print(f"\nCIFAR100 dataset prepared at {output_dir}")
    print("Directory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── apple/")
    print(f"  │   ├── aquarium_fish/")
    print(f"  │   └── ... (98 more class directories)")
    print(f"  ├── val/")
    print(f"  │   ├── apple/")
    print(f"  │   ├── aquarium_fish/")
    print(f"  │   └── ... (98 more class directories)")
    print(f"  └── test/")
    print(f"      ├── apple/")
    print(f"      ├── aquarium_fish/")
    print(f"      └── ... (98 more class directories)")

    print("\nUsage with models:")
    print(f"python main.py --data_dir {args.output_dir} --dataset cifar100 --model_size base")