# Save this as download_beans_huggingface.py
import os
import shutil
import random
from tqdm import tqdm
from PIL import Image


def download_beans_from_huggingface():
    """
    Download the Beans dataset from Hugging Face and organize it into train/val/test splits.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Hugging Face datasets library not found. Installing...")
        import subprocess
        subprocess.run(['pip', 'install', 'datasets'], check=True)
        from datasets import load_dataset

    # Set up directories
    base_dir = "datasets"
    output_dir = os.path.join(base_dir, "beans_split")

    os.makedirs(base_dir, exist_ok=True)

    # Define class names
    class_names = ['angular_leaf_spot', 'bean_rust', 'healthy']

    # Create directories for each split and class
    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    # Load the beans dataset from Hugging Face
    print("Loading Beans dataset from Hugging Face...")
    try:
        dataset = load_dataset("beans")
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Map the label IDs to class names
    label_to_class = {
        0: 'angular_leaf_spot',
        1: 'bean_rust',
        2: 'healthy'
    }

    # Process each split
    splits = {
        'train': dataset['train'],
        'val': dataset['validation'],
        'test': dataset['test']
    }

    # Process and save images
    for split_name, split_data in splits.items():
        print(f"Processing {split_name} split...")

        for i, example in enumerate(tqdm(split_data)):
            # Get image and label
            image = example['image']
            label = example['labels']

            # Get class name
            class_name = label_to_class.get(label, f"unknown_{label}")

            # Create a unique filename
            filename = f"{class_name}_{i + 1:04d}.jpg"
            output_path = os.path.join(output_dir, split_name, class_name, filename)

            # Save the image
            try:
                image.save(output_path)
            except Exception as e:
                print(f"Error saving image {i} in {split_name}/{class_name}: {e}")

    print(f"\nBeans dataset successfully prepared at {output_dir}")
    print("You can now use it with your HN-Freeze model:")
    print(f"python main.py --data_dir {output_dir} --dataset beans --model_size base")
    return output_dir


if __name__ == "__main__":
    download_beans_from_huggingface()