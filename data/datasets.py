import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
def create_dataloaders(data_dir, config):
    """
    Create training, validation, and test data loaders.

    Args:
        data_dir: Path to dataset directory containing train, val, and test folders
        config: Configuration object

    Returns:
        train_loader, val_loader, test_loader
    """
    # Define data transformations
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_eval = transforms.Compose([
        transforms.Resize(int(config.img_size * 1.14)),  # 256 when img_size=224
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_eval)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def create_food101_dataloaders(data_dir, config):
    """
    Create data loaders specifically for Food-101 dataset.

    Args:
        data_dir: Path to prepared Food-101 dataset directory
        config: Configuration object

    Returns:
        train_loader, val_loader, test_loader
    """
    # Food-specific transforms with color augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config.img_size),
        transforms.RandomHorizontalFlip(),
        # Food-specific augmentations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_eval = transforms.Compose([
        transforms.Resize(int(config.img_size * 1.14)),
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_eval)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_eval)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


class BeansDataset(Dataset):
    """Custom dataset for Beans dataset."""

    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with beans data
            split (string): 'train', 'val', or 'test'
            transform (callable, optional): Transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        # Map 'val' to 'validation' for directory structure
        if split == 'val':
            split = 'val'

        # Define the class names
        self.classes = ['angular_leaf_spot', 'bean_rust', 'healthy']

        # Get all image paths and labels
        self.image_paths = []
        self.labels = []

        # Navigate through the directory structure
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, split, class_name)
            if not os.path.exists(class_dir):
                raise ValueError(f"Class directory not found: {class_dir}")

            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


def create_beans_dataloaders(data_dir, config):
    """
    Create DataLoaders for Beans dataset.

    Args:
        data_dir (string): Path to the beans dataset directory
        config: Configuration object

    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = BeansDataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform
    )

    val_dataset = BeansDataset(
        root_dir=data_dir,
        split='val',
        transform=val_transform
    )

    test_dataset = BeansDataset(
        root_dir=data_dir,
        split='test',
        transform=val_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader