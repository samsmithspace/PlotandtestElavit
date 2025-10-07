from .default import DefaultConfig

class ViTBaseConfig(DefaultConfig):
    model_name = 'google/vit-base-patch16-224-in21k'
    base_lr = 1e-5

class ViTLargeConfig(DefaultConfig):
    model_name = 'google/vit-large-patch16-224-in21k'
    base_lr = 5e-6
    batch_size = 16  # Reduced batch size due to memory constraints

class ViTSmallConfig(DefaultConfig):
    model_name = 'google/vit-small-patch16-224-in21k'
    base_lr = 3e-5


class Food101Config(DefaultConfig):
    """Configuration for Food-101 dataset fine-tuning."""
    # Dataset has 101 food categories
    num_classes = 101

    # Adjust batch size based on available GPU memory
    batch_size = 32

    # Learning rate settings
    base_lr = 2e-5

    # Training settings
    num_epochs = 15

    # Food-101 images often have more complex patterns
    # Adjust freezing threshold to be more conservative
    freeze_threshold = 25

    # Augmentation strategy for food images
    use_color_augmentation = True


class BeansConfig:
    """Configuration for Beans dataset."""

    def __init__(self):
        # Dataset parameters
        self.num_classes = 3  # healthy, angular leaf spot, bean rust
        self.image_size = 224
        self.channels = 3

        # Model parameters
        self.model_name = 'google/vit-base-patch16-224-in21k'

        # Training parameters
        self.batch_size = 32
        self.num_epochs = 20
        self.base_lr = 2e-5
        self.weight_decay = 0.01
        self.warmup_steps = 0

        # HN-Freeze parameters
        self.freeze_threshold = 50  # Percentile
        self.attention_threshold = 0.7
        self.stability_window = 50
        self.freeze_percentile = 50

        # Other parameters
        self.seed = 42


class CIFAR100Config:
    """Configuration for CIFAR-100 dataset."""

    def __init__(self):
        # Dataset parameters
        self.num_classes = 100  # CIFAR-100 has 100 classes
        self.img_size = 224  # Resize to ViT input size
        self.channels = 3

        # Model parameters
        self.model_name = 'google/vit-base-patch16-224-in21k'

        # Training parameters
        self.batch_size = 32  # CIFAR images are small, so we can use larger batches
        self.num_epochs = 15
        self.base_lr = 3e-4  # Slightly higher learning rate for CIFAR
        self.weight_decay = 0.01
        self.warmup_steps = 0

        # HN-Freeze parameters
        self.freeze_threshold = 50  # Percentile
        self.attention_threshold = 0.9
        self.stability_window = 50
        self.freeze_percentile = 50

        # DataLoader parameters
        self.num_workers = 4  # Number of workers for data loading

        # Other parameters
        self.seed = 42


# Add this to configs/model_configs.py

class DeiTConfig:
    """Base configuration for DeiT models."""

    def __init__(self):
        # Dataset parameters
        self.num_classes = 1000  # Will be overridden by specific dataset configs
        self.img_size = 224
        self.channels = 3

        # Model parameters
        self.model_name = 'facebook/deit-base-patch16-224'  # DeiT base model
        self.num_workers = 4  # Number of workers for data loading
        # Training parameters
        self.batch_size = 32
        self.num_epochs = 20
        self.base_lr = 5e-5  # Lower learning rate for fine-tuning
        self.weight_decay = 0.05  # DeiT uses higher weight decay
        self.warmup_steps = 0

        # HN-Freeze parameters
        self.freeze_threshold = 50  # Percentile
        self.attention_threshold = 0.9
        self.stability_window = 50
        self.freeze_percentile = 50

        # Other parameters
        self.seed = 42


class DeiTFoodConfig(DeiTConfig):
    """Configuration for DeiT with Food-101 dataset."""

    def __init__(self):
        super().__init__()
        self.num_classes = 101
        self.batch_size = 32
        self.base_lr = 1e-5


class DeiTBeansConfig(DeiTConfig):
    """Configuration for DeiT with Beans dataset."""

    def __init__(self):
        super().__init__()
        self.num_classes = 3
        self.batch_size = 32
        self.base_lr = 2e-5


class DeiTCIFAR100Config(DeiTConfig):
    """Configuration for DeiT with CIFAR-100 dataset."""

    def __init__(self):
        super().__init__()
        self.num_classes = 100
        self.batch_size = 32
        self.base_lr = 2e-4