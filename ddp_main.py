import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import torch.multiprocessing as mp

from hn_freeze.utils.train_with_hn_freeze_ddp import train_with_hn_freeze_ddp



def setup(rank, world_size):
    """
    Initialize the distributed environment.

    Args:
        rank: Unique identifier for each process
        world_size: Total number of processes
    """
    # Set the environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Set the device for this process
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def train_ddp(rank, world_size, args):
    """
    Training function for distributed data parallel.

    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Command line arguments
    """
    # Initialize distributed environment
    setup(rank, world_size)

    # Create model and move it to the correct device
    from models import HNFreeze
    from configs import BeansConfig  # Or other configurations

    # Choose config based on args
    if args.dataset == 'beans':
        config = BeansConfig()
    # Add other configs as needed...

    # Override config with command line arguments
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.lr is not None:
        config.base_lr = args.lr

    # Create model
    from transformers import ViTForImageClassification
    model = ViTForImageClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_classes
    )

    # Move model to the appropriate device
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank])

    # Create data loaders with DistributedSampler
    from torch.utils.data import DataLoader, DistributedSampler
    if args.dataset == 'beans':
        from data.beans_dataset import BeansDataset

        # Create datasets
        train_dataset = BeansDataset(
            root_dir=args.data_dir,
            split='train',
            transform=get_train_transform(config)
        )

        val_dataset = BeansDataset(
            root_dir=args.data_dir,
            split='val',
            transform=get_val_transform(config)
        )

        # Create samplers for distributed training
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
    # Add other datasets as needed...

    # Create HNFreeze instance
    hn_freeze = HNFreeze(model.module, config, device)  # Note: Use model.module to access the base model

    # Analyze layer stability on the main process only
    if rank == 0:
        print("Analyzing layer stability...")
        layer_mad_values = hn_freeze.analyze_layer_stability(train_loader, num_classes=config.num_classes)

        # Visualize MAD values if needed
        from utils.visualization import plot_mad_values
        plot_mad_values(layer_mad_values, save_path=os.path.join(args.output_dir, 'mad_values_cifar100.png'))

    # Synchronize all processes to ensure layer_mad_values is computed
    dist.barrier()

    # Broadcast layer_mad_values from rank 0 to all other processes
    # (Implementation depends on the structure of layer_mad_values)

    # Freeze stable layers
    frozen_layers = hn_freeze.freeze_stable_layers(percentile=config.freeze_percentile)

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.base_lr,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs
    )

    # Create criterion
    criterion = nn.CrossEntropyLoss()

    # Train the model using a modified version of your existing training function


    results = train_with_hn_freeze_ddp(
        model=model,
        hn_freeze=hn_freeze,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        rank=rank,
        world_size=world_size,
        experiment_dir=args.output_dir if rank == 0 else None
    )

    # Only the main process performs evaluation and visualization
    if rank == 0:
        # Load the best model for evaluation
        best_model_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.module.load_state_dict(checkpoint['model_state_dict'])

            # Create test data loader
            test_dataset = BeansDataset(
                root_dir=args.data_dir,
                split='test',
                transform=get_val_transform(config)
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            # Evaluate on test set
            from utils.metrics import evaluate_model
            test_results = evaluate_model(model.module, test_loader, device, criterion)

            # Visualization and reporting as needed
            from utils.visualization import plot_confusion_matrix, plot_training_curves

            # Plot training curves
            plot_training_curves(
                results['history'],
                save_path=os.path.join(args.output_dir, 'training_curves.png')
            )

            # Plot confusion matrix
            plot_confusion_matrix(
                test_results['predictions'],
                test_results['targets'],
                class_names=test_dataset.classes,
                save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
            )

            # Print final results
            print(f"Test accuracy: {test_results['accuracy']:.2f}%")

    # Clean up the distributed environment
    cleanup()


def get_train_transform(config):
    """Get training transforms"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transform(config):
    """Get validation/test transforms"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def main():
    """Main function to set up distributed training"""
    parser = argparse.ArgumentParser(description='HN-Freeze Distributed Training')

    # Add your existing arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--dataset', type=str, default='generic',
                        choices=['generic', 'food101', 'beans'],
                        help='Dataset type to use specialized configurations')
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'],
                        help='ViT model size')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (override config)')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs (override config)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (override config)')

    # Add distributed training specific arguments
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs per node')
    parser.add_argument('--node_rank', type=int, default=0, help='Ranking within the nodes')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master node address')
    parser.add_argument('--master_port', type=str, default='12355', help='Master node port')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    # Calculate world size and rank
    world_size = args.nodes * args.gpus

    if world_size > 1:
        # Use distributed training
        mp.spawn(
            train_ddp,
            args=(world_size, args),
            nprocs=args.gpus,
            join=True
        )
    else:
        # Fallback to single GPU training
        print("Using single GPU training")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Run your original training code here
        # This would be a simplified version of train_ddp but without the distributed parts

        print("Single GPU training not implemented in this file")
        print("Please use your original training script for single GPU")


if __name__ == "__main__":
    main()