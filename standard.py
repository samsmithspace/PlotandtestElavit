import os
import argparse
import torch
import numpy as np
import random
from transformers import ViTForImageClassification
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import time
from datetime import datetime

from configs import Food101Config
from data.datasets import create_food101_dataloaders
from utils.metrics import evaluate_model, calculate_parameter_efficiency
from utils.visualization import plot_training_curves


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_regular(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device,
                  experiment_dir=None):
    """
    Train a ViT model with regular fine-tuning (no HN-Freeze).

    Args:
        model: ViT model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration object
        device: Device to train on
        experiment_dir: Directory to save results (optional)

    Returns:
        Dictionary with training history and results
    """
    if experiment_dir is None:
        experiment_dir = os.path.join('experiments',
                                      f'regular_finetuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(experiment_dir, exist_ok=True)

    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        config_dict = {k: v for k, v in vars(config).items()
                       if not k.startswith('__') and not callable(v)}
        json.dump(config_dict, f, indent=4)

    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'lr': [],
        'epoch_time': []  # Track time per epoch
    }

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(config.num_epochs):
        # Record epoch start time
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}')
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'acc': 100. * correct / total,
                'lr': optimizer.param_groups[0]['lr']
            })

        # Learning rate scheduler step
        if scheduler is not None:
            scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.logits, targets)

                val_loss += loss.item()
                _, predicted = outputs.logits.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100.0 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Record epoch end time and calculate duration
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time'].append(epoch_duration)

        print(f'Epoch {epoch + 1}/{config.num_epochs}: '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'Val Acc: {val_acc:.2f}% | '
              f'Time: {epoch_duration:.2f}s')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, os.path.join(experiment_dir, 'best_model.pth'))

            print(f'Best model saved with accuracy: {best_acc:.2f}%')

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
            }, os.path.join(experiment_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        # Save history
        with open(os.path.join(experiment_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

    end_time = time.time()
    training_time = end_time - start_time
    avg_epoch_time = np.mean(history['epoch_time'])

    # Calculate parameter efficiency (all parameters are trainable)
    efficiency = calculate_parameter_efficiency(model)

    # Final results
    results = {
        'best_accuracy': best_acc,
        'final_accuracy': val_acc,
        'training_time': training_time,
        'avg_epoch_time': avg_epoch_time,
        'trainable_parameters': efficiency['trainable_params'],
        'total_parameters': efficiency['total_params'],
        'parameter_efficiency': efficiency['trainable_percentage'],
        'history': history
    }

    # Save results
    with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f'Training completed in {training_time:.2f} seconds')
    print(f'Average epoch time: {avg_epoch_time:.2f} seconds')
    print(f'Best accuracy: {best_acc:.2f}%')
    print(f'Parameter efficiency: {efficiency["trainable_percentage"]:.2f}% trainable parameters')

    return results


def measure_inference_time(model, dataloader, device, num_batches=100):
    """
    Measure inference time.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run on
        num_batches: Number of batches to measure

    Returns:
        Average inference time per batch in milliseconds
    """
    model.to(device)
    model.eval()

    # Warm-up runs
    for inputs, _ in dataloader:
        with torch.no_grad():
            _ = model(inputs.to(device))
        break

    # Measure inference time
    start_time = time.time()
    batch_count = 0

    with torch.no_grad():
        for inputs, _ in dataloader:
            if batch_count >= num_batches:
                break

            _ = model(inputs.to(device))
            batch_count += 1

    end_time = time.time()
    avg_time = (end_time - start_time) / batch_count

    # Convert to milliseconds
    avg_time_ms = avg_time * 1000

    return avg_time_ms


def main():
    parser = argparse.ArgumentParser(description='Regular Fine-tuning for ViT on Food-101')

    # Dataset arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Directory to save results')

    # Model arguments
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'],
                        help='ViT model size')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (override config)')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs (override config)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (override config)')

    # Misc arguments
    parser.add_argument('--seed', type=int, default=None, help='Random seed (override config)')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load')

    args = parser.parse_args()

    # Create experiment directory
    experiment_name = f"regular_{args.model_size}_food101"
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Select config
    config = Food101Config()
    if args.model_size == 'small':
        config.model_name = 'google/vit-small-patch16-224-in21k'
        config.base_lr = 5e-5
    elif args.model_size == 'large':
        config.model_name = 'google/vit-large-patch16-224-in21k'
        config.base_lr = 1e-5
        config.batch_size = 16  # Reduced for larger model

    # Override config with command line arguments
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.lr is not None:
        config.base_lr = args.lr
    if args.seed is not None:
        config.seed = args.seed

    # Set random seed
    set_seed(config.seed)

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_food101_dataloaders(args.data_dir, config)

    # Get number of classes from dataset
    num_classes = len(train_loader.dataset.classes)
    print(f"Number of classes: {num_classes}")

    # Create model
    model_name = config.model_name

    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Create model with appropriate number of classes
        model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded model from epoch {checkpoint['epoch']} with accuracy: {checkpoint['accuracy']:.2f}%")
    else:
        print(f"Creating model: {model_name}")
        model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes)

    model.to(device)

    # For evaluation only
    if args.eval_only:
        # Evaluate model
        results = evaluate_model(model, test_loader, device)
        print(f"Test accuracy: {results['accuracy']:.2f}%")

        # Measure inference time
        inference_time = measure_inference_time(model, test_loader, device)
        print(f"Average inference time: {inference_time:.2f} ms/batch")

        # Calculate parameter efficiency
        efficiency = calculate_parameter_efficiency(model)
        print(f"Parameter efficiency: {efficiency['trainable_percentage']:.2f}% trainable")

        return

    # Initialize criterion, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.base_lr,
        weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Train model
    print(f"Starting regular fine-tuning for {config.num_epochs} epochs...")
    results = train_regular(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        experiment_dir=experiment_dir
    )

    # Load best model for evaluation
    best_model_path = os.path.join(experiment_dir, 'best_model.pth')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with accuracy: {checkpoint['accuracy']:.2f}%")

    # Plot training curves
    plot_training_curves(results['history'],
                         save_path=os.path.join(experiment_dir, 'training_curves.png'))

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = evaluate_model(model, test_loader, device, criterion)
    print(f"Test accuracy: {test_results['accuracy']:.2f}%")

    # Measure inference time
    print("Measuring inference time...")
    inference_time = measure_inference_time(model, test_loader, device)
    print(f"Average inference time: {inference_time:.2f} ms/batch")

    # Save final results
    final_results = {
        'test_accuracy': test_results['accuracy'],
        'parameter_efficiency': calculate_parameter_efficiency(model),
        'inference_time_ms': inference_time
    }

    with open(os.path.join(experiment_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"All results saved to {experiment_dir}")


if __name__ == '__main__':
    main()