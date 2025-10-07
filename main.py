import os
import argparse
import torch
import numpy as np
import random
from transformers import ViTForImageClassification, ViTConfig
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from configs import BeansConfig
from configs import ViTBaseConfig, ViTLargeConfig, ViTSmallConfig
from configs import (
    DeiTConfig, DeiTFoodConfig, DeiTBeansConfig, DeiTCIFAR100Config
)
from hn_freeze.models.deit_hn_freeze import DeiTHNFreeze
from models.deit_model import create_deit_model

from models import AttentionHook, HNFreeze
from utils.training import train_with_hn_freeze
from utils.metrics import evaluate_model, calculate_parameter_efficiency
from utils.visualization import (
    plot_mad_values, plot_training_curves, plot_confusion_matrix,
    plot_epoch_times, plot_training_metrics_with_time,
    # New imports for visualization data storage
    save_training_history, save_layer_mad_values, save_layer_k_values,
    save_model_comparison_data,
    # Imports for loading visualizations
    plot_mad_values_from_file, plot_training_curves_from_file,
    plot_training_metrics_with_time_from_file, plot_parameter_efficiency_comparison_from_file
)
from configs import Food101Config,CIFAR100Config


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='HN-Freeze: Hopfield Network-Guided Vision Transformer Fine-Tuning')

    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default=None, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--dataset', type=str, default='generic',
                        choices=['generic', 'food101', 'beans', 'cifar100'],
                        help='Dataset type to use specialized configurations')

    # Model arguments
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'],
                        help='ViT model size')
    parser.add_argument('--model_type', type=str, default='vit', choices=['vit', 'deit'],
                        help='Model type (ViT or DeiT)')  # Add this line
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (override config)')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs (override config)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (override config)')

    # HN-Freeze arguments
    parser.add_argument('--freeze_threshold', type=float, default=None,
                        help='MAD threshold for freezing layers (override config)')
    parser.add_argument('--stability_window', type=int, default=None,
                        help='Number of steps to compute MAD (override config)')
    parser.add_argument('--calibrate_threshold', action='store_true',
                        help='Calibrate freezing threshold automatically')

    # Misc arguments
    parser.add_argument('--seed', type=int, default=None, help='Random seed (override config)')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load')

    # Visualization arguments
    parser.add_argument('--visualize_only', type=str, default=None,
                        help='Path to experiment directory to create visualizations only')

    return parser.parse_args()


def visualize_training_results(results, experiment_dir, experiment_name=None):
    """
    Visualize training results including timing information.

    Args:
        results: Dictionary with training results
        experiment_dir: Directory to save visualizations
        experiment_name: Name of the experiment (optional)
    """
    # Create visualization directory if it doesn't exist
    vis_dir = os.path.join(experiment_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Create visualization data directory if it doesn't exist
    vis_data_dir = os.path.join(experiment_dir, 'visualization_data')
    os.makedirs(vis_data_dir, exist_ok=True)

    # Get history from results
    history = results['history']

    # Save training history data for later use
    if experiment_name is None:
        experiment_name = os.path.basename(experiment_dir)

    save_training_history(
        history=history,
        experiment_name=experiment_name,
        directory=vis_data_dir
    )

    # Plot epoch times
    plot_epoch_times(
        history=history,
        save_path=os.path.join(vis_dir, 'epoch_times.png')
    )

    # Plot comprehensive training metrics with timing
    plot_training_metrics_with_time(
        history=history,
        save_path=os.path.join(vis_dir, 'training_metrics_with_time.png')
    )


def create_visualizations_from_saved_data(experiment_dir):
    """
    Create visualizations from saved data files.

    This function can be called separately after training to recreate all plots.

    Args:
        experiment_dir: Path to experiment directory
    """
    # Get experiment name
    experiment_name = os.path.basename(experiment_dir)

    # Define directories
    vis_data_dir = os.path.join(experiment_dir, 'visualization_data')
    vis_dir = os.path.join(experiment_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Load and create MAD values plot
    mad_values_file = os.path.join(vis_data_dir, f"{experiment_name}_layer_mad_values.json")
    if os.path.exists(mad_values_file):
        plot_mad_values_from_file(
            filepath=mad_values_file,
            save_path=os.path.join(vis_dir, 'mad_values_cifar100.png')
        )

    # Load and create training metrics plots
    history_file = os.path.join(vis_data_dir, f"{experiment_name}_training_history.json")
    if os.path.exists(history_file):
        plot_training_curves_from_file(
            filepath=history_file,
            save_path=os.path.join(vis_dir, 'training_curves.png')
        )

        plot_training_metrics_with_time_from_file(
            filepath=history_file,
            save_path=os.path.join(vis_dir, 'training_metrics_with_time.png')
        )

    # Load and create model comparison plot
    model_comparison_file = os.path.join(vis_data_dir, f"{experiment_name}_model_comparison.json")
    if os.path.exists(model_comparison_file):
        plot_parameter_efficiency_comparison_from_file(
            filepath=model_comparison_file,
            save_path=os.path.join(vis_dir, 'parameter_efficiency.png')
        )

    print(f"Visualizations created in {vis_dir}")


def main():
    args = parse_args()

    # Handle visualization-only mode
    if args.visualize_only:
        create_visualizations_from_saved_data(args.visualize_only)
        return

    # Create experiment directory
    experiment_name = f"hnfreeze_{args.model_size}_{args.dataset}_{args.model_type}_{args.freeze_threshold}"
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Create visualization data directory
    vis_data_dir = os.path.join(experiment_dir, 'visualization_data')
    os.makedirs(vis_data_dir, exist_ok=True)
    # Select config based on dataset and model size/type
    if args.model_type == 'deit':
        if args.dataset == 'food101':
            config = DeiTFoodConfig()
        elif args.dataset == 'beans':
            config = DeiTBeansConfig()
        elif args.dataset == 'cifar100':
            config = DeiTCIFAR100Config()
        else:
            config = DeiTConfig()

        # Adjust model name based on size
        if args.model_size == 'small':
            config.model_name = 'facebook/deit-small-patch16-224'
        elif args.model_size == 'base':
            config.model_name = 'facebook/deit-base-patch16-224'
        elif args.model_size == 'large':
            config.model_name = 'facebook/deit-large-patch16-224'
    else:
        # Your existing ViT config selection code

        # Select config based on dataset and model size
        if args.dataset == 'food101':
            config = Food101Config()
            if args.model_size == 'small':
                config.model_name = 'google/vit-small-patch16-224-in21k'
                config.base_lr = 5e-5
            elif args.model_size == 'large':
                config.model_name = 'google/vit-large-patch16-224-in21k'
                config.base_lr = 1e-5
                config.batch_size = 16  # Reduced for larger model
        elif args.dataset == 'beans':  # Add this section for beans
            config = BeansConfig()
            if args.model_size == 'small':
                config.model_name = 'google/vit-small-patch16-224-in21k'
                config.base_lr = 5e-5
            elif args.model_size == 'large':
                config.model_name = 'google/vit-large-patch16-224-in21k'
                config.base_lr = 1e-5
                config.batch_size = 16  # Reduced for larger model
        elif args.dataset == 'cifar100':
            config = CIFAR100Config()
            if args.model_size == 'small':
                config.model_name = 'google/vit-small-patch16-224-in21k'
                config.base_lr = 5e-4
            elif args.model_size == 'large':
                config.model_name = 'google/vit-large-patch16-224-in21k'
                config.base_lr = 1e-4
                config.batch_size = 64  # Reduced for larger model
        else:
            # Default configs for other datasets
            if args.model_size == 'small':
                config = ViTSmallConfig()
            elif args.model_size == 'base':
                config = ViTBaseConfig()
            elif args.model_size == 'large':
                config = ViTLargeConfig()

    # Override config with command line arguments
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.lr is not None:
        config.base_lr = args.lr
    if args.freeze_threshold is not None:
        config.freeze_threshold = args.freeze_threshold
    if args.stability_window is not None:
        config.stability_window = args.stability_window
    if args.seed is not None:
        config.seed = args.seed

    # Set random seed
    set_seed(config.seed)

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    if args.data_dir is None:
        raise ValueError("Data directory must be specified with --data_dir")

    if args.dataset == 'food101':
        from data.datasets import create_food101_dataloaders
        train_loader, val_loader, test_loader = create_food101_dataloaders(args.data_dir, config)
    elif args.dataset == 'beans':  # Add this section for beans
        from data.datasets import create_beans_dataloaders
        train_loader, val_loader, test_loader = create_beans_dataloaders(args.data_dir, config)
    elif args.dataset == 'cifar100':
        from data.datasets import create_dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(args.data_dir, config)
    else:
        from data.datasets import create_dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(args.data_dir, config)

    # Get number of classes from dataset
    num_classes = len(train_loader.dataset.classes)
    print(f"Number of classes: {num_classes}")

    # Create model
    model_name = config.model_name

    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Create model with appropriate number of classes
        if args.model_type == 'deit':
            model = create_deit_model(model_name, num_classes=num_classes)
            hn_freeze = DeiTHNFreeze(model, config, device)
        else:
            print("ohno3")
            model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes)
            hn_freeze = HNFreeze(model, config, device)

        model.load_state_dict(checkpoint['model_state_dict'])

        # Get frozen layers from checkpoint
        frozen_layers = checkpoint.get('frozen_layers', [])

        # Apply frozen layers (no need to change this, both HN-Freeze classes handle it the same way)
        for layer_idx in frozen_layers:
            for param in model.vit.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

        print(f"Loaded model with frozen layers: {frozen_layers}")
    else:
        print(f"Creating model: {model_name}")
        if args.model_type == 'deit':
            print(f"Creating DeiT model: {model_name}")
            model = create_deit_model(model_name, num_classes=num_classes)

            # Create DeiT-specific HN-Freeze instance
            hn_freeze = DeiTHNFreeze(model, config, device)
        else:
            print(f"Creating ViT model: {model_name}")
            model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes)
            print("ohno2")
            # Create standard HN-Freeze instance
            hn_freeze = HNFreeze(model, config, device)

    model.to(device)

    # Create HNFreeze instance
    #hn_freeze = HNFreeze(model, config, device)

    # For evaluation only
    if args.eval_only:
        # Initialize attention hook
        attention_hook = AttentionHook(model)

        # Evaluate model
        results = evaluate_model(model, test_loader, device)
        print(f"Test accuracy: {results['accuracy']:.2f}%")

        # Measure inference time
        standard_time = hn_freeze.measure_inference_time(test_loader, use_skipping=False)
        optimized_time = hn_freeze.measure_inference_time(test_loader, use_skipping=True)

        standard_time_ms = standard_time * 1000
        optimized_time_ms = optimized_time * 1000
        time_improvement = (standard_time - optimized_time) / standard_time * 100

        print(f"Standard inference time: {standard_time_ms:.2f} ms/batch")
        print(f"Optimized inference time: {optimized_time_ms:.2f} ms/batch")
        print(f"Time improvement: {time_improvement:.2f}%")

        # Calculate parameter efficiency
        efficiency = calculate_parameter_efficiency(model)
        print(f"Parameter efficiency: {efficiency['trainable_percentage']:.2f}% trainable")

        # Clean up
        attention_hook.remove_hooks()
        return

    # Initialize criterion, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.base_lr,
        weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Initialize attention hook
    #attention_hook = AttentionHook(model)

    # Analyze layer stability
    print("Analyzing layer stability...")
    layer_mad_values = hn_freeze.analyze_layer_stability(train_loader, num_classes=num_classes)

    # Save layer MAD values for later visualization
    save_layer_mad_values(
        layer_mad_values=layer_mad_values,
        frozen_layers=None,  # We haven't frozen layers yet
        threshold=config.freeze_threshold,
        experiment_name=experiment_name,
        directory=vis_data_dir
    )

    # Visualize MAD values
    plot_mad_values(layer_mad_values, save_path=os.path.join(experiment_dir, 'mad_values_cifar100.png'))

    # Calibrate threshold or use the configured value
    if args.calibrate_threshold:
        print("Calibrating freeze threshold...")
        freeze_threshold = hn_freeze.calibrate_freeze_threshold(val_loader, criterion)
    else:
        freeze_threshold = config.freeze_threshold

    # Freeze stable layers
    frozen_layers = hn_freeze.freeze_stable_layers(percentile=freeze_threshold)

    # Update saved MAD values with frozen layers
    save_layer_mad_values(
        layer_mad_values=layer_mad_values,
        frozen_layers=frozen_layers,
        threshold=config.freeze_threshold,
        experiment_name=experiment_name,
        directory=vis_data_dir
    )

    # Save layer k values if available
    if hasattr(hn_freeze, 'layer_k_values'):
        save_layer_k_values(
            layer_k_values=hn_freeze.layer_k_values,
            frozen_layers=frozen_layers,
            experiment_name=experiment_name,
            directory=vis_data_dir
        )

    # Visualize MAD values after freezing
    plot_mad_values(layer_mad_values, frozen_layers,
                    save_path=os.path.join(experiment_dir, 'mad_values_with_frozen.png'),
                    threshold=freeze_threshold)

    # Update optimizer to exclude frozen parameters
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.base_lr,
        weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Calculate parameter efficiency
    efficiency = calculate_parameter_efficiency(model)
    print(f"Parameter efficiency: {efficiency['trainable_percentage']:.2f}% parameters trainable")

    # Remove hooks before training
    #attention_hook.remove_hooks()

    # Train model
    print(f"Starting training for {config.num_epochs} epochs...")
    results = train_with_hn_freeze(
        model=model,
        hn_freeze=hn_freeze,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        experiment_dir=experiment_dir
    )

    # Visualize training results and save visualization data
    visualize_training_results(results, experiment_dir, experiment_name)

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

    # Plot confusion matrix
    plot_confusion_matrix(
        test_results['predictions'],
        test_results['targets'],
        class_names=test_loader.dataset.classes,
        save_path=os.path.join(experiment_dir, 'confusion_matrix.png')
    )

    # Measure inference time
    print("Measuring inference time...")
    standard_time = hn_freeze.measure_inference_time(test_loader, use_skipping=False)
    optimized_time = hn_freeze.measure_inference_time(test_loader, use_skipping=True)

    # Calculate speedup and convert to milliseconds for display
    standard_time_ms = standard_time * 1000
    optimized_time_ms = optimized_time * 1000
    time_improvement = (standard_time - optimized_time) / standard_time * 100

    print(f"Standard inference time: {standard_time_ms:.2f} ms/batch")
    print(f"Optimized inference time: {optimized_time_ms:.2f} ms/batch")
    print(f"Time improvement: {time_improvement:.2f}%")

    # Save model comparison data for parameter efficiency visualization
    models_data = [
        {
            'name': 'Full Fine-tuning',
            'total_params': efficiency['total_params'],
            'trainable_params': efficiency['total_params'],  # All params are trainable
            'performance': 0.0  # If you have baseline accuracy, put it here
        },
        {
            'name': f'HN-Freeze ({len(frozen_layers)}/{hn_freeze.num_layers} frozen)',
            'total_params': efficiency['total_params'],
            'trainable_params': efficiency['trainable_params'],
            'performance': test_results['accuracy']
        }
    ]

    save_model_comparison_data(
        models_data=models_data,
        experiment_name=experiment_name,
        directory=vis_data_dir
    )

    # Save final results
    final_results = {
        'test_accuracy': test_results['accuracy'],
        'parameter_efficiency': efficiency,
        'inference_time': {
            'standard_ms': standard_time_ms,
            'optimized_ms': optimized_time_ms,
            'improvement_percentage': time_improvement
        },
        'frozen_layers': frozen_layers,
        'threshold': freeze_threshold
    }

    with open(os.path.join(experiment_dir, 'final_results.json'), 'w') as f:
        import json
        json.dump(final_results, f, indent=4)

    print(f"All results saved to {experiment_dir}")


if __name__ == '__main__':
    main()

