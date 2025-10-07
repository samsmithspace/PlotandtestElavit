import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, FuncFormatter
# Set academic style for plots
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'figure.dpi': 300
})


def plot_mad_values(layer_mad_values, frozen_layers=None, save_path=None, threshold=0):
    """
    Plot MAD values for each layer with academic styling.

    Args:
        layer_mad_values: Dictionary mapping layer indices to MAD values
        frozen_layers: List of frozen layer indices (optional)
        save_path: Path to save the plot (optional)
        threshold: Threshold value for freezing (optional)
    """
    layers = sorted(layer_mad_values.keys())
    mad_values = [layer_mad_values[l] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create color schemes for frozen and trainable layers
    frozen_color = '#1f77b4'  # Blue
    trainable_color = '#2ca02c'  # Green

    # Create bars with appropriate colors
    colors = []
    for layer in layers:
        if frozen_layers is not None and layer in frozen_layers:
            colors.append(frozen_color)
        else:
            colors.append(trainable_color)

    bars = ax.bar(layers, mad_values, alpha=0.8, color=colors)

    # Add threshold line if provided
    if threshold > 0 or (frozen_layers is not None and len(frozen_layers) > 0):
        ax.axhline(y=threshold, color='#d62728', linestyle='--', alpha=0.7,
                   linewidth=2, label='Freeze threshold')

    # Academic styling
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Median Absolute Deviation (MAD)', fontweight='bold')
    ax.set_title('Layer-wise Stability Analysis', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add y-axis scale on right side
    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks([])

    # Add detailed legend
    from matplotlib.patches import Patch
    if frozen_layers is not None:
        legend_elements = [
            Patch(facecolor=frozen_color, alpha=0.8, label='Frozen Layers'),
            Patch(facecolor=trainable_color, alpha=0.8, label='Trainable Layers')
        ]
        if threshold > 0:
            legend_elements.insert(0, plt.Line2D([0], [0], color='#d62728', linestyle='--',
                                                 linewidth=2, label=f'Freeze Threshold ({threshold:.4f})'))
        ax.legend(handles=legend_elements, loc='best', frameon=True, framealpha=0.9)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, rotation=0)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation curves with academic styling.

    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot training and validation loss
    ax1.plot(epochs, history['train_loss'], 'o-', color='#1f77b4', linewidth=2, label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 's-', color='#d62728', linewidth=2, label='Validation Loss')

    # Add markers at key points (min val loss)
    min_val_idx = np.argmin(history['val_loss'])
    ax1.plot(min_val_idx + 1, history['val_loss'][min_val_idx], 'D', color='black',
             markersize=8, label=f'Min Val Loss: {history["val_loss"][min_val_idx]:.4f}')

    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.legend(frameon=True, framealpha=0.9)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot validation accuracy
    ax2.plot(epochs, history['val_accuracy'], 'o-', color='#2ca02c', linewidth=2, label='Validation Accuracy')

    # Add markers at key points (max accuracy)
    max_acc_idx = np.argmax(history['val_accuracy'])
    ax2.plot(max_acc_idx + 1, history['val_accuracy'][max_acc_idx], 'D', color='black',
             markersize=8, label=f'Max Accuracy: {history["val_accuracy"][max_acc_idx]:.2f}%')

    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('Validation Accuracy', fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(frameon=True, framealpha=0.9)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add a super title
    fig.suptitle('HN-Freeze Training Performance', fontweight='bold', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_confusion_matrix(predictions, targets, class_names=None, save_path=None):
    """
    Plot confusion matrix with academic styling.

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        class_names: List of class names (optional)
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(targets, predictions)

    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a custom colormap that goes from white to blue
    colors = plt.cm.Blues(np.linspace(0.2, 1, 256))
    blue_cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm_norm, interpolation='nearest', cmap=blue_cmap, aspect='equal')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Frequency', fontweight='bold')

    # Set tick labels
    tick_marks = np.arange(len(cm))
    if class_names:
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)

    # Loop over data dimensions and create text annotations
    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, format(cm_norm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")

    # Academic styling
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('Normalized Confusion Matrix', fontweight='bold', fontsize=16)

    # Add grid lines for better separation
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_layer_operational_modes(layer_k_values, frozen_layers=None, save_path=None):
    """
    Plot operational modes (k-values) for each layer with academic styling.

    Args:
        layer_k_values: Dictionary mapping layer indices to lists of k values
        frozen_layers: List of frozen layer indices (optional)
        save_path: Path to save the plot (optional)
    """
    layers = sorted(layer_k_values.keys())
    data = [layer_k_values[l] for l in layers]

    # Compute statistics for each layer
    medians = [np.median(k_vals) for k_vals in data]
    means = [np.mean(k_vals) for k_vals in data]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create color schemes for frozen and trainable layers
    frozen_color = '#1f77b4'  # Blue
    trainable_color = '#2ca02c'  # Green
    colors = [frozen_color if l in (frozen_layers or []) else trainable_color for l in layers]

    # Create boxplots
    box = ax.boxplot(data, patch_artist=True, labels=layers, showmeans=True,
                     meanprops={'marker': 'o', 'markerfacecolor': 'white',
                                'markeredgecolor': 'black', 'markersize': 6})

    # Style the boxplots
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    # Academic styling
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Operational Mode (k)', fontweight='bold')
    ax.set_title('Distribution of Operational Modes Across Layers', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add legend
    if frozen_layers:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=frozen_color, alpha=0.7, label='Frozen Layers'),
            Patch(facecolor=trainable_color, alpha=0.7, label='Trainable Layers'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                       markeredgecolor='black', markersize=6, label='Mean')
        ]
        ax.legend(handles=legend_elements, loc='best', frameon=True, framealpha=0.9)

    # Add a trend line connecting the medians
    ax.plot(range(1, len(layers) + 1), medians, 'k--', alpha=0.7, label='Median Trend')

    # Add annotations for medians and variability
    for i, (med, k_vals) in enumerate(zip(medians, data)):
        var = np.std(k_vals)
        ax.annotate(f'σ={var:.2f}', xy=(i + 1, med), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=8)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def visualize_attention_patterns(attention_weights, image, layer_idx=0, head_idx=None, save_path=None):
    """
    Visualize attention patterns on an input image with academic styling.

    Args:
        attention_weights: Attention weights from a model forward pass [batch, heads, seq_len, seq_len]
        image: Input image tensor [C, H, W]
        layer_idx: Index of the layer to visualize (default=0)
        head_idx: Index of the attention head to visualize (if None, average across heads)
        save_path: Path to save the visualization (optional)
    """
    fig = plt.figure(figsize=(12, 6))

    # Create a 1x3 grid for: original image, attention map, and overlay
    grid = plt.GridSpec(1, 3, wspace=0.3, hspace=0.1)

    # Convert image tensor to numpy for visualization (assuming normalized)
    if isinstance(image, torch.Tensor):
        img = image.cpu().permute(1, 2, 0).numpy()
        # Denormalize if needed
        img = np.clip((img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]), 0, 1)
    else:
        img = image

    # Extract attention weights
    if head_idx is None:
        # Average across all heads
        attn = attention_weights[0].mean(dim=0).cpu().numpy()
    else:
        attn = attention_weights[0, head_idx].cpu().numpy()

    # Get CLS token attention to the image patches (skip the CLS token itself)
    cls_attn = attn[0, 1:]

    # Reshape attention to match image patches (assuming standard ViT patch structure)
    patch_size = int(np.sqrt(len(cls_attn)))
    attention_map = cls_attn.reshape(patch_size, patch_size)

    # Resize attention map to match image dimensions using bicubic interpolation
    from scipy.ndimage import zoom
    zoom_factor = img.shape[0] / attention_map.shape[0]
    attention_map_resized = zoom(attention_map, zoom_factor, order=3)

    # Plot original image
    ax1 = fig.add_subplot(grid[0])
    ax1.imshow(img)
    ax1.set_title('Original Image', fontweight='bold')
    ax1.axis('off')

    # Plot attention heatmap
    ax2 = fig.add_subplot(grid[1])
    im = ax2.imshow(attention_map, cmap='viridis')
    ax2.set_title('CLS Token Attention', fontweight='bold')
    ax2.axis('off')
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontweight='bold')

    # Plot overlay of attention on original image
    ax3 = fig.add_subplot(grid[2])
    ax3.imshow(img)
    attention_overlay = ax3.imshow(attention_map_resized, cmap='plasma', alpha=0.5)
    ax3.set_title('Attention Overlay', fontweight='bold')
    ax3.axis('off')

    # Add super title
    layer_info = f"Layer {layer_idx}"
    if head_idx is not None:
        layer_info += f", Head {head_idx}"
    else:
        layer_info += ", All Heads (Avg)"
    fig.suptitle(f"Attention Visualization - {layer_info}", fontweight='bold', fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_parameter_efficiency_comparison(models_data, save_path=None):
    """
    Create a bar chart comparing parameter efficiency across different models.

    Args:
        models_data: List of dictionaries with keys 'name', 'total_params',
                    'trainable_params', and optionally 'performance'
        save_path: Path to save the plot (optional)
    """
    names = [data['name'] for data in models_data]
    efficiency = [100 * data['trainable_params'] / data['total_params'] for data in models_data]

    # Calculate parameter reduction
    reduction = [100 - eff for eff in efficiency]

    # Create a dataframe for easier plotting
    df = pd.DataFrame({
        'Model': names,
        'Trainable Parameters (%)': efficiency,
        'Frozen Parameters (%)': reduction
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create stacked bar chart
    df.plot(x='Model', y=['Trainable Parameters (%)', 'Frozen Parameters (%)'],
            kind='bar', stacked=True, ax=ax,
            color=['#2ca02c', '#1f77b4'], alpha=0.8)

    # Add performance values if available
    if 'performance' in models_data[0]:
        ax2 = ax.twinx()
        performance = [data['performance'] for data in models_data]
        ax2.plot(df.index, performance, 'D-', color='#d62728', linewidth=2, markersize=8, label='Accuracy')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold', color='#d62728')
        ax2.tick_params(axis='y', colors='#d62728')

    # Academic styling
    ax.set_xlabel('Model Configuration', fontweight='bold')
    ax.set_ylabel('Parameters (%)', fontweight='bold')
    ax.set_title('Parameter Efficiency Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar in ax.patches:
        height = bar.get_height()
        if height > 5:  # Only add labels to bigger segments
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                        ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_epoch_times(history, save_path=None):
    """
    Plot training time per epoch with academic styling.

    Args:
        history: Dictionary containing training history with 'epoch_time' key
        save_path: Path to save the plot (optional)
    """
    # Make sure we have epoch times
    if 'epoch_time' not in history:
        raise ValueError("No epoch time data found in training history")

    epoch_times = history['epoch_time']
    epochs = range(1, len(epoch_times) + 1)

    # Calculate statistics
    total_time = sum(epoch_times)
    avg_time = np.mean(epoch_times)
    min_time = np.min(epoch_times)
    max_time = np.max(epoch_times)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot epoch times
    bars = ax.bar(epochs, epoch_times, alpha=0.7, color='#1f77b4')

    # Add a horizontal line for the average time
    ax.axhline(y=avg_time, color='#d62728', linestyle='--',
               linewidth=2, label=f'Average: {avg_time:.2f}s')

    # Academic styling
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax.set_title('Training Time per Epoch', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add a text box with statistics
    stats_text = (
        f"Total time: {total_time:.2f}s ({total_time / 60:.2f}min)\n"
        f"Average: {avg_time:.2f}s\n"
        f"Min: {min_time:.2f}s (Epoch {np.argmin(epoch_times) + 1})\n"
        f"Max: {max_time:.2f}s (Epoch {np.argmax(epoch_times) + 1})"
    )

    # Add text box at top right
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # Add trend line
    ax.plot(epochs, epoch_times, 'o-', color='#ff7f0e', alpha=0.6, linewidth=1.5, label='Time Trend')

    ax.legend()
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig


def plot_training_metrics_with_time(history, save_path=None):
    """
    Plot comprehensive training metrics including epoch times with academic styling.

    Args:
        history: Dictionary containing training history with 'epoch_time' key
        save_path: Path to save the plot (optional)
    """
    # Make sure we have all required metrics
    required_keys = ['train_loss', 'val_loss', 'val_accuracy', 'epoch_time']
    for key in required_keys:
        if key not in history:
            raise ValueError(f"Missing key in training history: {key}")

    epochs = range(1, len(history['train_loss']) + 1)

    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Training and validation loss
    ax1 = axs[0, 0]
    ax1.plot(epochs, history['train_loss'], 'o-', color='#1f77b4', linewidth=2, label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 's-', color='#d62728', linewidth=2, label='Validation Loss')

    # Add markers at key points
    min_val_idx = np.argmin(history['val_loss'])
    ax1.plot(min_val_idx + 1, history['val_loss'][min_val_idx], 'D', color='black',
             markersize=8, label=f'Min Val Loss: {history["val_loss"][min_val_idx]:.4f}')

    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.legend(frameon=True, framealpha=0.9)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 2. Validation accuracy
    ax2 = axs[0, 1]
    ax2.plot(epochs, history['val_accuracy'], 'o-', color='#2ca02c', linewidth=2, label='Validation Accuracy')

    # Add markers at key points
    max_acc_idx = np.argmax(history['val_accuracy'])
    ax2.plot(max_acc_idx + 1, history['val_accuracy'][max_acc_idx], 'D', color='black',
             markersize=8, label=f'Max Accuracy: {history["val_accuracy"][max_acc_idx]:.2f}%')

    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('Validation Accuracy', fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(frameon=True, framealpha=0.9)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 3. Training time per epoch
    ax3 = axs[1, 0]
    bars = ax3.bar(epochs, history['epoch_time'], alpha=0.7, color='#1f77b4')

    # Add a horizontal line for the average time
    avg_time = np.mean(history['epoch_time'])
    ax3.axhline(y=avg_time, color='#d62728', linestyle='--',
                linewidth=2, label=f'Average: {avg_time:.2f}s')

    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax3.set_title('Training Time per Epoch', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.legend()

    # 4. Cumulative training time
    ax4 = axs[1, 1]
    cumulative_time = np.cumsum(history['epoch_time'])
    ax4.plot(epochs, cumulative_time, 'o-', color='#9467bd', linewidth=2)

    # Format y-axis as hours:minutes:seconds
    def format_time(x, pos):
        hours = int(x // 3600)
        minutes = int((x % 3600) // 60)
        seconds = int(x % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    ax4.yaxis.set_major_formatter(FuncFormatter(format_time))

    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Cumulative Training Time', fontweight='bold')
    ax4.set_title('Cumulative Training Time', fontweight='bold')
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add total time annotation
    total_time = cumulative_time[-1]
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    ax4.annotate(f"Total: {time_str}",
                 xy=(epochs[-1], cumulative_time[-1]),
                 xytext=(-10, -10),
                 textcoords="offset points",
                 ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    # Add super title
    fig.suptitle('HN-Freeze Training Performance and Timing Analysis',
                 fontsize=18, fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig


def plot_training_time_comparison(regular_history, hnfreeze_history, save_path=None):
    """
    Compare training times between regular fine-tuning and HN-Freeze.

    Args:
        regular_history: Dictionary containing regular model training history
        hnfreeze_history: Dictionary containing HN-Freeze model training history
        save_path: Path to save the plot (optional)
    """
    # Make sure we have epoch times for both models
    if 'epoch_time' not in regular_history or 'epoch_time' not in hnfreeze_history:
        raise ValueError("Missing epoch time data in training history")

    # Get epoch times
    regular_times = regular_history['epoch_time']
    hnfreeze_times = hnfreeze_history['epoch_time']

    # Match lengths if different
    min_epochs = min(len(regular_times), len(hnfreeze_times))
    regular_times = regular_times[:min_epochs]
    hnfreeze_times = hnfreeze_times[:min_epochs]
    epochs = range(1, min_epochs + 1)

    # Calculate statistics
    reg_total = sum(regular_times)
    hn_total = sum(hnfreeze_times)

    reg_avg = np.mean(regular_times)
    hn_avg = np.mean(hnfreeze_times)

    # Calculate speedup
    speedup_per_epoch = [r / h for r, h in zip(regular_times, hnfreeze_times)]
    avg_speedup = reg_avg / hn_avg

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Bar chart comparing epoch times
    x = np.arange(len(epochs))
    width = 0.35

    ax1.bar(x - width / 2, regular_times, width, label='Regular Fine-tuning', color='#1f77b4', alpha=0.7)
    ax1.bar(x + width / 2, hnfreeze_times, width, label='HN-Freeze', color='#2ca02c', alpha=0.7)

    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax1.set_title('Training Time per Epoch Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(epochs)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add avg time lines
    ax1.axhline(y=reg_avg, color='#1f77b4', linestyle='--', alpha=0.7,
                linewidth=1.5, label=f'Reg Avg: {reg_avg:.2f}s')
    ax1.axhline(y=hn_avg, color='#2ca02c', linestyle='--', alpha=0.7,
                linewidth=1.5, label=f'HN Avg: {hn_avg:.2f}s')

    # 2. Plot speedup
    ax2.plot(epochs, speedup_per_epoch, 'o-', color='#ff7f0e', linewidth=2)
    ax2.axhline(y=avg_speedup, color='#d62728', linestyle='--',
                linewidth=2, label=f'Avg Speedup: {avg_speedup:.2f}x')

    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Speedup (Regular / HN-Freeze)', fontweight='bold')
    ax2.set_title('HN-Freeze Training Speedup', fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend()

    # Add text summary
    stats_text = (
        f"Regular Total: {reg_total:.2f}s ({reg_total / 60:.2f}min)\n"
        f"HN-Freeze Total: {hn_total:.2f}s ({hn_total / 60:.2f}min)\n"
        f"Total Time Saved: {reg_total - hn_total:.2f}s ({(reg_total - hn_total) / 60:.2f}min)\n"
        f"Overall Speedup: {reg_total / hn_total:.2f}x"
    )

    # Add text box at top right
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    # Add super title
    fig.suptitle('Training Time Comparison: Regular Fine-tuning vs. HN-Freeze',
                 fontsize=16, fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig


def save_visualization_data(data, filename, directory=None):
    """
    Save visualization data to disk for later plotting.

    Args:
        data: Dictionary or object containing data for visualization
        filename: Name of the file to save
        directory: Directory to save in (default: 'visualization_data')

    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    if directory is None:
        directory = 'visualization_data'

    os.makedirs(directory, exist_ok=True)

    # Full path to file
    filepath = os.path.join(directory, filename)

    # Determine file extension and format
    if not any(filepath.endswith(ext) for ext in ['.json', '.pkl', '.npz']):
        # Default to JSON for simple data
        filepath += '.json'

    # Convert torch tensors to numpy arrays
    if isinstance(data, dict):
        data_copy = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data_copy[key] = value.detach().cpu().numpy()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                data_copy[key] = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in value]
            else:
                data_copy[key] = value
        data = data_copy

    # Save based on file extension
    if filepath.endswith('.json'):
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
        except (TypeError, OverflowError):
            print(f"Warning: Could not save as JSON. Falling back to pickle format.")
            filepath = filepath[:-5] + '.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

    elif filepath.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    elif filepath.endswith('.npz'):
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary for .npz format")
        np.savez(filepath, **data)

    print(f"Visualization data saved to {filepath}")
    return filepath


def load_visualization_data(filepath):
    """
    Load visualization data from disk for plotting.

    Args:
        filepath: Path to the saved data file

    Returns:
        Loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)

    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

    elif filepath.endswith('.npz'):
        data = dict(np.load(filepath, allow_pickle=True))

    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    print(f"Visualization data loaded from {filepath}")
    return data


# Helper class for JSON serialization of numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


# Functions to save specific types of visualization data

def save_training_history(history, experiment_name, directory=None):
    """
    Save training history data for later visualization.

    Args:
        history: Dictionary containing training history
        experiment_name: Name of the experiment
        directory: Directory to save in (default: 'visualization_data')

    Returns:
        Path to the saved file
    """
    filename = f"{experiment_name}_training_history.json"
    return save_visualization_data(history, filename, directory)


def save_layer_mad_values(layer_mad_values, frozen_layers, threshold, experiment_name, directory=None):
    """
    Save layer MAD values for later visualization.

    Args:
        layer_mad_values: Dictionary mapping layer indices to MAD values
        frozen_layers: List of frozen layer indices
        threshold: Threshold value used for freezing
        experiment_name: Name of the experiment
        directory: Directory to save in (default: 'visualization_data')

    Returns:
        Path to the saved file
    """
    data = {
        'layer_mad_values': layer_mad_values,
        'frozen_layers': frozen_layers,
        'threshold': threshold
    }

    filename = f"{experiment_name}_layer_mad_values.json"
    return save_visualization_data(data, filename, directory)


def save_layer_k_values(layer_k_values, frozen_layers, experiment_name, directory=None):
    """
    Save layer operational mode (k) values for later visualization.

    Args:
        layer_k_values: Dictionary mapping layer indices to lists of k values
        frozen_layers: List of frozen layer indices
        experiment_name: Name of the experiment
        directory: Directory to save in (default: 'visualization_data')

    Returns:
        Path to the saved file
    """
    data = {
        'layer_k_values': layer_k_values,
        'frozen_layers': frozen_layers
    }

    filename = f"{experiment_name}_layer_k_values.pkl"
    return save_visualization_data(data, filename, directory)


def save_attention_data(attention_weights, image, layer_idx, head_idx, experiment_name, directory=None):
    """
    Save attention visualization data for later plotting.

    Args:
        attention_weights: Attention weights tensor
        image: Input image tensor
        layer_idx: Index of the layer
        head_idx: Index of the attention head (or None for all heads)
        experiment_name: Name of the experiment
        directory: Directory to save in (default: 'visualization_data')

    Returns:
        Path to the saved file
    """
    # Convert torch tensors to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    data = {
        'attention_weights': attention_weights,
        'image': image,
        'layer_idx': layer_idx,
        'head_idx': head_idx
    }

    head_info = f"_head{head_idx}" if head_idx is not None else "_allavg"
    filename = f"{experiment_name}_attention_layer{layer_idx}{head_info}.npz"
    return save_visualization_data(data, filename, directory)


def save_model_comparison_data(models_data, experiment_name, directory=None):
    """
    Save model comparison data for later visualization.

    Args:
        models_data: List of dictionaries with model information
        experiment_name: Name of the experiment
        directory: Directory to save in (default: 'visualization_data')

    Returns:
        Path to the saved file
    """
    filename = f"{experiment_name}_model_comparison.json"
    return save_visualization_data(models_data, filename, directory)


def save_training_time_comparison(regular_history, hnfreeze_history, experiment_name, directory=None):
    """
    Save training time comparison data for later visualization.

    Args:
        regular_history: Dictionary containing regular model training history
        hnfreeze_history: Dictionary containing HN-Freeze model training history
        experiment_name: Name of the experiment
        directory: Directory to save in (default: 'visualization_data')

    Returns:
        Path to the saved file
    """
    data = {
        'regular_history': regular_history,
        'hnfreeze_history': hnfreeze_history
    }

    filename = f"{experiment_name}_training_time_comparison.json"
    return save_visualization_data(data, filename, directory)


# Functions to create visualizations from saved data

def plot_training_curves_from_file(filepath, save_path=None):
    """
    Plot training curves from saved history file.

    Args:
        filepath: Path to the saved history file
        save_path: Path to save the plot (optional)

    Returns:
        Figure object
    """
    history = load_visualization_data(filepath)
    return plot_training_curves(history, save_path)


def plot_mad_values_from_file(filepath, save_path=None):
    """
    Plot MAD values from saved file.

    Args:
        filepath: Path to the saved MAD values file
        save_path: Path to save the plot (optional)

    Returns:
        Figure object
    """
    data = load_visualization_data(filepath)
    return plot_mad_values(
        data['layer_mad_values'],
        data.get('frozen_layers'),
        save_path,
        data.get('threshold', 0)
    )


def plot_layer_operational_modes_from_file(filepath, save_path=None):
    """
    Plot layer operational modes from saved file.

    Args:
        filepath: Path to the saved k values file
        save_path: Path to save the plot (optional)

    Returns:
        Figure object
    """
    data = load_visualization_data(filepath)
    return plot_layer_operational_modes(
        data['layer_k_values'],
        data.get('frozen_layers'),
        save_path
    )


def visualize_attention_patterns_from_file(filepath, save_path=None):
    """
    Visualize attention patterns from saved file.

    Args:
        filepath: Path to the saved attention data file
        save_path: Path to save the plot (optional)

    Returns:
        Figure object
    """
    data = load_visualization_data(filepath)
    return visualize_attention_patterns(
        data['attention_weights'],
        data['image'],
        data.get('layer_idx', 0),
        data.get('head_idx'),
        save_path
    )


def plot_parameter_efficiency_comparison_from_file(filepath, save_path=None):
    """
    Plot parameter efficiency comparison from saved file.

    Args:
        filepath: Path to the saved model comparison file
        save_path: Path to save the plot (optional)

    Returns:
        Figure object
    """
    models_data = load_visualization_data(filepath)
    return plot_parameter_efficiency_comparison(models_data, save_path)


def plot_epoch_times_from_file(filepath, save_path=None):
    """
    Plot epoch times from saved history file.

    Args:
        filepath: Path to the saved history file
        save_path: Path to save the plot (optional)

    Returns:
        Figure object
    """
    history = load_visualization_data(filepath)
    return plot_epoch_times(history, save_path)


def plot_training_metrics_with_time_from_file(filepath, save_path=None):
    """
    Plot training metrics with time from saved history file.

    Args:
        filepath: Path to the saved history file
        save_path: Path to save the plot (optional)

    Returns:
        Figure object
    """
    history = load_visualization_data(filepath)
    return plot_training_metrics_with_time(history, save_path)


def plot_training_time_comparison_from_file(filepath, save_path=None):
    """
    Plot training time comparison from saved file.

    Args:
        filepath: Path to the saved comparison file
        save_path: Path to save the plot (optional)

    Returns:
        Figure object
    """
    data = load_visualization_data(filepath)
    return plot_training_time_comparison(
        data['regular_history'],
        data['hnfreeze_history'],
        save_path
    )