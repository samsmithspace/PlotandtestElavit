import torch
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(model, data_loader, device, criterion=None):
    """
    Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        criterion: Loss function (optional)

    Returns:
        Dictionary containing accuracy, loss (if criterion provided), and other metrics
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if criterion is not None:
                loss = criterion(outputs.logits, targets)
                running_loss += loss.item()

            _, predicted = outputs.logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Store predictions and targets for detailed metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = 100.0 * correct / total

    # Create result dictionary
    results = {
        'accuracy': accuracy,
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets),
    }

    if criterion is not None:
        results['loss'] = running_loss / len(data_loader)

    return results


def compute_detailed_metrics(predictions, targets, class_names=None):
    """
    Compute detailed classification metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        class_names: List of class names (optional)

    Returns:
        Dictionary with detailed metrics
    """
    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions)

    # Get classification report
    if class_names is not None:
        report = classification_report(targets, predictions,
                                       target_names=class_names,
                                       output_dict=True)
    else:
        report = classification_report(targets, predictions,
                                       output_dict=True)

    return {
        'confusion_matrix': cm,
        'classification_report': report
    }

'''

def measure_inference_time(model, data_loader, device, hn_freeze=None, num_batches=100):
    """
    Measure inference time.

    Args:
        model: Model to evaluate
        data_loader: DataLoader for inputs
        device: Device to run on
        hn_freeze: HNFreeze instance for layer skipping (optional)
        num_batches: Number of batches to measure

    Returns:
        Average inference time per batch in milliseconds
    """
    model.to(device)
    model.eval()

    # Warm-up runs
    with torch.no_grad():
        for inputs, _ in data_loader:
            _ = model(inputs.to(device))
            break

    # Measure inference time
    times = []
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break

            inputs = inputs.to(device)

            # Time the forward pass
            start_time = time.time()

            if hn_freeze is not None:
                _ = hn_freeze.inference_with_layer_skipping(inputs)
            else:
                _ = model(inputs)

            torch.cuda.synchronize(device)  # Ensure GPU computation is complete
            end_time = time.time()

            times.append((end_time - start_time) * 1000)  # Convert to ms

    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)

    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'times_ms': times
    }

'''
def calculate_parameter_efficiency(model):
    """
    Calculate parameter efficiency metrics.

    Args:
        model: ViT model

    Returns:
        Dictionary with parameter counts and efficiency metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    efficiency = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_percentage': 100 * trainable_params / total_params,
        'frozen_percentage': 100 * frozen_params / total_params
    }

    return efficiency
