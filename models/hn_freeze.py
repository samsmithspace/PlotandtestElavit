import torch
import torch.nn as nn
import numpy as np
import copy
from tqdm import tqdm
import time



from utils.hopfield import compute_stability_score

from hn_freeze.qk_attention_hook import QKAttentionHook


class HNFreeze:
    """
    HN-Freeze: Hopfield Network-Guided Transformer Fine-Tuning
    Using Q and K matrices to compute attention patterns
    """

    def __init__(self, model, config, device):
        """
        Initialize HN-Freeze for a Transformer model.

        Args:
            model: Transformer model (HuggingFace)
            config: Configuration object
            device: Device to run model on
        """
        self.model = model
        self.config = config
        self.device = device
        self.frozen_layers = []
        self.layer_mad_values = {}
        self.layer_k_values = {}

        # Determine number of layers based on model structure
        if hasattr(model, 'vit') and hasattr(model.vit, 'encoder') and hasattr(model.vit.encoder, 'layer'):
            self.num_layers = len(model.vit.encoder.layer)
            print(f"Using ViT structure with {self.num_layers} layers")
            self.encoder_layers = model.vit.encoder.layer
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            self.num_layers = len(model.encoder.layer)
            print(f"Using standard encoder structure with {self.num_layers} layers")
            self.encoder_layers = model.encoder.layer
        else:
            raise ValueError("Model structure not recognized. Cannot determine number of layers.")

        # Create QK-based attention hook instead of regular AttentionHook
        self.attention_hook = QKAttentionHook(model)

    def compute_operational_mode(self, attention_weights, threshold=None):
        """
        Compute operational mode (k) for attention weights.

        As described in the HN-Freeze paper, k is an indicator of how many
        tokens are effectively attended to by each token.

        Args:
            attention_weights: Tensor of attention weights from a transformer layer
            threshold: Cumulative threshold for determining operational mode

        Returns:
            k: Operational mode value
        """
        if threshold is None:
            threshold = self.config.attention_threshold

        # Make sure we're working with a tensor
        if not isinstance(attention_weights, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(attention_weights)}")

        # Add safety check for detaching if tensor requires grad
        if attention_weights.requires_grad:
            attention_weights = attention_weights.detach()

        # Move to CPU for processing if needed
        if attention_weights.device.type != 'cpu':
            attention_weights = attention_weights.cpu()

        # Print tensor shape for debugging
        shape = attention_weights.shape
        print(f"Attention weights shape: {shape}")

        # Handle different attention weight shapes
        if len(shape) == 4:  # [batch_size, num_heads, seq_len, seq_len]
            # Average across batch and heads
            avg_attention = attention_weights.mean(dim=(0, 1))
        elif len(shape) == 3:
            # Could be [batch, seq_len, seq_len] or [num_heads, seq_len, seq_len]
            # Average across first dimension
            avg_attention = attention_weights.mean(dim=0)
        else:
            # Assume it's already [seq_len, seq_len]
            avg_attention = attention_weights

        # Sort attention weights in descending order
        sorted_attn, _ = torch.sort(avg_attention, descending=True, dim=-1)

        # Compute cumulative sum
        cumulative_attn = torch.cumsum(sorted_attn, dim=-1)

        # Find the first index where cumulative sum exceeds threshold
        # This will be different for each token, so we average across all tokens
        k_values = torch.sum(cumulative_attn < threshold, dim=-1).float() + 1

        # Return the average k value across all tokens
        return k_values.mean().item()

    def analyze_layer_stability(self, train_loader, num_classes=None, samples_per_class=3):
        """
        Analyze layer stability by computing operational modes and their MAD values.

        Following the paper, we sample multiple instances per class to get a comprehensive
        representation of attention patterns.

        Args:
            train_loader: DataLoader for training data
            num_classes: Number of classes in dataset (if None, will be automatically detected)
            samples_per_class: Number of samples per class to analyze

        Returns:
            Dictionary mapping layer indices to their MAD values
        """
        print("Analyzing layer stability...")

        # Initialize dictionary to store k values for each layer over steps
        self.layer_k_values = {i: [] for i in range(self.num_layers)}

        # Try to automatically detect num_classes if not provided
        if num_classes is not None:
            try:
                num_classes = int(num_classes)
            except (TypeError, ValueError):
                print(f"Warning: num_classes was not an integer: {type(num_classes)}. Using auto-detection.")
                num_classes = None

        # If num_classes is still None, try to detect from the dataset
        if num_classes is None:
            # Try to infer from dataloader's dataset
            if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'classes'):
                num_classes = len(train_loader.dataset.classes)
                print(f"Auto-detected {num_classes} classes from dataset")
            else:
                # Set a reasonable default or use stability_window
                print("Could not detect number of classes, using stability_window for sampling")
                samples_needed = self.config.stability_window
                class_samples = None

        # Now set up our sampling strategy
        if num_classes is not None:
            class_samples = {cls: 0 for cls in range(num_classes)}
            samples_needed = num_classes * samples_per_class
        else:
            # If we still don't have num_classes, use stability_window
            samples_needed = self.config.stability_window
            class_samples = None

        # Go into train mode but don't compute gradients
        self.model.train()
        collected_samples = 0

        # Enable gradient computation to observe dynamic layer behaviour
        # This is important as per the paper - we want to see how the layers behave
        # during actual fine-tuning, not just inference
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            if collected_samples >= samples_needed:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Process only needed samples
            if class_samples is not None:
                # Only select samples from classes we still need
                batch_samples = []
                batch_targets = []

                for i, target in enumerate(targets):
                    cls = target.item()
                    if cls in class_samples and class_samples[cls] < samples_per_class:
                        batch_samples.append(inputs[i:i + 1])
                        batch_targets.append(targets[i:i + 1])
                        class_samples[cls] += 1
                        collected_samples += 1

                if not batch_samples:
                    continue

                # Combine selected samples
                inputs = torch.cat(batch_samples)
                targets = torch.cat(batch_targets)
            else:
                collected_samples += inputs.size(0)

            # Forward pass with gradients to simulate fine-tuning behavior
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
            optimizer.zero_grad()

            # Forward pass to capture Q and K values
            outputs = self.model(inputs)

            # Compute loss using an appropriate criterion
            if hasattr(outputs, 'logits'):
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs.logits, targets)
            else:
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Compute attention weights from Q and K matrices
            self.attention_hook.compute_attention_weights()

            # Analyze attention patterns
            for layer_idx in range(self.num_layers):
                if layer_idx in self.attention_hook.attention_patterns:
                    attn_pattern = self.attention_hook.attention_patterns[layer_idx]

                    # Make sure we have a tensor to work with
                    if isinstance(attn_pattern, torch.Tensor):
                        try:
                            k = self.compute_operational_mode(attn_pattern,
                                                              threshold=self.config.attention_threshold)
                            self.layer_k_values[layer_idx].append(k)
                            print(f"Layer {layer_idx} operational mode k: {k}")
                        except Exception as e:
                            print(f"Error computing operational mode for layer {layer_idx}: {e}")
                    else:
                        print(f"Warning: Attention pattern for layer {layer_idx} is not a tensor: {type(attn_pattern)}")
                else:
                    print(f"Warning: No attention pattern found for layer {layer_idx}")

            # Don't actually update weights
            optimizer.zero_grad()

            # Clear attention patterns for next batch to save memory
            self.attention_hook.clear_cached_values()

        # Compute MAD for each layer
        self.layer_mad_values = {
            layer_idx: compute_stability_score(k_values)
            for layer_idx, k_values in self.layer_k_values.items()
            if k_values  # Only compute if we have values
        }

        # Sort layers by MAD
        sorted_mad = sorted(self.layer_mad_values.items(), key=lambda x: x[1])
        print("Layer MAD values (sorted from most to least stable):")
        for layer_idx, mad in sorted_mad:
            print(
                f"  Layer {layer_idx}: MAD = {mad:.4f}, k values: min={min(self.layer_k_values[layer_idx]):.2f}, max={max(self.layer_k_values[layer_idx]):.2f}, std={np.std(self.layer_k_values[layer_idx]):.2f}")

        return self.layer_mad_values

    def freeze_stable_layers(self, percentile=None):
        """
        Freeze layers with MAD values below the percentile threshold.
        According to the paper, a 50% percentile threshold works well.

        Args:
            percentile: MAD percentile for freezing (default: from config)
        Returns:
            List of frozen layer indices
        """
        if percentile is None:
            percentile = self.config.freeze_percentile

        # Get threshold value based on percentile of MAD values
        mad_values = np.array(list(self.layer_mad_values.values()))
        threshold = np.percentile(mad_values, percentile)
        print(f"Freezing layers with MAD < {threshold:.4f} (percentile: {percentile}%)")

        self.frozen_layers = []
        for layer_idx, mad in self.layer_mad_values.items():
            if mad < threshold:
                # Freeze the layer
                for param in self.encoder_layers[layer_idx].parameters():
                    param.requires_grad = False
                self.frozen_layers.append(layer_idx)

        # Sort frozen layers for readability
        self.frozen_layers.sort()

        # Calculate parameter efficiency
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        param_efficiency = trainable_params / total_params * 100

        print(f"Frozen layers: {self.frozen_layers}")
        print(f"Parameter efficiency: {param_efficiency:.2f}% parameters trainable")

        return self.frozen_layers

    def calibrate_freeze_threshold(self, val_loader, criterion, val_steps=50):
        """
        Calibrate the freezing threshold using validation performance.

        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function
            val_steps: Number of steps for quick fine-tuning

        Returns:
            Optimal threshold value
        """
        print("Calibrating freeze threshold...")

        # Sort layers by MAD values
        sorted_layers = sorted(self.layer_mad_values.items(), key=lambda x: x[1])

        best_acc = 0
        best_threshold = 0
        best_percentile = 0
        initial_acc = self._evaluate_model(val_loader)
        print(f"Initial validation accuracy: {initial_acc:.2f}%")

        # Try different percentiles
        percentiles = [25, 50, 75, 90]

        for percentile in percentiles:
            # Calculate threshold for this percentile
            mad_values = np.array(list(self.layer_mad_values.values()))
            threshold = np.percentile(mad_values, percentile)

            # Determine which layers to freeze
            layers_to_freeze = [layer_idx for layer_idx, mad in self.layer_mad_values.items()
                                if mad < threshold]

            # Make a copy of the model for this threshold
            model_copy = copy.deepcopy(self.model)

            # Freeze selected layers
            for l in layers_to_freeze:
                # Access layers through encoder_layers
                for param in self.encoder_layers[l].parameters():
                    param.requires_grad = False

            # Optimizer for unfrozen parameters
            optimizer = getattr(torch.optim, self.config.optimizer)(
                [p for p in model_copy.parameters() if p.requires_grad],
                lr=self.config.base_lr
            )

            # Quick fine-tuning
            model_copy.train()
            for _ in range(val_steps):
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = model_copy(inputs)

                    # Handle different output formats
                    if hasattr(outputs, 'logits'):
                        loss = criterion(outputs.logits, targets)
                    else:
                        loss = criterion(outputs, targets)

                    loss.backward()
                    optimizer.step()
                    break  # Only use one batch per step

            # Evaluate
            model_copy.to(self.device)
            acc = self._evaluate_model(val_loader, model=model_copy)
            print(
                f"Percentile {percentile}% (threshold {threshold:.4f}, freezing {len(layers_to_freeze)} layers): {acc:.2f}%")

            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
                best_percentile = percentile

            # Stop if accuracy drops significantly
            if acc < 0.90 * initial_acc:
                print(f"Accuracy dropped significantly, stopping calibration")
                break

        print(f"Optimal percentile: {best_percentile}% (threshold {best_threshold:.4f}) with accuracy {best_acc:.2f}%")
        return best_percentile

    def _evaluate_model(self, val_loader, model=None):
        """
        Evaluate model accuracy on validation set.

        Args:
            val_loader: DataLoader for validation data
            model: Model to evaluate (default: self.model)

        Returns:
            Accuracy in percentage
        """
        if model is None:
            model = self.model

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)

                # Handle different output formats
                if hasattr(outputs, 'logits'):
                    _, predicted = outputs.logits.max(1)
                else:
                    _, predicted = outputs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total

    def measure_inference_time(self, dataloader, use_skipping=True, num_batches=100):
        """
        Measure inference time with and without layer skipping.

        Args:
            dataloader: DataLoader for evaluation
            use_skipping: Whether to use layer skipping
            num_batches: Number of batches to measure

        Returns:
            Average inference time per batch
        """
        self.model.to(self.device)
        self.model.eval()

        # Warm-up runs
        for inputs, _ in dataloader:
            with torch.no_grad():
                _ = self.model(inputs.to(self.device))
            break

        # Measure inference time
        start_time = time.time()
        batch_count = 0

        with torch.no_grad():
            for inputs, _ in dataloader:
                if batch_count >= num_batches:
                    break

                if use_skipping and self.frozen_layers:
                    # In a real implementation, you would optimize frozen layer computation
                    # Here we just use the normal forward pass to measure theoretical time
                    _ = self.model(inputs.to(self.device))
                else:
                    _ = self.model(inputs.to(self.device))
                batch_count += 1

        end_time = time.time()
        avg_time = (end_time - start_time) / batch_count

        return avg_time