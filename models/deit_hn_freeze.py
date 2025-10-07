import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from utils.hopfield import compute_operational_mode, compute_stability_score
from hn_freeze.qkv_attention_hook import QKVAttentionHook, get_model_head_dimensions


class DeiTHNFreeze:
    """
    HN-Freeze: Hopfield Network-Guided Vision Transformer Fine-Tuning
    Modified to use QKV attention hooks
    """

    def __init__(self, model, config, device):
        """
        Initialize HN-Freeze for a Transformer model.

        Args:
            model: Transformer model
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
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            self.num_layers = len(model.encoder.layer)
            print(f"Using standard encoder structure with {self.num_layers} layers")
        else:
            raise ValueError("Model structure not recognized. Cannot determine number of layers.")

        # Create attention hook - using QKVAttentionHook instead of AttentionHook
        self.attention_hook = QKVAttentionHook(model)

        # Get model head dimensions
        self.num_heads, self.head_dim = get_model_head_dimensions(model)
        print(f"Model has {self.num_heads} heads with dimension {self.head_dim}")

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
        print("Analyzing layer stability using QKV attention patterns...")

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

            # Forward pass to get QKV values
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

            # Compute attention patterns from QKV matrices
            self.attention_hook.compute_attention_weights(
                num_heads=self.num_heads,
                head_dim=self.head_dim
            )

            # Analyze attention patterns
            for layer_idx in range(self.num_layers):
                if layer_idx in self.attention_hook.attention_patterns:
                    attn_pattern = self.attention_hook.attention_patterns[layer_idx]

                    # Make sure we have a tensor to work with
                    if isinstance(attn_pattern, torch.Tensor):
                        try:
                            k = compute_operational_mode(attn_pattern,
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

            # Clear cached data to save memory
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

    # The rest of the HNFreeze class methods remain the same
    # freeze_stable_layers, calibrate_freeze_threshold, etc.

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
                # Freeze the layer based on model structure
                layer = None
                if hasattr(self.model, 'vit') and hasattr(self.model.vit, 'encoder'):
                    layer = self.model.vit.encoder.layer[layer_idx]
                elif hasattr(self.model, 'encoder'):
                    layer = self.model.encoder.layer[layer_idx]

                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = False
                    self.frozen_layers.append(layer_idx)
                else:
                    print(f"Warning: Could not find layer {layer_idx} to freeze")

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
        # Implementation remains the same
        pass

    def _evaluate_model(self, val_loader, model=None):
        """
        Evaluate model accuracy on validation set.
        """
        # Implementation remains the same
        pass

    def measure_inference_time(self, dataloader, use_skipping=True, num_batches=100):
        """
        Measure inference time with and without layer skipping.
        """
        # Implementation remains the same
        pass