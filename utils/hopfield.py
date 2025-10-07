import torch
import numpy as np


def compute_operational_mode(attention_weights, threshold=0.9):
    """
    Compute the operational mode k̄ for attention patterns.
    k̄ is the minimal number of tokens required to capture 90% of attention mass.

    Args:
        attention_weights: Tensor of shape [batch_size, num_heads, seq_len, seq_len]
                          containing attention weights
        threshold: Threshold for cumulative attention mass (default: 0.9)

    Returns:
        Median k̄ across all heads and tokens
    """
    # Check if input is a tensor and convert to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()

    batch_size, num_heads, seq_len, _ = attention_weights.shape
    k_values = []

    for batch_idx in range(batch_size):
        for head_idx in range(num_heads):
            for token_idx in range(seq_len):
                # Extract attention weights for current token
                attn_weights = attention_weights[batch_idx, head_idx, token_idx]

                # Sort attention weights in descending order
                sorted_weights = np.sort(attn_weights)[::-1]

                # Compute cumulative sum
                cumulative_weights = np.cumsum(sorted_weights)

                # Find minimum k for which sum exceeds threshold
                k = np.argmax(cumulative_weights >= threshold) + 1

                k_values.append(k)
    print(np.median(k_values))
    # Return median k across all tokens and heads
    return np.median(k_values)


def compute_stability_score(k_values):
    """
    Compute the Median Absolute Deviation (MAD) of operational modes k̄.
    MADℓ = median_t(|k̄ℓt - median_t'(k̄ℓt-19:t)|)

    Args:
        k_values: List of k̄ values over training steps

    Returns:
        MAD score for the layer
    """
    if len(k_values) < 20:
        # If we don't have enough samples, use all available ones
        window_size = max(1, len(k_values) // 2)
    else:
        window_size = 20

    # Convert to numpy array if it's not already
    k_array = np.array(k_values)

    # For each point, compute absolute deviation from median of previous window
    deviations = []
    for t in range(window_size, len(k_array)):
        # Get median of previous window
        prev_median = np.median(k_array[t - window_size:t])

        # Compute absolute deviation
        deviation = abs(k_array[t] - prev_median)
        deviations.append(deviation)

    # If we don't have enough data points
    if not deviations:
        # Compute variance of all values instead
        return np.std(k_array) if len(k_array) > 1 else 0

    # Return median of absolute deviations
    return np.median(deviations)


def compute_batch_operational_modes(attention_patterns, threshold=0.9):
    """
    Compute operational modes for a batch of attention patterns.

    Args:
        attention_patterns: Dictionary mapping layer indices to attention weights
        threshold: Threshold for cumulative attention mass (default: 0.9)

    Returns:
        Dictionary mapping layer indices to their k̄ values
    """
    layer_k_values = {}

    for layer_idx, attn_pattern in attention_patterns.items():
        layer_k_values[layer_idx] = compute_operational_mode(
            attn_pattern, threshold=threshold
        )

    return layer_k_values


class AttentionHook:
    """
    Hook to capture attention patterns from ViT model.
    """

    def __init__(self, model):
        self.model = model
        self.attention_patterns = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks for all attention modules"""
        # This implementation will depend on your specific ViT model structure
        # For huggingface transformers:

        for i, layer in enumerate(self.model.vit.encoder.layer):
            # Adjust this based on your model's attention structure
            self.handles.append(
                layer.attention.attention.register_forward_hook(
                    self._create_hook_fn(i)
                )
            )

    def _create_hook_fn(self, layer_idx):
        """Create a hook function for the given layer"""

        def hook_fn(module, input, output):
            # Capture attention weights
            # For huggingface transformers, this might be:
            # attention_probs = output[1]  # Adjust based on your model's output structure

            # Inspect the output to find attention weights
            # This will depend on your model's specific implementation
            # For example, some models return attention weights directly:
            attention_probs = output  # Adjust as needed

            self.attention_patterns[layer_idx] = attention_probs

        return hook_fn

    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []