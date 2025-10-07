import torch
import math


class QKVAttentionHook:
    """
    Hook for extracting attention patterns by intercepting QKV projections
    and computing attention weights manually.
    """

    def __init__(self, model):
        """
        Initialize QKV-based attention hook for Transformer model.

        Args:
            model: Transformer model (ViT or DeiT)
        """
        self.model = model
        self.attention_patterns = {}
        self.hooks = []
        self.qkv_outputs = {}

        # Register hooks to extract QKV
        self._register_hooks()

    def _hook_qkv(self, layer_idx):
        """
        Define hook function for capturing QKV matrices.

        Args:
            layer_idx: Index of the layer being hooked

        Returns:
            Hook function
        """

        def hook(module, input, output):
            # Store the QKV output
            self.qkv_outputs[layer_idx] = output

        return hook

    def _register_hooks(self):
        """Register hooks on QKV projections in all transformer layers."""
        try:
            print("Registering QKV hooks on attention layers...")

            # Determine model structure
            if hasattr(self.model, 'vit') and hasattr(self.model.vit, 'encoder') and hasattr(self.model.vit.encoder,
                                                                                             'layer'):
                base_model = self.model.vit
                layers = base_model.encoder.layer
                print(f"Using ViT structure with {len(layers)} layers")
            elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                base_model = self.model
                layers = base_model.encoder.layer
                print(f"Using standard encoder structure with {len(layers)} layers")
            else:
                print("Could not find standard layer structure")
                return

            # Register hooks for each layer's QKV projection
            for i, layer in enumerate(layers):
                print(layer)
                # First, locate the attention module
                if hasattr(layer, 'attention'):
                    attn_module = layer.attention
                    print(f"Found attention at layer.attention for layer {i}")
                elif hasattr(layer, 'attn'):
                    attn_module = layer.attn
                    print(f"Found attention at layer.attn for layer {i}")
                else:
                    print(f"WARNING: No attention module found in layer {i}")
                    continue

                # Find QKV projection
                if hasattr(attn_module, 'qkv'):
                    qkv_module = attn_module.qkv
                    handle = qkv_module.register_forward_hook(self._hook_qkv(i))
                    self.hooks.append(handle)
                    print(f"Registered QKV hook for layer {i}")
                else:
                    print(f"WARNING: QKV projection not found in layer {i}")
        except Exception as e:
            print(f"Error registering hooks: {e}")

    def compute_attention_weights(self, num_heads=12, head_dim=64):
        """
        Compute attention weights from stored QKV matrices.

        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head

        Returns:
            Dictionary of computed attention patterns
        """
        print("Computing attention weights from QKV matrices...")

        # Clear any previous patterns
        self.attention_patterns = {}

        # For each layer where we have QKV outputs
        for layer_idx, qkv_output in self.qkv_outputs.items():
            try:
                # Extract QKV from the output
                batch_size, seq_len, _ = qkv_output.shape

                # Reshape and separate Q, K, V
                # The qkv projection outputs a tensor of shape [batch_size, seq_len, 3 * num_heads * head_dim]
                # We need to reshape it to [batch_size, seq_len, 3, num_heads, head_dim]
                qkv = qkv_output.reshape(batch_size, seq_len, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]

                # Separate Q, K, V
                q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [batch_size, num_heads, seq_len, head_dim]

                # Print shapes for debugging
                print(f"Layer {layer_idx} - QKV shape: {qkv_output.shape}")
                print(f"Layer {layer_idx} - Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

                # Compute attention scores: QK^T / sqrt(head_dim)
                # Q: [batch_size, num_heads, seq_len, head_dim]
                # K: [batch_size, num_heads, seq_len, head_dim]
                # attention_scores: [batch_size, num_heads, seq_len, seq_len]
                k_t = k.transpose(-2, -1)  # [batch_size, num_heads, head_dim, seq_len]
                attention_scores = torch.matmul(q, k_t) / math.sqrt(head_dim)

                # Apply softmax to get attention weights
                attention_weights = torch.softmax(attention_scores, dim=-1)

                # Store the computed attention pattern
                self.attention_patterns[layer_idx] = attention_weights.detach().cpu()
                print(f"Computed attention weights for layer {layer_idx}, shape: {attention_weights.shape}")

            except Exception as e:
                print(f"Error computing attention for layer {layer_idx}: {e}")

        return self.attention_patterns

    def clear_cached_values(self):
        """Clear cached QKV values to free memory."""
        self.qkv_outputs = {}
        self.attention_patterns = {}

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.clear_cached_values()


# Helper function to get model-specific head dimensions
def get_model_head_dimensions(model):
    """
    Get the number of heads and head dimension from the model.

    Args:
        model: ViT or DeiT model

    Returns:
        tuple: (num_heads, head_dim)
    """
    # Try different ways to access model config
    config = None
    if hasattr(model, 'config'):
        config = model.config
    elif hasattr(model, 'vit') and hasattr(model.vit, 'config'):
        config = model.vit.config

    # Get num_heads
    num_heads = 12  # Default for base models
    if config is not None:
        if hasattr(config, 'num_attention_heads'):
            num_heads = config.num_attention_heads
        elif hasattr(config, 'num_heads'):
            num_heads = config.num_heads

    # Determine hidden size
    hidden_size = 768  # Default for base models
    if config is not None:
        if hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size

    # Calculate head dimension
    head_dim = hidden_size // num_heads

    return num_heads, head_dim


# Example usage in your HN-Freeze class:
"""
def analyze_layer_stability(self, train_loader, num_classes=None, samples_per_class=3):
    # Initialize QKVAttentionHook instead of regular AttentionHook
    self.attention_hook = QKVAttentionHook(self.model)

    # ... your existing code ...

    # Forward pass to capture QKV values
    outputs = self.model(inputs)

    # Get head dimensions
    num_heads, head_dim = get_model_head_dimensions(self.model)

    # Compute attention weights from QKV matrices
    self.attention_hook.compute_attention_weights(num_heads=num_heads, head_dim=head_dim)

    # ... rest of your code using self.attention_hook.attention_patterns ...
"""


# Test function to check if QKV extraction works
def test_qkv_attention_hook():
    """Test the QKV attention hook with a ViT model."""
    from transformers import ViTForImageClassification

    # Create a model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Create the hook
    hook = QKVAttentionHook(model)

    # Create a random input
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    x = torch.randn(batch_size, channels, height, width)

    # Forward pass to capture QKV values
    model.eval()
    with torch.no_grad():
        outputs = model(x)

    # Get head dimensions
    num_heads, head_dim = get_model_head_dimensions(model)
    print(f"Model has {num_heads} heads with dimension {head_dim}")

    # Compute attention weights
    attention_patterns = hook.compute_attention_weights(num_heads=num_heads, head_dim=head_dim)

    # Check results
    print(f"Extracted attention patterns for {len(attention_patterns)} layers")
    for layer_idx, pattern in attention_patterns.items():
        print(f"Layer {layer_idx} attention pattern shape: {pattern.shape}")

    # Clean up
    hook.remove_hooks()
    print("QKV attention hook test complete!")


if __name__ == "__main__":
    # Test the QKV attention hook
    test_qkv_attention_hook()