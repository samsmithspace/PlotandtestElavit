import torch
import math


class QKAttentionHook:
    """
    Hook for extracting attention patterns by capturing query (Q) and key (K)
    matrices from Transformer models and computing attention weights manually.
    """

    def __init__(self, model):
        """
        Initialize QK-based attention hook for Transformer model.

        Args:
            model: Transformer model (HuggingFace)
        """
        self.model = model
        self.attention_patterns = {}
        self.hooks = []
        self.q_values = {}
        self.k_values = {}

        # Register hooks to extract Q and K
        self._register_hooks()

    def _hook_query(self, layer_idx):
        """
        Define hook function for capturing query matrices.

        Args:
            layer_idx: Index of the layer being hooked

        Returns:
            Hook function
        """

        def hook(module, input, output):
            # Store the query matrix output
            self.q_values[layer_idx] = output.detach()

        return hook

    def _hook_key(self, layer_idx):
        """
        Define hook function for capturing key matrices.

        Args:
            layer_idx: Index of the layer being hooked

        Returns:
            Hook function
        """

        def hook(module, input, output):
            # Store the key matrix output
            self.k_values[layer_idx] = output.detach()

        return hook

    def _register_hooks(self):
        """Register hooks on query and key projections in all transformer layers."""
        try:
            print("Registering QK hooks on attention layers...")

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

            # Register hooks for each layer's query and key projections
            for i, layer in enumerate(layers):
                # First, locate the attention module
                attn_module = None
                if hasattr(layer, 'attention'):
                    attn_module = layer.attention
                    print(f"Found attention at layer.attention for layer {i}")
                elif hasattr(layer, 'self_attention'):
                    attn_module = layer.self_attention
                    print(f"Found attention at layer.self_attention for layer {i}")
                elif hasattr(layer, 'self_attn'):
                    attn_module = layer.self_attn
                    print(f"Found attention at layer.self_attn for layer {i}")
                else:
                    print(f"WARNING: No attention module found in layer {i}======")
                    continue

                # Now, find query and key projections
                if hasattr(attn_module, 'attention') and attn_module.attention is not attn_module:
                    # Nested attention module (as in BERT-like models)
                    attn_module = attn_module.attention

                # Try different naming conventions for query/key projections
                query_module = None
                key_module = None

                # Check common naming patterns
                if hasattr(attn_module, 'query'):
                    query_module = attn_module.query
                    print(f"Found query projection at attn_module.query for layer {i}")
                elif hasattr(attn_module, 'q_proj'):
                    query_module = attn_module.q_proj
                    print(f"Found query projection at attn_module.q_proj for layer {i}")
                elif hasattr(attn_module, 'q'):
                    query_module = attn_module.q
                    print(f"Found query projection at attn_module.q for layer {i}")

                if hasattr(attn_module, 'key'):
                    key_module = attn_module.key
                    print(f"Found key projection at attn_module.key for layer {i}")
                elif hasattr(attn_module, 'k_proj'):
                    key_module = attn_module.k_proj
                    print(f"Found key projection at attn_module.k_proj for layer {i}")
                elif hasattr(attn_module, 'k'):
                    key_module = attn_module.k
                    print(f"Found key projection at attn_module.k for layer {i}")

                # If we found both query and key projections, register hooks
                if query_module is not None and key_module is not None:
                    handle_q = query_module.register_forward_hook(self._hook_query(i))
                    handle_k = key_module.register_forward_hook(self._hook_key(i))
                    self.hooks.extend([handle_q, handle_k])
                    print(f"Registered QK hooks for layer {i}")
                else:
                    print(f"WARNING: Could not find query/key projections in layer {i}")
        except Exception as e:
            print(f"Error registering hooks: {e}")

    def compute_attention_weights(self):
        """
        Compute attention weights from stored Q and K matrices.

        This computes the standard attention formula: softmax(QK^T / sqrt(d_k))

        Returns:
            Dictionary of computed attention patterns
        """
        print("Computing attention weights from Q and K matrices...")

        # Clear any previous patterns
        self.attention_patterns = {}

        # For each layer where we have both Q and K
        for layer_idx in self.q_values.keys():
            if layer_idx in self.k_values:
                try:
                    q = self.q_values[layer_idx]
                    k = self.k_values[layer_idx]

                    # Print shapes for debugging
                    print(f"Layer {layer_idx} - Q shape: {q.shape}, K shape: {k.shape}")

                    # Compute attention weights: softmax(QK^T / sqrt(d_k))
                    # Q shape: [batch_size, seq_len, num_heads, head_dim]
                    # K shape: [batch_size, seq_len, num_heads, head_dim]

                    # Handle different model implementations
                    if len(q.shape) == 4:  # [batch, seq_len, num_heads, head_dim]
                        # Compute for each head separately
                        batch_size, seq_len, num_heads, head_dim = q.shape

                        # Reshape for batched matmul
                        q_reshaped = q.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
                        k_reshaped = k.permute(0, 2, 3, 1)  # [batch, num_heads, head_dim, seq_len]

                        # Compute attention scores
                        scores = torch.matmul(q_reshaped, k_reshaped) / math.sqrt(head_dim)
                        attention = torch.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]

                    elif len(q.shape) == 3:  # [batch, seq_len, hidden_dim]
                        # We need to reshape to extract heads
                        batch_size, seq_len, hidden_dim = q.shape

                        # Guess the number of heads (common values: 8, 12, 16)
                        # Try to infer from model config
                        num_heads = 12  # Default fallback for BERT-base, ViT-base
                        if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_attention_heads'):
                            num_heads = self.model.config.num_attention_heads

                        head_dim = hidden_dim // num_heads

                        # Reshape query and key to include head dimension
                        q_reshaped = q.view(batch_size, seq_len, num_heads, head_dim)
                        k_reshaped = k.view(batch_size, seq_len, num_heads, head_dim)

                        # Permute for batched matmul
                        q_reshaped = q_reshaped.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
                        k_reshaped = k_reshaped.permute(0, 2, 3, 1)  # [batch, num_heads, head_dim, seq_len]

                        # Compute attention scores
                        scores = torch.matmul(q_reshaped, k_reshaped) / math.sqrt(head_dim)
                        attention = torch.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]

                    else:
                        print(f"Unexpected Q/K shapes for layer {layer_idx}")
                        continue

                    # Store the computed attention pattern
                    self.attention_patterns[layer_idx] = attention.detach().cpu()
                    print(f"Computed attention weights for layer {layer_idx}, shape: {attention.shape}")

                except Exception as e:
                    print(f"Error computing attention for layer {layer_idx}: {e}")

        return self.attention_patterns

    def clear_cached_values(self):
        """Clear cached Q and K values to free memory."""
        self.q_values = {}
        self.k_values = {}
        self.attention_patterns = {}

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.clear_cached_values()