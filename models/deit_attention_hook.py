import torch


class DeiTAttentionHook:
    """
    Hook for extracting attention patterns from DeiT models.
    DeiT has a different structure than ViT, so we need a specialized hook.
    """

    def __init__(self, model):
        """
        Initialize attention hook for DeiT model.

        Args:
            model: DeiT model (wrapped with DeiTModelWrapper)
        """
        self.model = model
        self.attention_patterns = {}
        self.hooks = []
        self.q_values = {}
        self.k_values = {}

        # Register hooks
        self._register_hooks()

    def _hook_attention(self, layer_idx):
        """
        Define hook function for capturing attention scores directly.
        For DeiT, we need to capture the attention scores from the attention block.

        Args:
            layer_idx: Index of the layer being hooked

        Returns:
            Hook function
        """

        def hook(module, inputs, outputs):
            # DeiT attention blocks output attention scores as part of outputs
            # This format may vary based on the specific DeiT implementation
            try:
                # Try to access attention weights directly if available
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    self.attention_patterns[layer_idx] = outputs.attentions.detach()
                    print(f"✓ Captured attention directly for layer {layer_idx}")
                elif isinstance(outputs, tuple) and len(outputs) > 1:
                    # Some implementations may return attention as second element in tuple
                    if isinstance(outputs[1], torch.Tensor):
                        self.attention_patterns[layer_idx] = outputs[1].detach()
                        print(f"✓ Captured attention from tuple for layer {layer_idx}")
                else:
                    print(f"! Could not identify attention weights for layer {layer_idx}")
            except Exception as e:
                print(f"Error in attention hook for layer {layer_idx}: {e}")

        return hook

    def _hook_query(self, layer_idx):
        """
        Hook function to capture query matrices for calculating attention.

        Args:
            layer_idx: Index of the layer being hooked

        Returns:
            Hook function
        """

        def hook(module, inputs, outputs):
            self.q_values[layer_idx] = outputs.detach()

        return hook

    def _hook_key(self, layer_idx):
        """
        Hook function to capture key matrices for calculating attention.

        Args:
            layer_idx: Index of the layer being hooked

        Returns:
            Hook function
        """

        def hook(module, inputs, outputs):
            self.k_values[layer_idx] = outputs.detach()

        return hook

    def _register_hooks(self):
        """Register hooks on DeiT layers to capture attention patterns."""
        try:
            # For DeiT models, the structure is different
            # We need to identify the correct modules to hook

            # Access the encoder layers through the vit attribute (from wrapper)
            layers = self.model.vit.encoder.layer
            num_layers = len(layers)
            print(f"Found {num_layers} encoder layers in DeiT model")

            # Explore the structure of the first layer to help debug
            first_layer = layers[0]
            print(f"First layer attributes: {dir(first_layer)}")

            # Check if attention is directly available
            if hasattr(first_layer, 'attention'):
                print("Found attention module directly in layers")
                for i, layer in enumerate(layers):
                    hook = layer.attention.register_forward_hook(self._hook_attention(i))
                    self.hooks.append(hook)
                    print(f"Registered attention hook for layer {i}")

            # If self-attention is nested differently in DeiT
            elif hasattr(first_layer, 'self_attention'):
                print("Found self_attention module")
                for i, layer in enumerate(layers):
                    hook = layer.self_attention.register_forward_hook(self._hook_attention(i))
                    self.hooks.append(hook)
                    print(f"Registered self_attention hook for layer {i}")

            # If attention is in a different location, try to find it
            else:
                # Try to look for attention in nested structure
                for i, layer in enumerate(layers):
                    # Check various common patterns for attention modules
                    attention_module = None

                    # Check if it's in a different attribute
                    for attr_name in dir(layer):
                        if 'attrn' in attr_name.lower() and not attr_name.startswith('__'):
                            attention_module = getattr(layer, attr_name)
                            print(f"Found attention at layer.{attr_name} for layer {i}")
                            hook = attention_module.register_forward_hook(self._hook_attention(i))
                            self.hooks.append(hook)
                            break

                    # If still not found, fall back to capturing Q and K matrices
                    # This is a more reliable approach if we can identify Q and K projections
                    if attention_module is None:
                        # Search for potential attention components
                        found_q_k = False
                        # Check if we have a layernorm before attention
                        if hasattr(layer, 'layernorm_before') and hasattr(layer, 'attention'):
                            print(f"Found layernorm_before and attention in layer {i}")
                            # Check if attention has query and key projections
                            attn = layer.attention
                            print("good")
                            if hasattr(attn, 'query') and hasattr(attn, 'key'):
                                print(f"Found query and key in attention for layer {i}")
                                q_hook = attn.query.register_forward_hook(self._hook_query(i))
                                k_hook = attn.key.register_forward_hook(self._hook_key(i))
                                self.hooks.extend([q_hook, k_hook])
                                found_q_k = True

                        # If structure is different, try to find intermediate
                        if not found_q_k and hasattr(layer, 'intermediate'):
                            print(f"Found intermediate in layer {i}")
                            # There might be a self-attention before intermediate
                            for potential_attn_name in ['attn', 'attention', 'self_attention', 'self_attn']:
                                if hasattr(layer, potential_attn_name):
                                    attn = getattr(layer, potential_attn_name)
                                    print(f"Found {potential_attn_name} in layer {i}")
                                    # Check for query and key projections
                                    for q_name in ['query', 'q_proj', 'q']:
                                        for k_name in ['key', 'k_proj', 'k']:
                                            if hasattr(attn, q_name) and hasattr(attn, k_name):
                                                print(
                                                    f"Found {q_name} and {k_name} in {potential_attn_name} for layer {i}")
                                                q_hook = getattr(attn, q_name).register_forward_hook(
                                                    self._hook_query(i))
                                                k_hook = getattr(attn, k_name).register_forward_hook(self._hook_key(i))
                                                self.hooks.extend([q_hook, k_hook])
                                                found_q_k = True
                                                break
                                        if found_q_k:
                                            break

                        if not found_q_k:
                            # Last resort: look for attention components anywhere in the layer
                            for attr_name in dir(layer):
                                if not attr_name.startswith('__') and not callable(getattr(layer, attr_name)):
                                    obj = getattr(layer, attr_name)
                                    if isinstance(obj, torch.nn.Module):
                                        # Check various names for query and key projections
                                        for q_name in ['query', 'q_proj', 'q']:
                                            if hasattr(obj, q_name):
                                                q_module = getattr(obj, q_name)
                                                print(
                                                    f"Found potential query at layer.{attr_name}.{q_name} for layer {i}")
                                                q_hook = q_module.register_forward_hook(self._hook_query(i))
                                                self.hooks.append(q_hook)

                                        for k_name in ['key', 'k_proj', 'k']:
                                            if hasattr(obj, k_name):
                                                k_module = getattr(obj, k_name)
                                                print(
                                                    f"Found potential key at layer.{attr_name}.{k_name} for layer {i}")
                                                k_hook = k_module.register_forward_hook(self._hook_key(i))
                                                self.hooks.append(k_hook)

                        if not self.hooks:
                            print(f"WARNING: No attention module found in layer {i}-----")

        except Exception as e:
            print(f"Error registering hooks: {e}")

    def compute_attention_weights(self):
        """
        Compute attention weights from stored Q and K matrices.
        This is used when we couldn't capture attention weights directly.

        Returns:
            Dictionary of computed attention patterns
        """
        import math

        print("Computing attention weights from Q and K matrices...")

        # For each layer where we have both Q and K
        for layer_idx in self.q_values.keys():
            if layer_idx in self.k_values:
                try:
                    q = self.q_values[layer_idx]
                    k = self.k_values[layer_idx]

                    # Print shapes for debugging
                    print(f"Layer {layer_idx} - Q shape: {q.shape}, K shape: {k.shape}")

                    # Compute attention weights
                    # Adjust based on the actual shapes of Q and K
                    if len(q.shape) == 3:  # [batch_size, seq_len, hidden_dim]
                        # Reshape assuming multi-head attention
                        batch_size, seq_len, hidden_dim = q.shape

                        # Infer number of heads (common values: 8, 12, 16)
                        # Try to get it from model config
                        num_heads = 12  # Default fallback
                        if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_attention_heads'):
                            num_heads = self.model.config.num_attention_heads

                        head_dim = hidden_dim // num_heads

                        # Reshape to include head dimension
                        q_reshaped = q.view(batch_size, seq_len, num_heads, head_dim)
                        k_reshaped = k.view(batch_size, seq_len, num_heads, head_dim)

                        # Transpose for batched matmul
                        q_reshaped = q_reshaped.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
                        k_reshaped = k_reshaped.permute(0, 2, 3, 1)  # [batch, heads, head_dim, seq_len]

                        # Compute attention scores and apply softmax
                        scores = torch.matmul(q_reshaped, k_reshaped) / math.sqrt(head_dim)
                        attention = torch.softmax(scores, dim=-1)  # [batch, heads, seq_len, seq_len]

                        # Store the computed attention pattern
                        self.attention_patterns[layer_idx] = attention.detach().cpu()
                        print(f"✓ Computed attention weights for layer {layer_idx}")

                except Exception as e:
                    print(f"Error computing attention for layer {layer_idx}: {e}")

        return self.attention_patterns

    def get_attention_patterns(self):
        """
        Get attention patterns, computing them from Q and K if not directly captured.

        Returns:
            Dictionary of attention patterns
        """
        # If we don't have attention patterns but have Q and K values
        if not self.attention_patterns and (self.q_values and self.k_values):
            self.compute_attention_weights()

        return self.attention_patterns

    def remove_hooks(self):
        """Remove all registered hooks and clear cached values."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_patterns = {}
        self.q_values = {}
        self.k_values = {}