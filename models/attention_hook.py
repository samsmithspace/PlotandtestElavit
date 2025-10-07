import torch


class AttentionHook:
    """
    Hook for extracting attention patterns from Transformer models.
    """

    def __init__(self, model):
        """
        Initialize attention hook for Transformer model.

        Args:
            model: Transformer model (HuggingFace)
        """
        self.model = model
        self.attention_patterns = {}
        self.hooks = []
        print("ohno1")
        # Try to enable output_attentions in config if it exists
        if hasattr(model, 'config'):
            if hasattr(model.config, 'output_attentions'):
                print("Setting model config to output attentions")
                model.config.output_attentions = True

        self._register_hooks()

    def _hook_fn(self, layer_idx):
        """
        Define hook function for capturing attention weights.

        Args:
            layer_idx: Index of the layer being hooked

        Returns:
            Hook function
        """

        def hook(module, input, output):
            # Debug output
            print(f"Layer {layer_idx} attention module output type: {type(output)}")
            if isinstance(output, tuple):
                print(f"Layer {layer_idx} output tuple length: {len(output)}")
                for i, item in enumerate(output):
                    if isinstance(item, torch.Tensor):
                        print(f"  - Output[{i}] is tensor with shape {item.shape}")
                    else:
                        print(f"  - Output[{i}] is {type(item)}")

            # Special handling for Hugging Face attention modules
            try:
                # We need to identify which element in the tuple contains the attention weights
                # For most HF models, it's typically returned as part of the output tuple

                # For BERT-like models with output_attentions=True
                if isinstance(output, tuple) and len(output) >= 3:
                    # The third element is often the attention weights
                    if isinstance(output[2], torch.Tensor):
                        self.attention_patterns[layer_idx] = output[2].detach().cpu()
                        print(f"✓ Captured attention from output[2] for layer {layer_idx}")
                        return

                # For other models that return a tuple with the last element being attentions
                if isinstance(output, tuple) and len(output) >= 2:
                    # Try the last element
                    if isinstance(output[-1], torch.Tensor):
                        self.attention_patterns[layer_idx] = output[-1].detach().cpu()
                        print(f"✓ Captured attention from output[-1] for layer {layer_idx}")
                        return

                # For models that return a tuple with the first element being attentions
                if isinstance(output, tuple) and len(output) >= 1:
                    if isinstance(output[0], torch.Tensor):
                        # Check if the shape looks like attention weights
                        if len(output[0].shape) >= 3:  # [batch_size, num_heads, seq_len, seq_len] or similar
                            self.attention_patterns[layer_idx] = output[0].detach().cpu()
                            print(f"✓ Captured attention from output[0] for layer {layer_idx}")
                            return

                # For models that return attention weights directly
                if isinstance(output, torch.Tensor):
                    self.attention_patterns[layer_idx] = output.detach().cpu()
                    print(f"✓ Captured attention directly for layer {layer_idx}")
                    return

                # If we get here, we couldn't identify the attention weights
                print(f"! Could not identify attention weights for layer {layer_idx}")

                # As a fallback, use any tensor in the tuple
                if isinstance(output, tuple):
                    for i, item in enumerate(output):
                        if isinstance(item, torch.Tensor):
                            # Use the first tensor we find
                            self.attention_patterns[layer_idx] = item.detach().cpu()
                            print(f"? Using fallback: output[{i}] for layer {layer_idx}")
                            return

            except Exception as e:
                print(f"Error in hook for layer {layer_idx}: {e}")

        return hook

    def _register_hooks_on_attention_output(self, layer_idx, attn_module):
        """
        Register hooks on the output method of an attention module.
        This is needed for some models where the forward hook doesn't capture attention weights.
        """
        # Try to find the method that computes attention
        if hasattr(attn_module, 'forward'):
            # Monkey patch the forward method to capture attention weights
            original_forward = attn_module.forward

            def patched_forward(*args, **kwargs):
                # Check if output_attentions is already in kwargs
                if 'output_attentions' not in kwargs:
                    # Only add it if it's not already there
                    kwargs['output_attentions'] = True

                # Call the original method
                outputs = original_forward(*args, **kwargs)

                # Extract attention weights if they're in the output
                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    if isinstance(outputs[2], torch.Tensor):
                        self.attention_patterns[layer_idx] = outputs[2].detach().cpu()

                return outputs

            # Replace the forward method
            attn_module.forward = patched_forward
            print(f"Patched forward method for layer {layer_idx}")

    def _register_hooks(self):
        """Register hooks on all transformer layers."""
        try:
            print("Registering hooks on attention layers...")

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

            # Force model to output attention weights
            if hasattr(base_model, 'config'):
                if hasattr(base_model.config, 'output_attentions'):
                    base_model.config.output_attentions = True
                    print("Enabled output_attentions in base model config")

            # Register hooks for each layer
            for i, layer in enumerate(layers):
                # Check for different attention module patterns
                if hasattr(layer, 'attention'):
                    attn_module = layer.attention
                    if hasattr(attn_module, 'attention'):
                        # For models with nested attention
                        inner_attn = attn_module.attention
                        handle = inner_attn.register_forward_hook(self._hook_fn(i))
                        self.hooks.append(handle)
                        print(f"Registered hook on layer.attention.attention for layer {i}")
                    else:
                        handle = attn_module.register_forward_hook(self._hook_fn(i))
                        self.hooks.append(handle)
                        print(f"Registered hook on layer.attention for layer {i}")
                elif hasattr(layer, 'self_attention'):
                    attn_module = layer.self_attention
                    handle = attn_module.register_forward_hook(self._hook_fn(i))
                    self.hooks.append(handle)
                    print(f"Registered hook on layer.self_attention for layer {i}")
                elif hasattr(layer, 'self_attn'):
                    attn_module = layer.self_attn
                    handle = attn_module.register_forward_hook(self._hook_fn(i))
                    self.hooks.append(handle)
                    print(f"Registered hook on layer.self_attn for layer {i}")
                else:
                    print(f"WARNING: No attention module found in layer {i}??????????")
        except Exception as e:
            print(f"Error registering hooks: {e}")

    def capture_attention_with_forward(self, input_tensor):
        """
        Alternative method to capture attention by doing a forward pass with output_attentions=True.
        This can be used if hooks are not working properly.

        Args:
            input_tensor: Input tensor for the model

        Returns:
            Dictionary of attention weights
        """
        print("Capturing attention with explicit forward pass...")

        # Save original config value
        output_attentions_orig = None
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'output_attentions'):
            output_attentions_orig = self.model.config.output_attentions
            self.model.config.output_attentions = True

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            try:
                # Try with output_attentions in kwargs
                outputs = self.model(input_tensor, output_attentions=True)
            except TypeError as e:
                print(f"Error with output_attentions in kwargs: {e}")
                # Try without explicit output_attentions (use config value)
                outputs = self.model(input_tensor)

        # Restore original config
        if output_attentions_orig is not None:
            self.model.config.output_attentions = output_attentions_orig

        # Extract attention weights from outputs
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            print("Found attention weights in outputs.attentions")
            for i, attn in enumerate(outputs.attentions):
                self.attention_patterns[i] = attn.cpu()
                print(f"Captured attention for layer {i}, shape: {attn.shape}")
        elif isinstance(outputs, tuple) and len(outputs) > 2:
            # Some models return attentions as the third element
            attentions = outputs[2]
            if isinstance(attentions, tuple):
                print(f"Found attention weights as tuple with {len(attentions)} elements")
                for i, attn in enumerate(attentions):
                    if isinstance(attn, torch.Tensor):
                        self.attention_patterns[i] = attn.cpu()
                        print(f"Captured attention for layer {i}, shape: {attn.shape}")
        else:
            print("Could not find attention weights in model outputs")

        return self.attention_patterns

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []