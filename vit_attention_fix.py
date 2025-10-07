import torch
from types import MethodType


def patch_vit_model_for_attention(model):
    """
    Patch a ViT model to force it to output attention weights.

    This function modifies the internal implementation of the model
    to ensure attention weights are properly returned.

    Args:
        model: The ViT model to patch

    Returns:
        The patched model
    """
    print("Patching ViT model to output attention weights...")

    # First, ensure the config is set properly
    if hasattr(model, 'config'):
        model.config.output_attentions = True

    # Find the backbone ViT model if this is a model with a head
    vit_model = model
    if hasattr(model, 'vit'):
        vit_model = model.vit
    elif hasattr(model, 'vision_model'):
        vit_model = model.vision_model

    # Patch the encoder's forward method
    if hasattr(vit_model, 'encoder'):
        original_encoder_forward = vit_model.encoder.forward

        def patched_encoder_forward(self, hidden_states, head_mask=None, output_attentions=True,
                                    output_hidden_states=None, return_dict=None):
            # Force output_attentions to True
            result = original_encoder_forward(hidden_states, head_mask, True, output_hidden_states, return_dict)
            return result

        vit_model.encoder.forward = MethodType(patched_encoder_forward, vit_model.encoder)
        print("Patched encoder forward method")

    # Patch each encoder layer
    if hasattr(vit_model, 'encoder') and hasattr(vit_model.encoder, 'layer'):
        for i, layer in enumerate(vit_model.encoder.layer):
            if hasattr(layer, 'attention'):
                # Patch the attention forward method
                if hasattr(layer.attention, 'forward'):
                    original_attention_forward = layer.attention.forward

                    def make_patched_attention_forward(original_forward, layer_idx):
                        def patched_attention_forward(self, hidden_states, head_mask=None, output_attentions=True):
                            # Force output_attentions to True and collect the attention weights
                            result = original_forward(hidden_states, head_mask, True)

                            # Debug the result structure
                            if isinstance(result, tuple):
                                print(f"Layer {layer_idx} attention output is tuple with {len(result)} elements")
                                for j, elem in enumerate(result):
                                    print(f"  Element {j} type: {type(elem)}")
                                    if isinstance(elem, torch.Tensor):
                                        print(f"  Element {j} shape: {elem.shape}")
                            else:
                                print(f"Layer {layer_idx} attention output type: {type(result)}")

                            return result

                        return patched_attention_forward

                    layer.attention.forward = MethodType(make_patched_attention_forward(original_attention_forward, i),
                                                         layer.attention)
                    print(f"Patched layer {i} attention forward method")

    # Patch the self-attention modules if they exist
    if hasattr(vit_model, 'encoder') and hasattr(vit_model.encoder, 'layer'):
        for i, layer in enumerate(vit_model.encoder.layer):
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'attention') and hasattr(
                    layer.attention.attention, 'forward'):
                # This is for models that have nested attention.attention structure
                original_self_attn_forward = layer.attention.attention.forward

                def make_patched_self_attn_forward(original_forward, layer_idx):
                    def patched_self_attn_forward(self, *args, **kwargs):
                        # Make sure output_attentions is True if it's a parameter
                        if 'output_attentions' in kwargs:
                            kwargs['output_attentions'] = True

                        # Call original method
                        result = original_forward(*args, **kwargs)

                        # If the result doesn't include attention weights, try to extract them
                        if isinstance(result, tuple) and len(result) >= 2:
                            # Look for attention weights in the tuple
                            attention_weights = None
                            for item in result:
                                if isinstance(item, torch.Tensor) and len(item.shape) == 4:
                                    # This is likely the attention weights
                                    attention_weights = item
                                    break

                            if attention_weights is not None:
                                # Return a tuple with attention weights explicitly added
                                if len(result) >= 3:
                                    return result
                                else:
                                    # Append attention weights
                                    return result + (attention_weights,)

                        return result

                    return patched_self_attn_forward

                layer.attention.attention.forward = MethodType(
                    make_patched_self_attn_forward(original_self_attn_forward, i),
                    layer.attention.attention
                )
                print(f"Patched layer {i} self-attention forward method")

    return model


def extract_attention_manually(model, inputs):
    """
    Extract attention weights manually by adding hooks to attention modules.

    Args:
        model: The ViT model
        inputs: Input tensor

    Returns:
        List of attention weight tensors
    """
    print("Extracting attention weights manually...")

    attention_weights = []
    hooks = []

    # Find the backbone ViT model if this is a model with a head
    vit_model = model
    if hasattr(model, 'vit'):
        vit_model = model.vit
    elif hasattr(model, 'vision_model'):
        vit_model = model.vision_model

    # Function to capture attention weights
    def capture_attention(module, input, output):
        if isinstance(output, tuple) and len(output) >= 1:
            # Try to find attention weights in the output tuple
            for item in output:
                if isinstance(item, torch.Tensor) and len(item.shape) >= 3:
                    # This is likely attention weights
                    attention_weights.append(item.detach().cpu())
                    return

            # If we didn't find attention weights, use the first tensor
            for item in output:
                if isinstance(item, torch.Tensor):
                    attention_weights.append(item.detach().cpu())
                    return
        elif isinstance(output, torch.Tensor):
            attention_weights.append(output.detach().cpu())

    # Register hooks on attention modules
    if hasattr(vit_model, 'encoder') and hasattr(vit_model.encoder, 'layer'):
        for layer in vit_model.encoder.layer:
            if hasattr(layer, 'attention'):
                if hasattr(layer.attention, 'attention'):
                    # For models with nested attention.attention
                    hook = layer.attention.attention.register_forward_hook(capture_attention)
                else:
                    # For models with direct attention module
                    hook = layer.attention.register_forward_hook(capture_attention)
                hooks.append(hook)

    # Forward pass to trigger hooks
    model.eval()
    with torch.no_grad():
        _ = model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print(f"Extracted {len(attention_weights)} attention weight tensors")
    for i, attn in enumerate(attention_weights):
        print(f"Layer {i} attention shape: {attn.shape}")

    return attention_weights


def direct_attention_extraction(model, inputs):
    """
    Try multiple methods to extract attention weights from a ViT model.

    Args:
        model: The ViT model
        inputs: Input tensor

    Returns:
        List of attention weight tensors
    """
    # First try with output_attentions in config
    if hasattr(model, 'config'):
        model.config.output_attentions = True

    attention_weights = []

    # Method 1: Use the model's standard output
    model.eval()
    with torch.no_grad():
        outputs = model(inputs, output_attentions=True)

    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        if isinstance(outputs.attentions, tuple):
            # Check if any of the attentions are not None
            has_valid_attentions = any(attn is not None for attn in outputs.attentions)
            if has_valid_attentions:
                print("Found attention weights in outputs.attentions")
                attention_weights = [
                    attn for attn in outputs.attentions if attn is not None
                ]
                return attention_weights

    print("Method 1 failed: outputs.attentions contains None values")

    # Method 2: Extract from encoder outputs
    # Find the backbone ViT model
    vit_model = model
    if hasattr(model, 'vit'):
        vit_model = model.vit
    elif hasattr(model, 'vision_model'):
        vit_model = model.vision_model

    # Try to extract attention directly from encoder
    if hasattr(vit_model, 'encoder'):
        with torch.no_grad():
            # Try to call encoder directly if it has a forward method
            if hasattr(vit_model.encoder, 'forward'):
                try:
                    # Prepare inputs for encoder
                    if hasattr(vit_model, 'embeddings'):
                        hidden_states = vit_model.embeddings(inputs)
                    else:
                        # If we can't get embeddings, just use the original forward pass
                        outputs = model(inputs, output_hidden_states=True)
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                            hidden_states = outputs.hidden_states[0]  # First hidden state
                        else:
                            print("Method 2 failed: Cannot extract hidden states for encoder input")
                            return extract_attention_manually(model, inputs)

                    # Call encoder with output_attentions=True
                    encoder_outputs = vit_model.encoder(
                        hidden_states,
                        output_attentions=True,
                        return_dict=True
                    )

                    if hasattr(encoder_outputs, 'attentions') and encoder_outputs.attentions is not None:
                        print("Found attention weights in encoder outputs")
                        attention_weights = [
                            attn for attn in encoder_outputs.attentions if attn is not None
                        ]
                        if attention_weights:
                            return attention_weights
                except Exception as e:
                    print(f"Error calling encoder directly: {e}")

    print("Method 2 failed: Could not get attention from encoder")

    # Method 3: Use manual extraction with hooks
    return extract_attention_manually(model, inputs)