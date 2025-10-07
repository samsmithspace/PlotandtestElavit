import torch
from transformers import DeiTModel, DeiTForImageClassification
from transformers.models.deit.configuration_deit import DeiTConfig as HFDeiTConfig
import timm
from timm.models.vision_transformer import VisionTransformer


class DeiTModelWrapper:
    """
    Wrapper for DeiT model to make it compatible with HN-Freeze's ViT implementation.
    This version uses timm's implementation as a fallback if HuggingFace's implementation fails.
    """

    def __init__(self, model_name, num_classes, **kwargs):
        """
        Initialize a DeiT model.

        Args:
            model_name: Name of the DeiT model from Hugging Face
            num_classes: Number of output classes
            **kwargs: Additional arguments to pass to the model
        """
        print(f"Initializing DeiT model: {model_name}")

        # Convert HuggingFace model name to timm model name
        if model_name == 'facebook/deit-base-patch16-224':
            timm_model_name = 'deit_base_patch16_224'
        elif model_name == 'facebook/deit-small-patch16-224':
            timm_model_name = 'deit_small_patch16_224'
        elif model_name == 'facebook/deit-tiny-patch16-224':
            timm_model_name = 'deit_tiny_patch16_224'
        elif model_name == 'facebook/deit-large-patch16-224':
            timm_model_name = 'deit_large_patch16_384'
        else:
            timm_model_name = 'deit_base_patch16_224'
            print(f"Warning: Unknown model name {model_name}, defaulting to {timm_model_name}")

        # Try to load from HuggingFace first
        try:
            print("Trying to load DeiT model from HuggingFace...")
            self.model = DeiTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                **kwargs
            )
            self.model_source = "huggingface"
            print("Successfully loaded DeiT model from HuggingFace")

            # Map attributes to match ViT interface
            self.deit = self.model.deit
            self.vit = self.deit

        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            print("Falling back to timm implementation...")

            # Load from timm as a fallback
            self.model = timm.create_model(
                timm_model_name,
                pretrained=True,
                num_classes=num_classes
            )
            self.model_source = "timm"
            print(f"Successfully loaded DeiT model from timm: {timm_model_name}")

            # For timm models, manually create a structure compatible with HN-Freeze
            # This is a bit hacky but will allow HN-Freeze to work with timm models
            if isinstance(self.model, VisionTransformer):
                self.vit = self.model
                self.vit.encoder = type('', (), {})()  # Create a dummy encoder object

                # Move the blocks to match HuggingFace's structure
                self.vit.encoder.layer = self.model.blocks

                # Give it a similar interface to HuggingFace models
                self.deit = self.vit

                print("Successfully mapped timm model structure to be compatible with HN-Freeze")
            else:
                raise ValueError("The timm model is not a VisionTransformer, unsupported model type")

    def __call__(self, x, **kwargs):
        """Forward pass with the same interface as ViT."""
        if self.model_source == "huggingface":
            return self.model(pixel_values=x, **kwargs)
        else:
            # For timm models, just call the model directly
            return type('', (), {'logits': self.model(x)})()

    def forward(self, x, **kwargs):
        """Forward pass with the same interface as ViT."""
        return self.__call__(x, **kwargs)

    def to(self, device):
        """Move model to device."""
        self.model.to(device)
        return self

    def train(self, mode=True):
        """Set model to training mode."""
        self.model.train(mode)
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self

    def state_dict(self):
        """Get model state dict."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """Load model state dict."""
        self.model.load_state_dict(state_dict)

    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()

    @property
    def config(self):
        """Get model config."""
        if hasattr(self.model, 'config'):
            return self.model.config
        else:
            # For timm models, create a simple config object
            return type('', (), {'hidden_size': self.model.embed_dim})()


def create_deit_model(model_name, num_classes, **kwargs):
    """
    Create a DeiT model with HN-Freeze compatibility.

    Args:
        model_name: Name of the DeiT model from Hugging Face
        num_classes: Number of output classes
        **kwargs: Additional arguments to pass to the model

    Returns:
        DeiT model wrapped to be compatible with HN-Freeze
    """
    return DeiTModelWrapper(model_name, num_classes, **kwargs)


# Test function to check if DeiT works with the expected interface
def test_deit_model():
    """Test if DeiT model works with the expected interface."""
    model = create_deit_model('facebook/deit-base-patch16-224', num_classes=10)

    # Create a random input
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    x = torch.randn(batch_size, channels, height, width)

    # Test forward pass
    outputs = model(x)

    # Check outputs
    print("Output type:", type(outputs))
    if hasattr(outputs, 'logits'):
        print("Logits shape:", outputs.logits.shape)

    # Check if the model has the expected structure for HN-Freeze
    print("Has vit attribute:", hasattr(model, 'vit'))
    print("Has vit.encoder.layer:", hasattr(model.vit, 'encoder') and hasattr(model.vit.encoder, 'layer'))
    print("Number of layers:", len(model.vit.encoder.layer))

    # Check parameter access
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)

    print("DeiT model test complete!")


if __name__ == "__main__":
    # Test the DeiT model
    test_deit_model()