import torch
import torch.nn as nn
import timm

from config import CFG

class ConvNextFeatureExtractor(nn.Module):
    """
    This class defines a feature extractor based on a ConvNext model.
    It uses a pretrained model from the 'timm' library and removes the
    final classification layer to output feature vectors.
    """
    def __init__(self, model_name: str = CFG.model_name, pretrained: bool = CFG.pretrained):
        """
        Initializes the ConvNext feature extractor.

        Args:
            model_name (str): The name of the ConvNext model to use from timm.
            pretrained (bool): Whether to load pretrained weights.
        """
        super().__init__()
        # Load the specified ConvNext model using the timm library.
        # By setting num_classes=0, timm returns a model that outputs features
        # from the global pooling layer instead of classification logits.
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        self.n_features = self.model.num_features
        print(f"Loaded model: {model_name}")
        print(f"Number of features: {self.n_features}")

    def forward(self, x) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): A batch of input images with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: A batch of feature vectors with shape (batch_size, n_features).
        """
        # The model directly outputs the feature vector
        features = self.model(x)
        return features

# Example of how to create the model
if __name__ == '__main__':
    # Ensure the configuration is loaded
    print("Creating model instance for testing...")

    # Create an instance of the feature extractor
    feature_extractor = ConvNextFeatureExtractor()
    feature_extractor.to(CFG.device)
    feature_extractor.eval()  # Set to evaluation mode

    # Create a dummy input tensor
    dummy_input = torch.randn(CFG.batch_size, 3, CFG.img_size[0], CFG.img_size[1]).to(CFG.device)
    print(f"\nInput shape: {dummy_input.shape}")

    # Perform a forward pass
    with torch.no_grad():
        output_features = feature_extractor(dummy_input)

    print(f"Output features shape: {output_features.shape}")
    assert output_features.shape == (CFG.batch_size, feature_extractor.n_features), "Output shape is incorrect!"
    print("\nModel instantiation and forward pass test successful.")
