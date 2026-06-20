"""CNN feature extractor for visual similarity search."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights


class EmbeddingModel(nn.Module):
    def __init__(self, backbone: str = "resnet50", normalize: bool = True) -> None:
        """Create a pretrained backbone that returns a compact embedding vector."""
        super().__init__()
        self.normalize = normalize
        self.backbone_name = backbone.lower()

        if self.backbone_name == "resnet50":
            base = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
            self.embedding_dim = 2048
        elif self.backbone_name == "efficientnet_b0":
            base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.feature_extractor = base.features
            self.pool = base.avgpool
            self.embedding_dim = 1280
        else:
            raise ValueError(
                "Unsupported backbone: {0}. Use 'resnet50' or 'efficientnet_b0'.".format(backbone)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized embeddings with shape (batch, embedding_dim)."""
        if self.backbone_name == "resnet50":
            features = self.feature_extractor(x)
            embeddings = features.flatten(1)
        elif self.backbone_name == "efficientnet_b0":
            features = self.feature_extractor(x)
            embeddings = self.pool(features).flatten(1)
        else:
            raise RuntimeError("EmbeddingModel is configured with an unsupported backbone.")

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


def load_model(backbone: str = "resnet50", device: Optional[str] = None) -> EmbeddingModel:
    """Load the model once, move it to the target device, and set eval mode."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EmbeddingModel(backbone=backbone).to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model
