"""Image preprocessing utilities for ImageNet-pretrained CNNs."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(image_size: int = 224) -> transforms.Compose:
    """Return the standard ImageNet evaluation transform pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image(path: Union[str, Path]) -> Image.Image:
    """Load an image from disk and force RGB output."""
    with Image.open(path) as image:
        return image.convert("RGB")


def preprocess_single(
    image: Union[str, Path, Image.Image],
    transform: transforms.Compose = None,
) -> torch.Tensor:
    """Preprocess a single image and return a batch-shaped tensor."""
    if transform is None:
        transform = get_transform()
    if isinstance(image, (str, Path)):
        image = load_image(image)
    tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for visualization."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu()

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (tensor * std + mean).clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
