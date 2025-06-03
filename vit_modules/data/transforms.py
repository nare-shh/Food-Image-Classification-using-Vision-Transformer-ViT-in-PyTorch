"""Data transformation utilities."""

import torch
import torchvision
from torchvision import transforms


def get_transforms(img_size: int = 224):
    """Get basic transforms for ViT training.
    
    Args:
        img_size (int): Target image size. Defaults to 224.
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def get_pretrained_transforms(model_name: str = "vit_b_16"):
    """Get transforms from pretrained ViT weights.
    
    Args:
        model_name (str): Name of the pretrained model. Defaults to "vit_b_16".
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline from pretrained weights.
    """
    if model_name == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        return weights.transforms()
    else:
        raise ValueError(f"Unsupported model: {model_name}")