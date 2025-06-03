"""Visualization utilities."""

import matplotlib.pyplot as plt
import torch
from typing import Dict, List
import numpy as np


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plot training and test loss/accuracy curves."""
    epochs = range(len(results['train_loss']))
    
    plt.figure(figsize=(15, 7))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['train_loss'], label='train_loss')
    plt.plot(epochs, results['test_loss'], label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['train_acc'], label='train_accuracy')
    plt.plot(epochs, results['test_acc'], label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_patches(image: torch.Tensor,
                     patch_size: int = 16,
                     class_names: List[str] = None,
                     label: int = None):
    """Visualize how an image is divided into patches."""
    # Convert image for matplotlib (C, H, W) -> (H, W, C)
    image_permuted = image.permute(1, 2, 0)
    
    img_size = image.shape[-1]
    num_patches = img_size // patch_size
    
    # Create subplot grid
    fig, axs = plt.subplots(
        nrows=num_patches,
        ncols=num_patches,
        figsize=(num_patches, num_patches),
        sharex=True,
        sharey=True
    )
    
    # Loop through height and width of image
    for i, patch_height in enumerate(range(0, img_size, patch_size)):
        for j, patch_width in enumerate(range(0, img_size, patch_size)):
            # Plot the patch
            axs[i, j].imshow(
                image_permuted[
                    patch_height:patch_height + patch_size,
                    patch_width:patch_width + patch_size,
                    :
                ]
            )
            
            # Set labels and remove ticks
            axs[i, j].set_ylabel(
                i + 1,
                rotation="horizontal",
                horizontalalignment="right",
                verticalalignment="center"
            )
            axs[i, j].set_xlabel(j + 1)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].label_outer()
    
    # Set title
    if class_names and label is not None:
        fig.suptitle(f"{class_names[label]} -> Patchified", fontsize=16)
    else:
        fig.suptitle("Image -> Patchified", fontsize=16)
    
    plt.show()