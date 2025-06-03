"""Prediction utilities for ViT models."""

import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple
from pathlib import Path


class Predictor:
    """Predictor class for ViT models."""
    
    def __init__(self,
                 model: nn.Module,
                 class_names: List[str],
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.class_names = class_names
        self.device = device
        
    def predict(self, 
                image_path: str,
                transform=None) -> Tuple[str, float]:
        """Make a prediction on a single image."""
        self.model.eval()
        
        # Load image
        image = Image.open(image_path)
        
        # Apply transforms if provided
        if transform:
            image = transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.inference_mode():
            pred_logits = self.model(image)
            pred_probs = torch.softmax(pred_logits, dim=1)
            pred_label = torch.argmax(pred_probs, dim=1)
            pred_class = self.class_names[pred_label.cpu()]
            pred_prob = pred_probs.max().cpu().item()
            
        return pred_class, pred_prob
    
    def predict_and_plot(self,
                        image_path: str,
                        transform=None,
                        figsize: Tuple[int, int] = (10, 7)):
        """Make a prediction and plot the result."""
        # Make prediction
        pred_class, pred_prob = self.predict(image_path, transform)
        
        # Plot image with prediction
        plt.figure(figsize=figsize)
        
        # Load and display image
        image = Image.open(image_path)
        plt.imshow(image)
        plt.title(f"Pred: {pred_class} | Prob: {pred_prob:.3f}")
        plt.axis(False)
        plt.show()
        
        return pred_class, pred_prob
