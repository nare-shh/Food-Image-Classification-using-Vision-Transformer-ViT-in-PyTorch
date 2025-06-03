from vit_modules.models import ViT
from vit_modules.training import Trainer
from vit_modules.data import get_transforms
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn as nn

# Set up data
transform = get_transforms(img_size=224)
train_dataset = ImageFolder('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, optimizer, loss
model = ViT(img_size=224, in_channels=3, num_classes=5)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

# Train
trainer = Trainer(model, optimizer, loss_fn)
trainer.train_step(train_loader)
