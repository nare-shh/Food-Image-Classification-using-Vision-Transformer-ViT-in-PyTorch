import torch
from torch import nn
from typing import Dict, List, Tuple
from tqdm.auto import tqdm


class Trainer:
    """Trainer class for ViT models."""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
    def train_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Perform a single training step."""
        self.model.train()
        
        train_loss, train_acc = 0, 0
        
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)

            # Forward pass
            y_pred = self.model(X)

            # Calculate and accumulate loss
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            # Optimizer zero grad
            self.optimizer.zero_grad()

            # Loss backward
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        
        return train_loss, train_acc

    def test_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Perform a single test step."""
        self.model.eval()
        
        test_loss, test_acc = 0, 0
        
        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                # Send data to target device
                X, y = X.to(self.device), y.to(self.device)

                # Forward pass
                test_pred_logits = self.model(X)

                # Calculate and accumulate loss
                loss = self.loss_fn(test_pred_logits, y)
                test_loss += loss.item()

                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        
        return test_loss, test_acc

    def train(self,
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              epochs: int = 5) -> Dict[str, List]:
        """Train the model for a given number of epochs."""
        
        # Create empty results dictionary
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }

        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.train_step(train_dataloader)
            test_loss, test_acc = self.test_step(test_dataloader)

            # Print out what's happening
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        # Return the filled results at the end of the epochs
        return results
