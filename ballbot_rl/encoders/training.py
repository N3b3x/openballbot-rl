"""Training utilities for encoder pretraining."""
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from pathlib import Path


def train_autoencoder(model, train_loader, val_loader, device, epochs, lr,
                      save_path):
    """
    Train an autoencoder model on depth images.
    
    Args:
        model: The autoencoder model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to train on (e.g., 'cuda' or 'cpu').
        epochs: Number of training epochs.
        lr: Learning rate.
        save_path: Path to save the best encoder model.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x in train_loader:
            x = x.to(device)
            out = model(x)
            loss = torch.nn.functional.mse_loss(out, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            counter = 0
            for x in val_loader:
                x = x.to(device)
                out = model(x)
                loss = torch.nn.functional.mse_loss(out, x)
                val_loss += loss.item() * x.size(0)

                if epoch % 5 == 0 and counter % 50 == 0:
                    out_1 = out[0, 0].detach().cpu().numpy()
                    x_1 = x[0, 0].detach().cpu().numpy()
                    plt.imshow(np.concatenate([out_1, x_1], 1))
                    plt.show()
                counter += 1

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.8f}, val_loss={val_loss:.8f}"
        )
        if val_loss < best_val:
            p_sum = sum([
                param.abs().sum().item()
                for param in model.encoder.parameters() if param.requires_grad
            ])
            print(f"improved val loss, saving **ENCODER** with p_sum={p_sum}")
            model.encoder.p_sum = p_sum
            best_val = val_loss
            save_path_obj = Path(save_path)
            save_path_obj.mkdir(parents=True, exist_ok=True)
            torch.save(model.encoder, f"{save_path}/encoder_epoch_{epoch}")

            torch.save(model, f"{save_path}/full_model_epoch_{epoch}")

