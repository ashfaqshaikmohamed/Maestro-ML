"""
Training script for the CNNGenreClassifier.

This script is intentionally simple and self-contained:
- Loads processed dataset (expects numpy mel files or a torch Dataset)
- Creates model, optimizer, criterion
- Trains for N epochs and saves best checkpoint
- Writes metrics JSON and a simple training log
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.models.cnn_classifier import CNNGenreClassifier
from src.training.evaluate import evaluate_model
from src.training.utils import save_checkpoint, write_metrics, SimpleTimer
from src.data.dataset import MelDataset  # expects processed .npy mel spectrograms
import time


def train(cfg: dict):
    # Unpack config with sensible defaults
    processed_dir = cfg.get("paths", {}).get("processed_dir", "data/processed")
    results_dir = cfg.get("paths", {}).get("results_dir", "results")
    num_classes = cfg.get("model", {}).get("num_classes", 10)
    batch_size = cfg.get("train", {}).get("batch_size", 16)
    epochs = cfg.get("train", {}).get("epochs", 10)
    lr = cfg.get("train", {}).get("lr", 1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset & loaders ---
    dataset = MelDataset(processed_dir)
    n_total = len(dataset)
    if n_total == 0:
        raise RuntimeError(f"No processed data found at {processed_dir}. Run feature extraction first.")

    val_size = int(0.2 * n_total)
    train_size = n_total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model, optimizer, loss ---
    model = CNNGenreClassifier(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(results_dir, exist_ok=True)
    best_val = 0.0
    metrics = {"train": [], "val": []}

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        with SimpleTimer(f"Epoch-{epoch}"):
            for batch in train_loader:
                # Expect tuple (x,y) from MelDataset
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        train_stats = evaluate_model(model, train_loader, device=device)
        val_stats = evaluate_model(model, val_loader, device=device)

        metrics["train"].append({"epoch": epoch, "loss": float(avg_loss), **train_stats})
        metrics["val"].append({"epoch": epoch, **val_stats})

        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - train_acc: {train_stats['accuracy']:.4f} - val_acc: {val_stats['accuracy']:.4f}")

        # Save best model
        if val_stats["accuracy"] > best_val:
            best_val = val_stats["accuracy"]
            ckpt_path = os.path.join(results_dir, "models", "best.pth")
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    # Save metrics to JSON for README/results
    metrics_path = os.path.join(results_dir, "example_metrics.json")
    write_metrics(metrics, metrics_path)
    print(f"Training complete. Metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a music genre classifier")
    parser.add_argument("--config", type=str, default="configs/cnn_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
