"""
Fine-tune model with REAL captcha images.

When you have actual CAPTCHA samples, use this to fine-tune the
pre-trained model for significantly better accuracy.

Directory structure for real data:
  real_data/
    labels.json          ← {"image1.png": "tabungan", "image2.png": "bayar", ...}
    image1.png
    image2.png
    ...

Usage:
  python finetune.py --data-dir real_data/ --mode classification
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image

from config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, DEVICE, MODEL_DIR, CHECKPOINT_PATH,
    NUM_WORDS, NUM_CLASSES, WORD_TO_IDX, CHAR_TO_IDX, MAX_LABEL_LENGTH,
    IDX_TO_WORD, IDX_TO_CHAR,
)
from model import ClassifierNet, CRNNNet
from dataset import CaptchaDataset


def finetune(data_dir, mode, epochs, lr, device):
    """Fine-tune a pre-trained model on real data."""

    # Check data
    labels_path = os.path.join(data_dir, "labels.json")
    if not os.path.exists(labels_path):
        print(f"ERROR: labels.json not found in {data_dir}")
        print("Create a labels.json with format: {\"filename.png\": \"word\", ...}")
        return

    with open(labels_path) as f:
        labels = json.load(f)
    print(f"Found {len(labels)} labeled images in {data_dir}")

    # Load pre-trained model
    if mode == "classification":
        base_model_path = CHECKPOINT_PATH.replace(".pth", "_classifier.pth")
        model = ClassifierNet(num_classes=NUM_WORDS)
    else:
        base_model_path = CHECKPOINT_PATH.replace(".pth", "_ocr.pth")
        model = CRNNNet(num_classes=NUM_CLASSES)

    if os.path.exists(base_model_path):
        checkpoint = torch.load(base_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded pre-trained model from {base_model_path}")
    else:
        print("No pre-trained model found. Training from scratch on real data.")

    model.to(device)

    # Dataset
    dataset = CaptchaDataset(data_dir, mode=mode)

    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if mode == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_total = 0

        if mode == "classification":
            for images, label_indices in train_loader:
                images = images.to(device)
                label_indices = label_indices.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, label_indices)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
                train_total += images.size(0)
        else:
            seq_length = model.seq_length
            for images, targets, target_lengths in train_loader:
                batch_size = images.size(0)
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                input_lengths = torch.full(
                    (batch_size,), seq_length, dtype=torch.long, device=device
                )
                loss = criterion(outputs, targets, input_lengths, target_lengths)
                if torch.isfinite(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    train_loss += loss.item() * batch_size
                train_total += batch_size

        train_loss /= train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            if mode == "classification":
                for images, label_indices in val_loader:
                    images = images.to(device)
                    label_indices = label_indices.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == label_indices).sum().item()
                    val_total += label_indices.size(0)
            else:
                for images, targets, target_lengths in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    _, max_indices = torch.max(outputs, dim=2)
                    for i in range(images.size(0)):
                        raw = max_indices[:, i].tolist()
                        chars = []
                        prev = None
                        for idx in raw:
                            if idx != prev:
                                if idx != 0:
                                    chars.append(IDX_TO_CHAR.get(idx, "?"))
                                prev = idx
                        pred_word = "".join(chars)
                        target_chars = targets[i][:target_lengths[i]].tolist()
                        true_word = "".join(IDX_TO_CHAR.get(c, "?") for c in target_chars)
                        if pred_word == true_word:
                            val_correct += 1
                        val_total += 1

        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f} Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = CHECKPOINT_PATH.replace(".pth", f"_finetuned_{mode}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "mode": mode,
            }, save_path)
            print(f"  → Saved best fine-tuned model (Acc: {val_acc:.4f})")

    print(f"\nFine-tuning complete! Best accuracy: {best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on real CAPTCHA data")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with real CAPTCHAs + labels.json")
    parser.add_argument("--mode", type=str, default="classification", choices=["classification", "ocr"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (lower for fine-tuning)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    finetune(args.data_dir, args.mode, args.epochs, args.lr, device)


if __name__ == "__main__":
    main()
