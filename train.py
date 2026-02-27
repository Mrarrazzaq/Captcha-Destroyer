"""
Training pipeline for Captcha Destroyer.

Supports two modes:
  - classification: word-level classification (recommended for small vocab)
  - ocr: character-level CRNN + CTC (for extensibility)

Usage:
  python train.py --mode classification
  python train.py --mode ocr
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE,
    MODEL_DIR, CHECKPOINT_PATH, IDX_TO_CHAR, IDX_TO_WORD,
    MAX_LABEL_LENGTH, NUM_WORDS, NUM_CLASSES,
)
from dataset import get_dataloaders
from model import ClassifierNet, CRNNNet


def decode_ctc(output, idx_to_char):
    """Decode CTC output using greedy (best-path) decoding."""
    # output: (T, B, C)
    _, max_indices = torch.max(output, dim=2)  # (T, B)
    decoded = []
    for b in range(max_indices.size(1)):
        raw = max_indices[:, b].tolist()
        # Collapse repeats and remove blanks (0)
        chars = []
        prev = None
        for idx in raw:
            if idx != prev:
                if idx != 0:
                    chars.append(idx_to_char.get(idx, "?"))
                prev = idx
        decoded.append("".join(chars))
    return decoded


def train_classification(num_epochs, device):
    """Train the classification model."""
    print("=" * 60)
    print("  Training: Classification Mode")
    print(f"  Device: {device}")
    print("=" * 60)

    train_loader, val_loader = get_dataloaders(mode="classification")
    model = ClassifierNet(num_classes=NUM_WORDS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    best_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(num_epochs):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = CHECKPOINT_PATH.replace(".pth", "_classifier.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "mode": "classification",
            }, save_path)
            print(f"  → New best model saved! Acc: {val_acc:.4f}")

        # Early stopping: if we reach near-perfect accuracy
        if val_acc >= 0.995:
            print(f"  → Reached {val_acc:.4f} accuracy! Stopping early.")
            break

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    return best_acc


def train_ocr(num_epochs, device):
    """Train the CRNN + CTC model."""
    print("=" * 60)
    print("  Training: OCR Mode (CRNN + CTC)")
    print(f"  Device: {device}")
    print("=" * 60)

    train_loader, val_loader = get_dataloaders(mode="ocr")
    model = CRNNNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_loss = float("inf")
    best_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)

    seq_length = model.seq_length

    for epoch in range(num_epochs):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_total = 0

        for images, targets, target_lengths in train_loader:
            batch_size = images.size(0)
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # (T, B, C)

            # Input lengths: all same = sequence length of model
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

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0

        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                batch_size = images.size(0)
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                outputs = model(images)
                input_lengths = torch.full(
                    (batch_size,), seq_length, dtype=torch.long, device=device
                )

                loss = criterion(outputs, targets, input_lengths, target_lengths)
                if torch.isfinite(loss):
                    val_loss += loss.item() * batch_size
                val_total += batch_size

                # Decode and check accuracy
                decoded = decode_ctc(outputs, IDX_TO_CHAR)
                for i, pred_word in enumerate(decoded):
                    target_chars = targets[i][:target_lengths[i]].tolist()
                    true_word = "".join(IDX_TO_CHAR.get(c, "?") for c in target_chars)
                    if pred_word == true_word:
                        val_correct += 1

        val_loss /= val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} Word Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        scheduler.step(val_loss)

        # Save best model (by accuracy)
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = CHECKPOINT_PATH.replace(".pth", "_ocr.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "mode": "ocr",
            }, save_path)
            print(f"  → New best model saved! Word Acc: {val_acc:.4f}")

        if val_acc >= 0.995:
            print(f"  → Reached {val_acc:.4f} accuracy! Stopping early.")
            break

    print(f"\nBest validation word accuracy: {best_acc:.4f}")
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Train CAPTCHA recognition model")
    parser.add_argument(
        "--mode", type=str, default="classification",
        choices=["classification", "ocr"],
        help="Training mode: 'classification' (recommended) or 'ocr'"
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Using device: {device}")

    if args.mode == "classification":
        train_classification(args.epochs, device)
    else:
        train_ocr(args.epochs, device)


if __name__ == "__main__":
    main()
