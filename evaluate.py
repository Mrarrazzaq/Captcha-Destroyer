"""
Evaluate model accuracy on validation set or a labeled test set.

Usage:
  python evaluate.py --mode classification
  python evaluate.py --mode ocr
  python evaluate.py --mode classification --data-dir path/to/test/
"""

import os
import json
import argparse
import torch
from torchvision import transforms
from PIL import Image
from collections import defaultdict

from config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, DEVICE, VAL_DIR,
    IDX_TO_CHAR, IDX_TO_WORD, WORD_TO_IDX, CHAR_TO_IDX,
    NUM_WORDS, NUM_CLASSES, CHECKPOINT_PATH,
)
from model import ClassifierNet, CRNNNet


def evaluate_classification(model_path, data_dir, device):
    """Evaluate classification model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = ClassifierNet(num_classes=NUM_WORDS)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    with open(os.path.join(data_dir, "labels.json"), "r") as f:
        labels = json.load(f)

    correct = 0
    total = 0
    per_word_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    errors = []

    with torch.no_grad():
        for filename, true_word in labels.items():
            img_path = os.path.join(data_dir, filename)
            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            pred_word = IDX_TO_WORD[predicted.item()]

            total += 1
            per_word_stats[true_word]["total"] += 1

            if pred_word == true_word:
                correct += 1
                per_word_stats[true_word]["correct"] += 1
            else:
                errors.append((filename, true_word, pred_word, confidence.item()))

    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"  Classification Evaluation Results")
    print(f"{'='*60}")
    print(f"  Total: {total} | Correct: {correct} | Accuracy: {accuracy:.4f}")
    print(f"\n  Per-word accuracy:")
    print(f"  {'Word':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print(f"  {'-'*45}")
    for word in sorted(per_word_stats.keys()):
        stats = per_word_stats[word]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {word:<15} {stats['correct']:<10} {stats['total']:<10} {acc:.4f}")

    if errors:
        print(f"\n  Sample errors (showing first 10):")
        for fn, true, pred, conf in errors[:10]:
            print(f"    {fn}: true='{true}' pred='{pred}' conf={conf:.4f}")

    return accuracy


def evaluate_ocr(model_path, data_dir, device):
    """Evaluate OCR model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = CRNNNet(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    with open(os.path.join(data_dir, "labels.json"), "r") as f:
        labels = json.load(f)

    correct = 0
    total = 0
    char_correct = 0
    char_total = 0
    errors = []

    with torch.no_grad():
        for filename, true_word in labels.items():
            img_path = os.path.join(data_dir, filename)
            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            output = model(tensor)
            _, max_indices = torch.max(output, dim=2)
            raw = max_indices[:, 0].tolist()

            # Decode
            chars = []
            prev = None
            for idx in raw:
                if idx != prev:
                    if idx != 0:
                        chars.append(IDX_TO_CHAR.get(idx, "?"))
                    prev = idx
            pred_word = "".join(chars)

            total += 1
            char_total += len(true_word)

            if pred_word == true_word:
                correct += 1
                char_correct += len(true_word)
            else:
                # Count matching chars
                for i, c in enumerate(true_word):
                    if i < len(pred_word) and pred_word[i] == c:
                        char_correct += 1
                errors.append((filename, true_word, pred_word))

    word_acc = correct / total if total > 0 else 0
    char_acc = char_correct / char_total if char_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  OCR Evaluation Results")
    print(f"{'='*60}")
    print(f"  Word Accuracy: {word_acc:.4f} ({correct}/{total})")
    print(f"  Char Accuracy: {char_acc:.4f} ({char_correct}/{char_total})")

    if errors:
        print(f"\n  Sample errors (showing first 10):")
        for fn, true, pred in errors[:10]:
            print(f"    {fn}: true='{true}' pred='{pred}'")

    return word_acc, char_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAPTCHA model")
    parser.add_argument(
        "--mode", type=str, default="classification",
        choices=["classification", "ocr"],
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Custom test data dir")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    data_dir = args.data_dir or VAL_DIR

    if args.mode == "classification":
        model_path = args.model or CHECKPOINT_PATH.replace(".pth", "_classifier.pth")
        evaluate_classification(model_path, data_dir, device)
    else:
        model_path = args.model or CHECKPOINT_PATH.replace(".pth", "_ocr.pth")
        evaluate_ocr(model_path, data_dir, device)


if __name__ == "__main__":
    main()
