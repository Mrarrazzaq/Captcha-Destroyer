"""
Inference / Prediction script for Captcha Destroyer.

Load a trained model and predict CAPTCHA text from images.

Usage:
  # Single image
  python predict.py --image path/to/captcha.png --mode classification

  # Folder of images
  python predict.py --folder path/to/captchas/ --mode classification

  # Use as Python module
  from predict import CaptchaPredictor
  predictor = CaptchaPredictor(mode="classification")
  result = predictor.predict("captcha.png")
"""

import os
import argparse
import glob
import torch
from torchvision import transforms
from PIL import Image

from config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, DEVICE, MODEL_DIR, CHECKPOINT_PATH,
    IDX_TO_CHAR, IDX_TO_WORD, NUM_WORDS, NUM_CLASSES,
)
from model import ClassifierNet, CRNNNet


class CaptchaPredictor:
    """
    High-level predictor for CAPTCHA images.

    Args:
        mode: "classification" or "ocr"
        model_path: path to checkpoint (auto-detected if None)
        device: "cuda" or "cpu"
    """

    def __init__(self, mode="classification", model_path=None, device=None):
        self.mode = mode
        self.device = device or DEVICE
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        # Auto-detect model path
        if model_path is None:
            if mode == "classification":
                model_path = CHECKPOINT_PATH.replace(".pth", "_classifier.pth")
            else:
                model_path = CHECKPOINT_PATH.replace(".pth", "_ocr.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run train.py first to train a model."
            )

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if mode == "classification":
            self.model = ClassifierNet(num_classes=NUM_WORDS)
        else:
            self.model = CRNNNet(num_classes=NUM_CLASSES)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        print(f"Model loaded from {model_path}")
        print(f"Mode: {mode} | Device: {self.device}")
        if "val_acc" in checkpoint:
            print(f"Model accuracy: {checkpoint['val_acc']:.4f}")

    def predict(self, image_input, return_confidence=True):
        """
        Predict text from a CAPTCHA image.

        Args:
            image_input: file path (str) or PIL Image
            return_confidence: if True, return (text, confidence)

        Returns:
            str or (str, float) depending on return_confidence
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.mode == "classification":
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                text = IDX_TO_WORD[predicted.item()]
                conf = confidence.item()
            else:
                output = self.model(tensor)  # (T, 1, C)
                # Greedy decode
                probs = torch.exp(output)  # log_softmax → softmax
                max_probs, max_indices = torch.max(probs, dim=2)  # (T, 1)

                raw = max_indices[:, 0].tolist()
                raw_probs = max_probs[:, 0].tolist()

                chars = []
                char_probs = []
                prev = None
                for idx, p in zip(raw, raw_probs):
                    if idx != prev:
                        if idx != 0:
                            chars.append(IDX_TO_CHAR.get(idx, "?"))
                            char_probs.append(p)
                        prev = idx

                text = "".join(chars)
                conf = min(char_probs) if char_probs else 0.0

        if return_confidence:
            return text, conf
        return text

    def predict_batch(self, image_paths):
        """
        Predict multiple images.

        Returns:
            List of (filename, predicted_text, confidence)
        """
        results = []
        for path in image_paths:
            try:
                text, conf = self.predict(path)
                results.append((os.path.basename(path), text, conf))
            except Exception as e:
                results.append((os.path.basename(path), f"ERROR: {e}", 0.0))
        return results


def main():
    parser = argparse.ArgumentParser(description="Predict CAPTCHA text")
    parser.add_argument("--image", type=str, help="Path to a single CAPTCHA image")
    parser.add_argument("--folder", type=str, help="Path to a folder of CAPTCHA images")
    parser.add_argument(
        "--mode", type=str, default="classification",
        choices=["classification", "ocr"],
    )
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    predictor = CaptchaPredictor(mode=args.mode, model_path=args.model, device=args.device)

    if args.image:
        text, confidence = predictor.predict(args.image)
        print(f"\nResult: '{text}' (confidence: {confidence:.4f})")

    elif args.folder:
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif")
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(args.folder, ext)))
        files.sort()

        if not files:
            print(f"No images found in {args.folder}")
            return

        print(f"\nPredicting {len(files)} images...\n")
        results = predictor.predict_batch(files)

        print(f"{'Filename':<30} {'Prediction':<20} {'Confidence':<10}")
        print("-" * 60)
        for filename, text, conf in results:
            print(f"{filename:<30} {text:<20} {conf:.4f}")

        # Summary
        print(f"\nTotal: {len(results)} images processed")
    else:
        print("Provide --image or --folder. Use --help for usage.")


if __name__ == "__main__":
    main()
