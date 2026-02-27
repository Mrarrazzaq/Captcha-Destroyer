"""
Dataset and DataLoader for CAPTCHA images.
Supports both:
  - Classification mode (word → class index)
  - OCR mode (word → character sequence for CTC)
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, CHAR_TO_IDX, WORD_TO_IDX,
    TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS, MAX_LABEL_LENGTH,
)


class CaptchaDataset(Dataset):
    """
    Dataset for CAPTCHA recognition.

    Args:
        data_dir: Path to directory with images and labels.json
        mode: "classification" or "ocr"
        transform: optional torchvision transforms
    """

    def __init__(self, data_dir, mode="classification", transform=None):
        self.data_dir = data_dir
        self.mode = mode

        with open(os.path.join(data_dir, "labels.json"), "r") as f:
            self.labels = json.load(f)

        self.filenames = sorted(self.labels.keys())

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.data_dir, filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        word = self.labels[filename]

        if self.mode == "classification":
            label = WORD_TO_IDX[word]
            return image, label
        else:
            # OCR mode: encode as character indices
            encoded = [CHAR_TO_IDX[c] for c in word]
            label_length = len(encoded)
            # Pad to MAX_LABEL_LENGTH
            encoded += [0] * (MAX_LABEL_LENGTH - len(encoded))
            return image, torch.tensor(encoded, dtype=torch.long), torch.tensor(label_length, dtype=torch.long)


def get_dataloaders(mode="classification"):
    """
    Create train and validation dataloaders.

    Args:
        mode: "classification" or "ocr"

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = CaptchaDataset(TRAIN_DIR, mode=mode)
    val_dataset = CaptchaDataset(VAL_DIR, mode=mode)

    if mode == "classification":
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
    else:
        # For CTC, we need custom collation
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )

    return train_loader, val_loader
