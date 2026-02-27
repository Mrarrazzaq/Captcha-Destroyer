"""
Neural Network Models for CAPTCHA Recognition.

Two architectures:
1. ClassifierNet — treats it as multi-class classification (pick word from list)
2. CRNNNet — CRNN with CTC loss for character-level OCR

For limited vocabulary, ClassifierNet is simpler and often more accurate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES, NUM_WORDS, IMAGE_HEIGHT, IMAGE_WIDTH


class ClassifierNet(nn.Module):
    """
    CNN-based classifier: input image → one of NUM_WORDS classes.
    Best when vocabulary is small and fixed.
    """

    def __init__(self, num_classes=NUM_WORDS):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            feat = self.features(dummy)
            self._flat_size = feat.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(self._flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CRNNNet(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) with CTC.
    Input image → feature maps → bidirectional LSTM → per-timestep character probabilities.
    Good for variable-length text or when vocabulary might expand.
    """

    def __init__(self, num_classes=NUM_CLASSES, hidden_size=256):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2, W/2

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4, W/4

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8, W/4

            # Block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 6
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/16, W/4
        )

        # Calculate RNN input size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            feat = self.cnn(dummy)
            # feat shape: (batch, channels, height, width)
            # We collapse height into channels for the RNN
            self._rnn_input_size = feat.size(1) * feat.size(2)
            self._seq_length = feat.size(3)

        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=self._rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # CNN
        conv = self.cnn(x)  # (B, C, H, W)
        b, c, h, w = conv.size()

        # Reshape: collapse height into channels, width becomes sequence
        conv = conv.view(b, c * h, w)  # (B, C*H, W)
        conv = conv.permute(0, 2, 1)    # (B, W, C*H) — sequence of feature vectors

        # RNN
        rnn_out, _ = self.rnn(conv)  # (B, W, hidden*2)

        # Per-timestep classification
        output = self.fc(rnn_out)  # (B, W, num_classes)

        # For CTC: need (T, B, C) format
        output = output.permute(1, 0, 2)  # (T, B, C)

        return F.log_softmax(output, dim=2)

    @property
    def seq_length(self):
        return self._seq_length
