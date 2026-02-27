"""
Synthetic CAPTCHA Generator
Generates training images that mimic the target CAPTCHA style.

Target CAPTCHA characteristics (from sample):
  - UPPERCASE text (e.g. "BAYAR")
  - White background
  - Dark/black text
  - Many diagonal lines crossing through the image
  - Blue border around the image
  - Serif-like font (Times New Roman style)
  - Image ~150x40 px
"""

import os
import random
import math
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from config import (
    WORDS, IMAGE_WIDTH, IMAGE_HEIGHT, BACKGROUND_COLOR, TEXT_COLOR,
    NOISE_LEVEL, ADD_LINES, LINE_COUNT, ADD_DOTS, DOT_COUNT,
    FONT_SIZE_RANGE, DISTORTION_LEVEL, FONT_DIR, TRAIN_DIR, VAL_DIR,
    TRAIN_SAMPLES, VAL_SAMPLES, ADD_BORDER, BORDER_COLOR, LINE_COLOR_RANGE,
)


def get_fonts():
    """
    Load available fonts from the fonts directory, or fall back to default.
    Prioritizes serif fonts (Times New Roman, Georgia) to match target CAPTCHA.
    """
    fonts = []
    if os.path.isdir(FONT_DIR):
        for f in os.listdir(FONT_DIR):
            if f.lower().endswith((".ttf", ".otf")):
                fonts.append(os.path.join(FONT_DIR, f))
    if not fonts:
        # Fallback: prioritize serif fonts to match target CAPTCHA style
        system_fonts = [
            # Serif fonts (preferred — matches target)
            "times.ttf",
            "timesbd.ttf",
            "timesi.ttf",
            "georgia.ttf",
            "georgiab.ttf",
            # Sans-serif fallbacks
            "arial.ttf",
            "arialbd.ttf",
            "verdana.ttf",
            "tahoma.ttf",
            "calibri.ttf",
        ]
        for sf in system_fonts:
            try:
                ImageFont.truetype(sf, 20)
                fonts.append(sf)
            except (IOError, OSError):
                pass
    if not fonts:
        fonts = [None]  # will use default PIL font
    return fonts


def add_noise(image, level):
    """Add salt-and-pepper noise."""
    arr = np.array(image)
    noise_mask = np.random.random(arr.shape[:2])
    arr[noise_mask < level / 2] = 0
    arr[noise_mask > 1 - level / 2] = 255
    return Image.fromarray(arr)


def add_diagonal_lines(draw, width, height, count_range, color_range):
    """
    Draw diagonal lines across the image — mimics the target CAPTCHA style.
    Lines go from one edge to another at various angles, similar to the sample.
    """
    n = random.randint(*count_range)
    lo, hi = color_range

    for _ in range(n):
        # Generate lines that cross the image diagonally
        # Pick start on left/top edge, end on right/bottom edge
        pattern = random.choice(["left_to_right", "top_to_bottom", "cross"])

        if pattern == "left_to_right":
            # Start from left edge, end at right edge
            x1 = 0
            y1 = random.randint(0, height)
            x2 = width
            y2 = random.randint(0, height)
        elif pattern == "top_to_bottom":
            # Start from top edge, end at bottom edge
            x1 = random.randint(0, width)
            y1 = 0
            x2 = random.randint(0, width)
            y2 = height
        else:
            # Full diagonal cross
            x1 = random.randint(-10, width // 4)
            y1 = random.randint(-10, height)
            x2 = random.randint(width * 3 // 4, width + 10)
            y2 = random.randint(-10, height)

        # Color: grayish-blue like the target
        gray = random.randint(lo, hi)
        blue_boost = random.randint(0, 40)
        color = (
            max(0, gray - random.randint(0, 30)),
            max(0, gray - random.randint(0, 20)),
            min(255, gray + blue_boost),
        )

        draw.line([(x1, y1), (x2, y2)], fill=color, width=1)


def add_random_dots(draw, width, height, count_range):
    """Scatter random dots."""
    n = random.randint(*count_range)
    for _ in range(n):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        draw.point((x, y), fill=color)


def apply_distortion(image, level):
    """Simple wave/shift distortion."""
    arr = np.array(image)
    result = np.copy(arr)
    rows, cols = arr.shape[:2]
    for i in range(rows):
        shift = int(level * math.sin(2 * math.pi * i / (rows / 3)))
        result[i] = np.roll(arr[i], shift, axis=0)
    return Image.fromarray(result)


def generate_captcha(word, fonts, variation=True):
    """
    Generate a single CAPTCHA image for the given word.
    Mimics the target style: uppercase text, diagonal lines, blue border.

    Args:
        word: The text to render (should be UPPERCASE)
        fonts: List of font paths
        variation: Whether to add random variations

    Returns:
        PIL Image
    """
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Choose font
    font_path = random.choice(fonts)
    font_size = random.randint(*FONT_SIZE_RANGE)

    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Calculate text position (roughly centered with slight randomness)
    bbox = draw.textbbox((0, 0), word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Center with slight offset like the target
    center_x = (IMAGE_WIDTH - tw) // 2
    center_y = (IMAGE_HEIGHT - th) // 2
    x = center_x + random.randint(-8, 8) if variation else center_x
    y = center_y + random.randint(-3, 3) if variation else center_y
    x = max(3, min(x, IMAGE_WIDTH - tw - 3))
    y = max(2, min(y, IMAGE_HEIGHT - th - 2))

    # Slight color variation on text
    if variation:
        r, g, b = TEXT_COLOR
        dr, dg, db = random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)
        text_color = (
            max(0, min(255, r + dr)),
            max(0, min(255, g + dg)),
            max(0, min(255, b + db)),
        )
    else:
        text_color = TEXT_COLOR

    # Draw text — character by character with slight vertical offset (like target)
    if variation and random.random() > 0.3:
        cx = x
        for ch in word:
            char_bbox = draw.textbbox((0, 0), ch, font=font)
            char_w = char_bbox[2] - char_bbox[0]
            cy_offset = random.randint(-2, 2)
            draw.text((cx, y + cy_offset), ch, fill=text_color, font=font)
            cx += char_w + random.randint(0, 2)
    else:
        draw.text((x, y), word, fill=text_color, font=font)

    # Add diagonal lines (the main noise feature of target CAPTCHA!)
    if ADD_LINES:
        add_diagonal_lines(draw, IMAGE_WIDTH, IMAGE_HEIGHT, LINE_COUNT, LINE_COLOR_RANGE)

    if ADD_DOTS:
        add_random_dots(draw, IMAGE_WIDTH, IMAGE_HEIGHT, DOT_COUNT)

    if NOISE_LEVEL > 0:
        img = add_noise(img, NOISE_LEVEL)

    if DISTORTION_LEVEL > 0 and variation:
        img = apply_distortion(img, DISTORTION_LEVEL)

    # Add border (like target CAPTCHA has a blue border)
    if ADD_BORDER:
        draw = ImageDraw.Draw(img)
        border_color = BORDER_COLOR
        if variation:
            br, bg, bb = border_color
            border_color = (
                max(0, min(255, br + random.randint(-20, 20))),
                max(0, min(255, bg + random.randint(-20, 20))),
                min(255, bb + random.randint(-10, 10)),
            )
        draw.rectangle(
            [(0, 0), (IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1)],
            outline=border_color, width=1,
        )

    # Slight blur occasionally
    if variation and random.random() > 0.8:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))

    return img


def generate_dataset(output_dir, num_samples, fonts):
    """
    Generate a full dataset of CAPTCHA images.

    Saves images as PNG and creates a labels.json mapping.
    """
    os.makedirs(output_dir, exist_ok=True)
    labels = {}

    for i in range(num_samples):
        word = random.choice(WORDS)
        img = generate_captcha(word, fonts)
        filename = f"{i:06d}.png"
        img.save(os.path.join(output_dir, filename))
        labels[filename] = word

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples} images...")

    # Save labels
    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    print(f"  Done! {num_samples} images saved to {output_dir}")
    return labels


def main():
    print("=" * 60)
    print("  Captcha Destroyer — Synthetic Data Generator")
    print("=" * 60)

    fonts = get_fonts()
    print(f"\nFound {len(fonts)} font(s)")
    print(f"Vocabulary: {len(WORDS)} words")
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print()

    print("[1/2] Generating training data...")
    generate_dataset(TRAIN_DIR, TRAIN_SAMPLES, fonts)

    print("[2/2] Generating validation data...")
    generate_dataset(VAL_DIR, VAL_SAMPLES, fonts)

    print("\nAll done! Data ready for training.")


if __name__ == "__main__":
    main()
