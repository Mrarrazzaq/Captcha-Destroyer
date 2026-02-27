"""
Tambah sample baru ke dataset dan (opsional) langsung finetune model.

Usage:
  # Tambah 1 gambar + label
  python add_sample.py --image test_images/xxx.png --label TABUNGAN

  # Tambah + langsung finetune
  python add_sample.py --image test_images/xxx.png --label TABUNGAN --finetune

  # Tambah banyak gambar sekaligus (pisah koma)
  python add_sample.py --image "img1.png,img2.png" --label "TABUNGAN,BAYAR"

  # Lihat semua sample yang sudah ditambahkan
  python add_sample.py --list
"""

import os
import json
import shutil
import argparse

REAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_data")
LABELS_FILE = os.path.join(REAL_DATA_DIR, "labels.json")


def load_labels():
    """Load existing labels or return empty dict."""
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_labels(labels):
    """Save labels to JSON."""
    os.makedirs(REAL_DATA_DIR, exist_ok=True)
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)


def add_sample(image_path, label):
    """
    Tambah satu sample ke real_data/.
    Copy gambar ke real_data/ dan update labels.json.
    """
    if not os.path.exists(image_path):
        print(f"ERROR: File tidak ditemukan: {image_path}")
        return False

    label = label.upper().strip()

    # Validasi label ada di vocab
    from config import WORDS
    if label not in WORDS:
        print(f"WARNING: '{label}' tidak ada di vocab WORDS di config.py")
        print(f"  Vocab saat ini: {', '.join(WORDS)}")
        resp = input("  Tetap tambahkan? (y/n): ").strip().lower()
        if resp != "y":
            print("  Dibatalkan.")
            return False

    os.makedirs(REAL_DATA_DIR, exist_ok=True)

    # Copy gambar ke real_data/
    filename = os.path.basename(image_path)
    dest_path = os.path.join(REAL_DATA_DIR, filename)
    src_abs = os.path.abspath(image_path)
    dest_abs = os.path.abspath(dest_path)

    if src_abs == dest_abs:
        # File sudah ada di real_data/, skip copy
        pass
    elif os.path.exists(dest_path):
        # File lain dengan nama sama, rename
        name, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(dest_path):
            filename = f"{name}_{counter}{ext}"
            dest_path = os.path.join(REAL_DATA_DIR, filename)
            counter += 1
        shutil.copy2(image_path, dest_path)
    else:
        shutil.copy2(image_path, dest_path)

    # Update labels.json
    labels = load_labels()
    labels[filename] = label
    save_labels(labels)

    print(f"  + {filename} -> {label}")
    return True


def list_samples():
    """Tampilkan semua sample yang sudah ditambahkan."""
    labels = load_labels()
    if not labels:
        print("Belum ada sample di real_data/")
        return

    print(f"\nTotal: {len(labels)} sample di real_data/\n")
    print(f"{'Filename':<40} {'Label':<20}")
    print("-" * 60)
    for filename, label in sorted(labels.items()):
        exists = "OK" if os.path.exists(os.path.join(REAL_DATA_DIR, filename)) else "MISSING"
        print(f"{filename:<40} {label:<20} [{exists}]")
    print()


def run_finetune(device="cpu", epochs=20, lr=0.0005):
    """Jalankan finetune dengan data di real_data/."""
    labels = load_labels()
    if len(labels) < 1:
        print("ERROR: Belum ada sample untuk finetune.")
        return

    print(f"\nMemulai finetune dengan {len(labels)} sample...")
    print(f"  Device: {device} | Epochs: {epochs} | LR: {lr}\n")

    # Import dan jalankan finetune
    from finetune import finetune
    finetune(
        data_dir=REAL_DATA_DIR,
        mode="classification",
        epochs=epochs,
        lr=lr,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="Tambah sample CAPTCHA + finetune model")
    parser.add_argument("--image", type=str, help="Path gambar (pisah koma untuk banyak)")
    parser.add_argument("--label", type=str, help="Label/kata (pisah koma untuk banyak)")
    parser.add_argument("--finetune", action="store_true", help="Langsung finetune setelah tambah sample")
    parser.add_argument("--list", action="store_true", help="Tampilkan semua sample")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--epochs", type=int, default=20, help="Jumlah epoch finetune")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate finetune")
    args = parser.parse_args()

    if args.list:
        list_samples()
        return

    if not args.image or not args.label:
        parser.print_help()
        return

    # Parse multiple images & labels
    images = [x.strip() for x in args.image.split(",")]
    labels = [x.strip() for x in args.label.split(",")]

    if len(images) != len(labels):
        print(f"ERROR: Jumlah gambar ({len(images)}) != jumlah label ({len(labels)})")
        return

    print(f"Menambahkan {len(images)} sample ke real_data/...\n")
    success = 0
    for img, lbl in zip(images, labels):
        if add_sample(img, lbl):
            success += 1

    print(f"\n{success}/{len(images)} sample berhasil ditambahkan.")
    list_samples()

    if args.finetune:
        run_finetune(device=args.device, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
