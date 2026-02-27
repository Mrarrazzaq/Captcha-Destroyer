# Captcha Destroyer 🔍

ML-based CAPTCHA text recognition untuk keperluan **testing** di lingkungan development.

## Fitur

- **Synthetic Data Generator** — generate ribuan training CAPTCHA secara otomatis
- **Dua mode ML**:
  - **Classification** — cocok untuk vocabulary terbatas (recommended)
  - **OCR (CRNN+CTC)** — cocok jika kata bisa bertambah
- **Fine-tuning** — bisa fine-tune dengan data CAPTCHA asli
- **Inference** — predict single image atau batch folder

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Opsional) Tambahkan Font

Taruh file `.ttf` / `.otf` di folder `fonts/`. Jika tidak ada, akan pakai system fonts.

### 3. Generate Training Data

```bash
python generate_data.py
```

Ini akan membuat 20,000 training + 2,000 validasi images di `data/`.

### 4. Train Model

```bash
# Mode classification (recommended untuk vocab kecil)
python train.py --mode classification

# Mode OCR (CRNN + CTC)
python train.py --mode ocr

# Custom epochs
python train.py --mode classification --epochs 30

# Force CPU
python train.py --mode classification --device cpu
```

### 5. Evaluate

```bash
python evaluate.py --mode classification
python evaluate.py --mode ocr
```

### 6. Predict

```bash
# Single image
python predict.py --image path/to/captcha.png --mode classification

# Folder of images
python predict.py --folder path/to/captchas/ --mode classification
```

### 7. (Opsional) Fine-tune dengan Data Asli

Jika punya sampel CAPTCHA asli, bisa fine-tune untuk akurasi lebih tinggi:

```
real_data/
  labels.json      ← {"img1.png": "tabungan", "img2.png": "bayar", ...}
  img1.png
  img2.png
  ...
```

```bash
python finetune.py --data-dir real_data/ --mode classification
```

## Menggunakan sebagai Module Python

```python
from predict import CaptchaPredictor

# Initialize
predictor = CaptchaPredictor(mode="classification")

# Predict single image
text, confidence = predictor.predict("captcha.png")
print(f"Text: {text}, Confidence: {confidence:.4f}")

# Predict batch
results = predictor.predict_batch(["cap1.png", "cap2.png", "cap3.png"])
for filename, text, conf in results:
    print(f"{filename}: {text} ({conf:.4f})")
```

## Konfigurasi

Edit `config.py` untuk menyesuaikan:

| Setting | Default | Keterangan |
|---------|---------|------------|
| `WORDS` | 30 kata | Daftar kata yang muncul di CAPTCHA |
| `IMAGE_WIDTH` | 200 | Lebar gambar |
| `IMAGE_HEIGHT` | 60 | Tinggi gambar |
| `TEXT_COLOR` | (0,0,0) | Warna teks CAPTCHA |
| `NOISE_LEVEL` | 0.05 | Level noise (0-1) |
| `TRAIN_SAMPLES` | 20000 | Jumlah training images |
| `BATCH_SIZE` | 64 | Batch size training |
| `NUM_EPOCHS` | 50 | Max epochs |
| `DEVICE` | "cuda" | Device (cuda/cpu) |

## Struktur Project

```
Captcha Destroyer/
├── config.py          # Semua konfigurasi
├── generate_data.py   # Generate synthetic CAPTCHA
├── dataset.py         # PyTorch Dataset & DataLoader
├── model.py           # Neural network architectures
├── train.py           # Training pipeline
├── evaluate.py        # Evaluation script
├── predict.py         # Inference / prediction
├── finetune.py        # Fine-tune dengan data asli
├── requirements.txt   # Dependencies
├── fonts/             # (opsional) Custom fonts
├── data/              # Generated training data
│   ├── train/
│   └── val/
└── models/            # Saved model checkpoints
```

## Tips untuk Akurasi Tinggi

1. **Sesuaikan `config.py`** dengan CAPTCHA target:
   - `TEXT_COLOR` — samakan warna
   - `IMAGE_WIDTH/HEIGHT` — samakan ukuran
   - `WORDS` — pastikan semua kata ada di list

2. **Tambahkan font** yang mirip → taruh di `fonts/`

3. **Fine-tune** dengan 50-100 sampel CAPTCHA asli → akurasi naik signifikan

4. **Classification mode** lebih bagus jika kata-katanya fix dan terbatas


Cara Run

== Predict Single Image ==
python predict.py --image "path/ke/captcha.png" --device cpu

== Predict For CLI ==
python predict.py --folder "path/ke/folder_captcha/" --device cpu

== Dari Python Code (integrasi ke script/testing lain) ==

from predict import CaptchaPredictor

predictor = CaptchaPredictor(mode="classification", device="cpu")

# Single image
text, confidence = predictor.predict("captcha.png")
print(f"{text} ({confidence:.2%})")

# Batch
results = predictor.predict_batch(["cap1.png", "cap2.png"])
for filename, text, conf in results:
    print(f"{filename}: {text} ({conf:.2%})")



# Satu file
python predict.py --image "test_images\captcha.png" --device cpu

# Semua file di folder
python predict.py --folder "test_images" --device cpu

# Solve dari file lokal
python solve_captcha.py --file "test_images/captcha_test.jpg" --device cpu

# Solve dari URL langsung
python solve_captcha.py --url "https://web-brimola.ddb.dev.bri.co.id/assets/admin/img/captcha/1772172097.7713.jpg" --device cpu

# Full auto (refresh + download + predict)

# Sekali solve (perlu cookies aktif)
python solve_captcha.py --cookies "csrf_cookie_name=xxx; ci_session=yyy" --device cpu

# 10x berturut + simpan gambar
python solve_captcha.py --cookies "csrf_cookie_name=xxx; ci_session=yyy" --loop 10 --save --device cpu


# Dari Python code (integrasi ke test script)
from solve_captcha import CaptchaSolver
solver = CaptchaSolver(cookies="csrf_cookie_name=xxx; ci_session=yyy", device="cpu")

# Full auto: refresh → download → predict
text, confidence, url = solver.solve()
print(f"CAPTCHA: {text} ({confidence:.2%})")

# Dari URL langsung
text, confidence = solver.solve_from_url("https://web-brimola.ddb.dev.bri.co.id/assets/admin/img/captcha/xxx.jpg")

# Download saja (seperti sebelumnya)
python download_captcha.py

# Download + langsung solve
python download_captcha.py --solve

# Download + solve 5 captcha berturut-turut
python download_captcha.py --solve --count 5

# Pakai GPU
python download_captcha.py --solve --device cuda