# Captcha Destroyer — Tutorial Lengkap

Tutorial step-by-step untuk menggunakan **Captcha Destroyer**, tool ML-based untuk membaca teks CAPTCHA secara otomatis.

---

## Daftar Isi

1. [Instalasi & Setup](#1-instalasi--setup)
2. [Generate Training Data](#2-generate-training-data)
3. [Training Model](#3-training-model)
4. [Predict / Membaca CAPTCHA](#4-predict--membaca-captcha)
5. [Download CAPTCHA dari Server](#5-download-captcha-dari-server)
6. [Download + Solve Otomatis](#6-download--solve-otomatis)
7. [Full Auto Solver (solve_captcha.py)](#7-full-auto-solver)
8. [Menambah Data Baru & Fine-tune](#8-menambah-data-baru--fine-tune)
9. [Konfigurasi](#9-konfigurasi)
10. [Struktur Project](#10-struktur-project)
11. [Tips & Troubleshooting](#11-tips--troubleshooting)

---

## 1. Instalasi & Setup

### Buat Virtual Environment (Opsional tapi Recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies yang diperlukan:
- `torch` >= 2.0.0
- `torchvision` >= 0.15.0
- `Pillow` >= 9.0.0
- `numpy` >= 1.21.0
- `requests` >= 2.28.0

### (Opsional) Tambahkan Font

Taruh file `.ttf` / `.otf` di folder `fonts/` agar gambar training lebih mirip CAPTCHA asli.
Jika tidak ada, akan pakai system fonts (Times New Roman, Arial, dll).

---

## 2. Generate Training Data

Generate ribuan gambar CAPTCHA sintetis untuk training:

```bash
python generate_data.py
```

Ini akan membuat:
- `data/train/` — 20,000 gambar + `labels.json`
- `data/val/` — 2,000 gambar + `labels.json`

Format `labels.json`:
```json
{
  "000000.png": "INVOICE",
  "000001.png": "TRANSFER",
  "000002.png": "SIMPANAN"
}
```

> **Note:** Jumlah gambar bisa diubah di `config.py` → `TRAIN_SAMPLES` dan `VAL_SAMPLES`.

---

## 3. Training Model

### Mode Classification (Recommended)

Paling cocok untuk vocabulary terbatas (kata-kata tetap):

```bash
# Training standar
python train.py --mode classification

# Custom epochs
python train.py --mode classification --epochs 30

# Pakai CPU (kalau tidak ada GPU)
python train.py --mode classification --device cpu
```

### Mode OCR (CRNN + CTC)

Untuk kasus kata bisa sangat bervariasi:

```bash
python train.py --mode ocr --device cpu
```

### Evaluate Model

Cek performa model setelah training:

```bash
python evaluate.py --mode classification
python evaluate.py --mode ocr
```

Model tersimpan di folder `models/`:
- `models/best_model_classifier.pth` — model classification
- `models/best_model_ocr.pth` — model OCR

---

## 4. Predict / Membaca CAPTCHA

### Predict Satu Gambar

```bash
python predict.py --image test_images/captcha.png --device cpu
```

Output:
```
Model loaded from models/best_model_classifier.pth
Mode: classification | Device: cpu
Model accuracy: 0.9950

Result: 'BAYAR' (confidence: 0.9987)
```

### Predict Semua Gambar di Folder

```bash
python predict.py --folder test_images --device cpu
```

Output:
```
Predicting 5 images...

Filename                       Prediction           Confidence
------------------------------------------------------------
1772172097.7713.jpg            TABUNGAN             0.9823
1772174157.4975.jpg            BAYAR                0.9945
captcha_001.jpg                INVOICE              0.8712

Total: 3 images processed
```

### Dari Python Code

```python
from predict import CaptchaPredictor

predictor = CaptchaPredictor(mode="classification", device="cpu")

# Single image
text, confidence = predictor.predict("captcha.png")
print(f"Text: {text}, Confidence: {confidence:.4f}")

# Batch
results = predictor.predict_batch(["cap1.png", "cap2.png"])
for filename, text, conf in results:
    print(f"{filename}: {text} ({conf:.2%})")
```

---

## 5. Download CAPTCHA dari Server

Download gambar CAPTCHA dari server (disimpan ke `test_images/`):

```bash
python download_captcha.py
```

> **Penting:** Update cookies di `download_captcha.py` dengan cookies session yang aktif dari browser kamu. Cookies bisa expired, jadi perlu di-update berkala.

### Cara Ambil Cookies

1. Buka website di browser
2. Buka DevTools (F12) → tab **Network**
3. Refresh halaman, cari request ke server
4. Copy nilai `csrf_cookie_name` dan `ci_session` dari header Cookie

---

## 6. Download + Solve Otomatis

Download CAPTCHA lalu langsung predict hasilnya:

```bash
# Download + langsung solve
python download_captcha.py --solve

# Download + solve 5 captcha berturut-turut
python download_captcha.py --solve --count 5

# Pakai GPU
python download_captcha.py --solve --device cuda
```

Output:
```
Refreshing captcha...
Target URL: https://web-brimola.../captcha/1772174157.4975.jpg
Berhasil! Gambar disimpan sebagai: test_images/1772174157.4975.jpg
Prediksi: 'BAYAR' (confidence: 0.9945)
```

---

## 7. Full Auto Solver

`solve_captcha.py` adalah solver paling lengkap — refresh captcha dari server, download, dan predict dalam satu langkah.

### Dari Command Line

```bash
# Sekali solve (perlu cookies aktif)
python solve_captcha.py --cookies "csrf_cookie_name=xxx; ci_session=yyy" --device cpu

# 10x berturut + simpan gambar
python solve_captcha.py --cookies "csrf_cookie_name=xxx; ci_session=yyy" --loop 10 --save --device cpu

# Solve dari file lokal
python solve_captcha.py --file test_images/captcha.jpg --device cpu

# Solve dari URL langsung
python solve_captcha.py --url "https://web-brimola.../captcha/xxx.jpg" --device cpu
```

### Dari Python Code (Integrasi ke Script Lain)

```python
from solve_captcha import CaptchaSolver

solver = CaptchaSolver(
    cookies="csrf_cookie_name=xxx; ci_session=yyy",
    device="cpu"
)

# Full auto: refresh → download → predict
text, confidence, url = solver.solve()
print(f"CAPTCHA: {text} ({confidence:.2%})")

# Dari URL langsung
text, confidence = solver.solve_from_url("https://web-brimola.../captcha/xxx.jpg")

# Dari file lokal
text, confidence = solver.solve_from_file("test_images/captcha.jpg")
```

### Opsi CLI solve_captcha.py

| Argumen       | Default   | Keterangan                                  |
|---------------|-----------|---------------------------------------------|
| `--cookies`   | built-in  | Cookie string dari browser                  |
| `--url`       | -         | URL langsung ke gambar CAPTCHA              |
| `--file`      | -         | Path ke file CAPTCHA lokal                  |
| `--loop`      | 1         | Berapa kali solve                           |
| `--save`      | false     | Simpan gambar ke `test_images/`             |
| `--delay`     | 1.0       | Delay antar request (detik)                 |
| `--device`    | cpu       | Device: `cpu` atau `cuda`                   |
| `--model`     | auto      | Path ke model checkpoint                    |

---

## 8. Menambah Data Baru & Fine-tune

Kalau model kurang akurat pada CAPTCHA asli, tambahkan sample real dan finetune.

### Langkah 1: Tambah Sample

```bash
# Tambah 1 gambar + label
python add_sample.py --image test_images/xxx.png --label TABUNGAN

# Tambah banyak sekaligus (pisah koma)
python add_sample.py --image "img1.png,img2.png" --label "TABUNGAN,BAYAR"

# Lihat semua sample yang sudah ditambahkan
python add_sample.py --list
```

Output:
```
Menambahkan 1 sample ke real_data/...

  + xxx.png -> TABUNGAN

1/1 sample berhasil ditambahkan.

Total: 1 sample di real_data/

Filename                                 Label
------------------------------------------------------------
xxx.png                                  TABUNGAN             [OK]
```

### Langkah 2: Fine-tune Model

```bash
# Tambah + langsung finetune
python add_sample.py --image test_images/xxx.png --label TABUNGAN --finetune

# Atau finetune manual dengan data yang sudah dikumpulkan
python finetune.py --data-dir real_data/ --mode classification --device cpu
```

### Langkah 3 (Alternatif): Fine-tune Manual

Kalau mau kontrol lebih, buat folder `real_data/` sendiri:

```
real_data/
├── labels.json
├── captcha_001.jpg
├── captcha_002.jpg
└── captcha_003.jpg
```

Isi `labels.json`:
```json
{
  "captcha_001.jpg": "TABUNGAN",
  "captcha_002.jpg": "BAYAR",
  "captcha_003.jpg": "INVOICE"
}
```

Lalu jalankan:
```bash
python finetune.py --data-dir real_data/ --mode classification --epochs 30 --lr 0.0001 --device cpu
```

### Opsi Fine-tune

| Argumen       | Default         | Keterangan                        |
|---------------|-----------------|-----------------------------------|
| `--data-dir`  | (required)      | Folder berisi gambar + labels.json|
| `--mode`      | classification  | classification atau ocr           |
| `--epochs`    | 30              | Jumlah epoch                      |
| `--lr`        | 0.0001          | Learning rate (kecil untuk finetune) |
| `--device`    | auto            | cpu atau cuda                     |

Model fine-tuned tersimpan sebagai `models/best_model_finetuned_classification.pth`.

---

## 9. Konfigurasi

Semua setting ada di `config.py`:

### Vocabulary

```python
WORDS = [
    "TABUNGAN", "BAYAR", "INVOICE", "TRANSFER", "TAGIHAN",
    "KREDIT", "DEBIT", "SALDO", "REKENING", "MUTASI",
    "SETORAN", "TUNAI", "GIRO", "DEPOSITO", "BUNGA",
    "ANGSURAN", "CICILAN", "PINJAMAN", "NASABAH", "TRANSAKSI",
    "PEMBAYARAN", "PENARIKAN", "SIMPANAN", "ASURANSI", "PREMI",
    "KLAIM", "DENDA", "BIAYA", "PAJAK", "MATERAI",
    "BRIMO", "PERTAMINA", "PESANAN",
]
```

> Tambah/kurangi kata sesuai CAPTCHA target. Setelah mengubah, **harus re-generate data dan re-train**.

### Image Settings

| Setting            | Default         | Keterangan                        |
|--------------------|-----------------|-----------------------------------|
| `IMAGE_WIDTH`      | 150             | Lebar gambar (px)                 |
| `IMAGE_HEIGHT`     | 40              | Tinggi gambar (px)                |
| `BACKGROUND_COLOR` | (255,255,255)   | Warna background                  |
| `TEXT_COLOR`       | (0,0,0)         | Warna teks                        |
| `BORDER_COLOR`     | (0,0,180)       | Warna border                      |
| `NOISE_LEVEL`      | 0.02            | Level noise (0-1)                 |
| `LINE_COUNT`       | (5, 12)         | Jumlah garis pengganggu           |
| `FONT_SIZE_RANGE`  | (18, 24)        | Range ukuran font                 |

### Training Settings

| Setting          | Default | Keterangan                         |
|------------------|---------|------------------------------------|
| `BATCH_SIZE`     | 64      | Batch size                         |
| `NUM_EPOCHS`     | 50      | Max epochs                         |
| `LEARNING_RATE`  | 0.001   | Learning rate                      |
| `TRAIN_SAMPLES`  | 20000   | Jumlah gambar training             |
| `VAL_SAMPLES`    | 2000    | Jumlah gambar validasi             |
| `DEVICE`         | cuda    | Device default (cuda/cpu)          |

---

## 10. Struktur Project

```
Captcha Destroyer/
├── config.py              # Semua konfigurasi (vocab, image, training)
├── generate_data.py       # Generate gambar CAPTCHA sintetis
├── dataset.py             # PyTorch Dataset & DataLoader
├── model.py               # Arsitektur neural network (CNN + CRNN)
├── train.py               # Training pipeline
├── evaluate.py            # Evaluasi model
├── predict.py             # Inference / prediction
├── finetune.py            # Fine-tune dengan data asli
├── add_sample.py          # Tambah sample baru + finetune
├── download_captcha.py    # Download CAPTCHA dari server + solve
├── solve_captcha.py       # Full auto solver (refresh → download → predict)
├── requirements.txt       # Dependencies
├── fonts/                 # (opsional) Custom fonts untuk generate data
├── data/                  # Data training & validasi (auto-generated)
│   ├── train/
│   │   ├── labels.json
│   │   └── *.png
│   └── val/
│       ├── labels.json
│       └── *.png
├── real_data/             # Data CAPTCHA asli untuk fine-tune
│   ├── labels.json
│   └── *.jpg / *.png
├── models/                # Model checkpoints
│   ├── best_model_classifier.pth
│   ├── best_model_ocr.pth
│   └── best_model_finetuned_classification.pth
└── test_images/           # Gambar CAPTCHA untuk testing / hasil download
```

---

## 11. Tips & Troubleshooting

### Tips untuk Akurasi Tinggi

1. **Sesuaikan `config.py`** — warna teks, ukuran gambar, garis pengganggu harus semirip mungkin dengan CAPTCHA target.

2. **Tambahkan font yang mirip** — taruh di `fonts/`. Font serif (Times New Roman) biasanya paling cocok.

3. **Fine-tune dengan data asli** — 50-100 sample CAPTCHA asli sudah cukup untuk menaikkan akurasi signifikan.

4. **Gunakan mode Classification** kalau kata-katanya tetap dan terbatas (< 50 kata).

### Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `CUDA not available` | Tambahkan `--device cpu` |
| `Model not found` | Jalankan `train.py` dulu |
| Cookies expired | Update cookies di `download_captcha.py` / pakai `--cookies` |
| `400 Bad Request` saat download | Cek URL gambar, mungkin ada karakter aneh |
| Akurasi rendah pada CAPTCHA asli | Fine-tune dengan `add_sample.py --finetune` |
| Kata baru tidak dikenali | Tambahkan ke `WORDS` di `config.py`, lalu re-generate + re-train |

### Alur Kerja yang Disarankan

```
1. Setup          → pip install -r requirements.txt
2. Generate       → python generate_data.py
3. Train          → python train.py --mode classification
4. Test           → python predict.py --folder test_images --device cpu
5. Download real  → python download_captcha.py --solve --count 10
6. Labeli manual  → python add_sample.py --image xxx.jpg --label TABUNGAN
7. Fine-tune      → python add_sample.py --list --finetune
8. Deploy         → Integrasikan CaptchaSolver ke script kamu
```
