# Panduan Training Captcha Destroyer di Google Colab

## Opsi 1: Training dengan Git Repository (Recommended)

### Langkah 1: Buat Notebook Colab
1. Buka https://colab.research.google.com
2. Pilih **File → New Notebook**
3. Rename ke "Captcha-Destroyer-Training"

### Langkah 2: Copy & Paste Code Berikut

```python
# ============================================
# SETUP GOOGLE COLAB ENVIRONMENT
# ============================================

# Mount Google Drive (untuk simpan results)
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/Mrarrazzaq/Captcha-Destroyer.git /content/captcha-destroyer
%cd /content/captcha-destroyer

# Install dependencies
!pip install -r requirements.txt

print("✓ Setup selesai!")
```

### Langkah 3: Generate Training Data

```python
# Generate synthetic training data (20k training + 2k validation)
!python generate_data.py
```

**⏱️ Waktu:** ~5-10 menit tergantung speed

### Langkah 4: Training Model

**Mode Classification (Recommended):**
```python
!python train.py --mode classification --epochs 50
```

**Mode OCR:**
```python
!python train.py --mode ocr --epochs 50
```

**⏱️ Waktu:** 
- Classification: ~30-60 menit untuk 50 epochs
- OCR: ~1-2 jam untuk 50 epochs

### Langkah 5: Evaluate Model

```python
!python evaluate.py --mode classification
```

### Langkah 6: Download Model ke Google Drive

```python
import shutil
import os

# Zip semua results
shutil.make_archive('/content/drive/MyDrive/captcha-results', 
                    'zip', 
                    '/content/captcha-destroyer/models')

# Copy best model
!cp /content/captcha-destroyer/models/best_model_classifier.pth \
    /content/drive/MyDrive/

print("✓ Model sudah di-download ke Google Drive!")
```

---

## Opsi 2: Training dengan Upload File Local

Jika ingin upload folder `data/` yang sudah ada di local:

### Langkah 1: Compress Data Lokal
```bash
# Di local machine (PowerShell)
Compress-Archive -Path "D:\Work\Private\Self\PVT\Captcha Destroyer\data" `
                -DestinationPath "D:\Work\Private\Self\PVT\Captcha Destroyer\data.zip"
```

### Langkah 2: Di Google Colab

```python
# Upload data.zip lewat file picker
from google.colab import files
uploaded = files.upload()

# Extract
import zipfile
for filename in uploaded.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('/content/data')

# Lanjutkan training
!cd /content && python train.py --mode classification --epochs 50
```

---

## Opsi 3: Training dengan Google Drive (Best untuk Iterasi)

Jika sudah push code ke GitHub:

```python
from google.colab import drive
drive.mount('/content/drive')

# Clone ke Google Drive (persistent storage)
!git clone https://github.com/Mrarrazzaq/Captcha-Destroyer.git \
    /content/drive/MyDrive/captcha-destroyer
    
%cd /content/drive/MyDrive/captcha-destroyer

# Install & Setup
!pip install -r requirements.txt
!python generate_data.py

# Training
!python train.py --mode classification --epochs 50

# Model auto-save di Google Drive
```

---

## Tips & Troubleshooting

### ❌ Error: Torch/CUDA issues
**Solution:** Colab sudah auto-detect GPU. Jika bermasalah:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### ❌ Out of Memory
Reduce batch size di `config.py`:
```python
BATCH_SIZE = 16  # turun dari 32
```

### ✅ Menggunakan Pre-trained Model
```python
# Download model dari Google Drive
!cp /content/drive/MyDrive/best_model_classifier.pth \
    /content/captcha-destroyer/models/

# Evaluate/Predict
!python predict.py --image test.png --mode classification
```

### ✅ Fine-tune dengan Real Data
```python
# Upload real_data/labels.json + images
# Kemudian:
!python finetune.py --data-dir real_data/ --mode classification --epochs 20
```

---

## Struktur File di Colab

```
/content/captcha-destroyer/
├── train.py
├── generate_data.py
├── evaluate.py
├── predict.py
├── data/
│   ├── train/     (auto-generated)
│   └── val/       (auto-generated)
└── models/
    └── best_model_classifier.pth (hasil training)
```

---

## Runtime Estimate

| Task | Time | Cost |
|------|------|------|
| Generate Data (20k) | 5-10 min | FREE |
| Training (50 epochs) | 30-60 min | FREE |
| Fine-tune (20 epochs) | 20-30 min | FREE |
| **Total** | **1-2 jam** | **FREE** |

Colab GPU FREE tier biasanya cukup untuk project ini!

---

## Next Steps

1. **✅ Code sudah di GitHub:**
   ```
   https://github.com/Mrarrazzaq/Captcha-Destroyer.git
   ```

2. **Buka Colab dan ikuti langkah di atas** 👈 Tinggal copy-paste!

3. **Download model hasil training** dari Google Drive

4. **Gunakan model lokal untuk inference**

---

Pertanyaan? Tanya di sini! 🚀
