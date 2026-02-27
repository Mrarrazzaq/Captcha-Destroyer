"""
Configuration for Captcha Destroyer
All settings in one place for easy tuning.
"""

import os

# ============================================================
# Vocabulary — daftar kata yang muncul di CAPTCHA
# Tambahkan / kurangi sesuai kebutuhan
# ============================================================
# PENTING: CAPTCHA target menggunakan HURUF KAPITAL
WORDS = [
    "TABUNGAN",
    "BAYAR",
    "INVOICE",
    "TRANSFER",
    "TAGIHAN",
    "KREDIT",
    "DEBIT",
    "SALDO",
    "REKENING",
    "MUTASI",
    "SETORAN",
    "TUNAI",
    "GIRO",
    "DEPOSITO",
    "BUNGA",
    "ANGSURAN",
    "CICILAN",
    "PINJAMAN",
    "NASABAH",
    "TRANSAKSI",
    "PEMBAYARAN",
    "PENARIKAN",
    "SIMPANAN",
    "ASURANSI",
    "PREMI",
    "KLAIM",
    "DENDA",
    "BIAYA",
    "PAJAK",
    "MATERAI",
    "BRIMO",
    "PERTAMINA",
    "PESANAN",
    "BRIVA",
    "OUTLET",
    "BRITAMA",
    "NONPSO",
    "PRODUK",
    "HARIAN",
    "SUBSIDI",
    "GUDANG",     
    "KIRIM",      
    "ECERAN",     
    "AGEN",       
    "TABUNG",     
    "HARGA",      
    "HISWANA",    
    "KUOTA",
    "LAPORAN",    
    "SIMELON",   
    "APLIKASI",   
    "ADMIN"       
]

# ============================================================
# Image settings — sesuaikan dengan CAPTCHA target
# ============================================================
IMAGE_WIDTH = 150                   # match target CAPTCHA width
IMAGE_HEIGHT = 40                   # match target CAPTCHA height
BACKGROUND_COLOR = (255, 255, 255)  # putih
TEXT_COLOR = (0, 0, 0)              # hitam
BORDER_COLOR = (0, 0, 180)          # border biru seperti di target
ADD_BORDER = True                   # border kotak di sekeliling
NOISE_LEVEL = 0.02                  # sedikit noise
ADD_LINES = True                    # garis-garis diagonal pengganggu (banyak!)
LINE_COUNT = (5, 12)                # banyak garis seperti di target
LINE_COLOR_RANGE = (100, 200)       # warna garis abu-abu/biru
ADD_DOTS = False                    # target tidak ada dots
DOT_COUNT = (0, 0)
FONT_SIZE_RANGE = (18, 24)          # font size match target
DISTORTION_LEVEL = 1                # sedikit distorsi

# ============================================================
# Character set — untuk mode OCR (baca per karakter)
# Dibangun otomatis dari WORDS
# ============================================================
CHARSET = sorted(set("".join(WORDS)))  # unique characters
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARSET)}  # 0 = CTC blank
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
IDX_TO_CHAR[0] = ""  # blank
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank

# Word-level classification
WORD_TO_IDX = {w: i for i, w in enumerate(WORDS)}
IDX_TO_WORD = {i: w for i, w in enumerate(WORDS)}
NUM_WORDS = len(WORDS)

# ============================================================
# Training settings
# ============================================================
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
TRAIN_SAMPLES = 20000      # jumlah gambar training yang di-generate
VAL_SAMPLES = 2000          # jumlah gambar validasi
NUM_WORKERS = 4
DEVICE = "cuda"             # "cuda" atau "cpu"

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
FONT_DIR = os.path.join(BASE_DIR, "fonts")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# ============================================================
# Max label length (for CTC)
# ============================================================
MAX_LABEL_LENGTH = max(len(w) for w in WORDS)
