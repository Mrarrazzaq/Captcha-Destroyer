# Add & Train New Wordlist

## 1. Edit Vocabulary di `config.py`
Buka file `config.py` dan ubah list `WORDS` dengan kata-kata target:

```python
WORDS = [
    "KATA1",
    "KATA2",
    "KATA3",
    # tambah sesuai kebutuhan
]
```

## 2. Generate Training Data
```bash
python generate_data.py
```
Menghasilkan 20,000 training images + 2,000 validasi di folder `data/`

## 3. Train Model
```bash
python train.py --mode classification
```
Model akan disimpan di `models/best_model.pth`

## 4. Run Solve Captcha
```bash
python solve_captcha.py

# Single File
python solve_captcha.py --file "test_images/captcha_test.jpg" --device cpu

# Folder
python solve_captcha.py --folder "test_images/captcha_test.jpg" --device cpu

```

---

**Opsional - Fine-tune dengan Data Real:**
```bash
python finetune.py --data-dir real_data/ --mode classification
```
