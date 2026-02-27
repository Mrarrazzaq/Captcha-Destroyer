import requests
import re
import json
import os
import argparse

from predict import CaptchaPredictor

# Konfigurasi URL dan Headers
BASE_URL = "https://web-brimola.ddb.dev.bri.co.id"
REFRESH_URL = f"{BASE_URL}/users/refresh_captcha"
CAPTCHA_BASE = "/assets/admin/img/captcha/"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Cookie": (
        "_ga_Z8W4LEC390=GS2.1.s1769509300$o22$g1$t1769510383$j60$l0$h0; "
        "_ga=GA1.1.569183695.1764904812; "
        "csrf_cookie_name=e9547c83e12744f5f2936125dafd98b9; "
        "ci_session=bg9f6gc5k21ig1qj0lc1v3036alkpoq6; "
        "TS018559cd=01a3215cff664fef854f908112c5e1af6dc2bc26e80b2c6be7fd25449324130d61637b7a3f8db14619b761154e19986af5562d632fbea1cfe2a583bc594570a5aa93fc1e71fdfdd42c4465f89782d2be53f8bee2bd"
    )
}

def extract_image_url(data):
    """Extract image URL dari response JSON (bisa berisi HTML <img> tag)."""
    if isinstance(data, dict):
        content = json.dumps(data)
    elif isinstance(data, str):
        content = data
    else:
        content = str(data)

    # Pattern 1: src="..." atau src='...' yang mengandung 'captcha'
    match = re.search(r'src=\\?["\']([^"\'\\]*captcha[^"\'\\]*)\\?["\']', content)
    if match:
        return match.group(1)

    # Pattern 2: path langsung ke folder captcha
    match = re.search(r'(/?assets/admin/img/captcha/[^\s"\'<>]+)', content)
    if match:
        return match.group(1)

    # Pattern 3: nama file captcha (angka.angka.jpg)
    match = re.search(r'(\d+\.\d+\.(?:jpg|png|gif))', content)
    if match:
        return CAPTCHA_BASE + match.group(1)

    return None


def download_captcha(solve=False, device="cpu", count=1):
    """Download captcha dan opsional langsung solve.
    
    Args:
        solve: Jika True, langsung predict setelah download
        device: Device untuk model ("cpu" atau "cuda")
        count: Jumlah captcha yang di-download & solve
    """
    predictor = None
    if solve:
        predictor = CaptchaPredictor(mode="classification", device=device)
        print()

    for i in range(count):
        if count > 1:
            print(f"--- [{i+1}/{count}] ---")
    try:
        # 1. Hit Refresh Captcha
        print("Refreshing captcha...")
        response = requests.get(REFRESH_URL, headers=headers, verify=False)
        response.raise_for_status()
        
        data = response.json()
        print(f"[DEBUG] Response: {json.dumps(data)[:500]}")

        # Extract image URL dari response (bisa berisi HTML img tag)
        img_path = extract_image_url(data)

        if not img_path:
            print("Gagal mendapatkan URL gambar dari response.")
            return

        # Normalize URL
        if img_path.startswith('http'):
            full_image_url = img_path
        elif img_path.startswith('/'):
            full_image_url = f"{BASE_URL}{img_path}"
        else:
            full_image_url = f"{BASE_URL}/{img_path}"

        print(f"Target URL: {full_image_url}")

        # 2. Download Gambar
        img_response = requests.get(full_image_url, headers=headers, verify=False)
        img_response.raise_for_status()

        # Simpan file ke folder test_images
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
        os.makedirs(save_dir, exist_ok=True)
        
        # Gunakan nama file dari URL
        filename = os.path.basename(img_path)
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(img_response.content)
        
        print(f"Berhasil! Gambar disimpan sebagai: {filepath}")

        # 3. Solve captcha jika diminta
        if predictor:
            text, confidence = predictor.predict(filepath)
            print(f"Prediksi: '{text}' (confidence: {confidence:.4f})")

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    # Matikan peringatan insecure request karena dev server biasanya self-signed
    requests.packages.urllib3.disable_warnings()

    parser = argparse.ArgumentParser(description="Download & Solve CAPTCHA")
    parser.add_argument("--solve", action="store_true", help="Langsung predict setelah download")
    parser.add_argument("--count", type=int, default=1, help="Jumlah captcha yang di-download (default: 1)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device untuk model")
    args = parser.parse_args()

    download_captcha(solve=args.solve, device=args.device, count=args.count)
