"""
Captcha Solver — End-to-end: fetch CAPTCHA from server → predict text.

Usage:
  python solve_captcha.py                          # sekali solve
  python solve_captcha.py --loop 10                # solve 10x berturut-turut
  python solve_captcha.py --loop 10 --save         # solve 10x + simpan gambar

Dari Python:
  from solve_captcha import CaptchaSolver
  solver = CaptchaSolver(cookies="...", device="cpu")
  text, confidence, image_url = solver.solve()
"""

import os
import re
import json
import time
import argparse
import requests
from io import BytesIO
from PIL import Image

from predict import CaptchaPredictor

# ============================================================
# Default session config — UPDATE cookies sesuai session kamu
# ============================================================
BASE_URL = "https://web-brimola.ddb.dev.bri.co.id"
REFRESH_ENDPOINT = "/users/refresh_captcha"
CAPTCHA_BASE = "/assets/admin/img/captcha/"

DEFAULT_HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "id,en-US;q=0.9,en;q=0.8,fi;q=0.7",
    "Connection": "keep-alive",
    "DNT": "1",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

IMAGE_HEADERS = {
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "id,en-US;q=0.9,en;q=0.8,fi;q=0.7",
    "Connection": "keep-alive",
    "DNT": "1",
    "Sec-Fetch-Dest": "image",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

# Default cookies — GANTI dengan cookies session kamu yang aktif
DEFAULT_COOKIES = "csrf_cookie_name=7318c7db3eeab9a548c13b2281b797d7; ci_session=0rne7kcduqjfmb26acl9v33aoqmgqtrp"


def parse_cookies(cookie_string):
    """Parse cookie string into dict."""
    cookies = {}
    for item in cookie_string.split(";"):
        item = item.strip()
        if "=" in item:
            key, val = item.split("=", 1)
            cookies[key.strip()] = val.strip()
    return cookies


class CaptchaSolver:
    """
    End-to-end CAPTCHA solver.

    1. Calls refresh_captcha to get new CAPTCHA
    2. Downloads the CAPTCHA image
    3. Predicts the text using the trained ML model
    """

    def __init__(self, cookies=None, device="cpu", model_path=None, mode="classification"):
        # Setup session
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

        cookie_str = cookies or DEFAULT_COOKIES
        self.session.cookies.update(parse_cookies(cookie_str))

        # Setup predictor
        self.predictor = CaptchaPredictor(mode=mode, model_path=model_path, device=device)

    def refresh_captcha(self):
        """
        Call refresh_captcha endpoint to get a new CAPTCHA.
        Returns the CAPTCHA image URL.
        """
        url = BASE_URL + REFRESH_ENDPOINT
        resp = self.session.get(url, headers=DEFAULT_HEADERS)
        resp.raise_for_status()

        data = resp.json()

        # Response biasanya berisi HTML img tag atau URL langsung
        # Coba parse dari berbagai format response
        if isinstance(data, dict):
            # Coba ambil URL dari response JSON
            # Format kemungkinan: {"captcha_image": "<img src='...'>"}
            # atau {"url": "..."}
            content = json.dumps(data)
        elif isinstance(data, str):
            content = data
        else:
            content = str(data)

        # Extract image URL dari response (bisa dalam img tag atau langsung URL)
        # Pattern 1: src="..."
        match = re.search(r'src=["\']([^"\']*captcha[^"\']*)["\']', content)
        if match:
            img_path = match.group(1)
        else:
            # Pattern 2: URL langsung ke jpg/png
            match = re.search(r'(/?assets/admin/img/captcha/[^\s"\'<>]+)', content)
            if match:
                img_path = match.group(1)
            else:
                # Pattern 3: angka.angka.jpg
                match = re.search(r'(\d+\.\d+\.(?:jpg|png|gif))', content)
                if match:
                    img_path = CAPTCHA_BASE + match.group(1)
                else:
                    print(f"[DEBUG] Response: {content[:500]}")
                    raise ValueError("Tidak bisa menemukan URL CAPTCHA dari response. Cek format response di atas.")

        # Normalize URL
        if img_path.startswith("http"):
            image_url = img_path
        elif img_path.startswith("/"):
            image_url = BASE_URL + img_path
        else:
            image_url = BASE_URL + "/" + img_path

        return image_url, data

    def download_captcha(self, image_url):
        """Download CAPTCHA image and return as PIL Image."""
        resp = self.session.get(image_url, headers=IMAGE_HEADERS)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        return image

    def solve(self, save_path=None):
        """
        Full pipeline: refresh → download → predict.

        Args:
            save_path: Optional path to save the downloaded image.

        Returns:
            (predicted_text, confidence, image_url)
        """
        # Step 1: Refresh CAPTCHA
        image_url, raw_response = self.refresh_captcha()

        # Step 2: Download image
        image = self.download_captcha(image_url)

        # Step 3: Save if requested
        if save_path:
            image.save(save_path)

        # Step 4: Predict
        text, confidence = self.predictor.predict(image)

        return text, confidence, image_url

    def solve_from_url(self, image_url, save_path=None):
        """
        Solve from a known image URL (skip refresh).

        Args:
            image_url: Direct URL to CAPTCHA image.
            save_path: Optional path to save the image.

        Returns:
            (predicted_text, confidence)
        """
        image = self.download_captcha(image_url)
        if save_path:
            image.save(save_path)
        text, confidence = self.predictor.predict(image)
        return text, confidence

    def solve_from_file(self, file_path):
        """Solve from a local file."""
        text, confidence = self.predictor.predict(file_path)
        return text, confidence


def main():
    parser = argparse.ArgumentParser(description="Solve CAPTCHA from BRI Mola")
    parser.add_argument("--cookies", type=str, default=None,
                        help="Cookie string dari browser (copy dari DevTools)")
    parser.add_argument("--url", type=str, default=None,
                        help="Direct URL ke gambar CAPTCHA (skip refresh)")
    parser.add_argument("--file", type=str, default=None,
                        help="Path ke local file CAPTCHA")
    parser.add_argument("--loop", type=int, default=1,
                        help="Berapa kali solve (default: 1)")
    parser.add_argument("--save", action="store_true",
                        help="Simpan gambar CAPTCHA ke test_images/")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay antar request dalam detik (default: 1)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  Captcha Destroyer — Auto Solver")
    print("=" * 60)

    solver = CaptchaSolver(
        cookies=args.cookies,
        device=args.device,
        model_path=args.model,
    )

    if args.save:
        os.makedirs("test_images", exist_ok=True)

    # Mode: solve from local file
    if args.file:
        text, confidence = solver.solve_from_file(args.file)
        print(f"\nFile: {args.file}")
        print(f"Result: '{text}' (confidence: {confidence:.4f})")
        return

    # Mode: solve from direct URL
    if args.url:
        save_path = "test_images/from_url.jpg" if args.save else None
        text, confidence = solver.solve_from_url(args.url, save_path=save_path)
        print(f"\nURL: {args.url}")
        print(f"Result: '{text}' (confidence: {confidence:.4f})")
        if save_path:
            print(f"Saved: {save_path}")
        return

    # Mode: full auto (refresh + download + predict)
    print(f"\nSolving {args.loop} CAPTCHA(s)...\n")
    print(f"{'#':<5} {'Prediction':<15} {'Confidence':<12} {'URL'}")
    print("-" * 80)

    results = []
    for i in range(args.loop):
        try:
            save_path = f"test_images/captcha_{i:03d}.jpg" if args.save else None
            text, confidence, image_url = solver.solve(save_path=save_path)
            results.append((text, confidence, image_url))
            short_url = image_url.split("/")[-1]
            print(f"{i+1:<5} {text:<15} {confidence:<12.4f} {short_url}")
        except Exception as e:
            print(f"{i+1:<5} ERROR: {e}")
            results.append(("ERROR", 0.0, str(e)))

        if i < args.loop - 1:
            time.sleep(args.delay)

    print(f"\n{'='*60}")
    print(f"Total: {args.loop} | Success: {sum(1 for r in results if r[0] != 'ERROR')}")
    if args.save:
        print(f"Images saved to: test_images/")


if __name__ == "__main__":
    main()
