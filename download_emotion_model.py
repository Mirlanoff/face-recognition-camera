"""
Downloads the FER+ emotion recognition ONNX model from the official ONNX model zoo.
Run once: python download_emotion_model.py
"""

import os
import urllib.request
from pathlib import Path

MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
MODEL_PATH = Path("models") / "emotion-ferplus-8.onnx"


def download():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print("[OK] Model already exists: " + str(MODEL_PATH))
        return
    print("[+] Downloading FER+ emotion model (~35 MB)...")
    print("    From: " + MODEL_URL)
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    print("[OK] Saved to: " + str(MODEL_PATH))


if __name__ == "__main__":
    download()
