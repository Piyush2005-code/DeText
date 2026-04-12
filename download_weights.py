"""
download_weights.py — Fetch all model weights from Hugging Face Hub.

Usage:
    python3 download_weights.py

The script downloads all weights into backend/weights/ relative to this file.
Skips files that already exist unless --force is passed.
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

HF_REPO = "pyconfaced/ClassicalNLP-LanguageDetectionModels"
BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

WEIGHTS = [
    "label_encoder.pkl",
    "vectorizer_char_wb_2_4.pkl",
    "vectorizer_char_wb_1_3_langdetect.pkl",
    "clf_ComplementNB.pkl",
    "clf_LinearSVC.pkl",
    "clf_PassiveAggressive.pkl",
    "clf_RidgeClassifier.pkl",
    "clf_SGDClassifier.pkl",
    "langdetect_style_complement_nb.pkl",
    "fasttext_weights.pth",
    "glotlid_weights.pth",
    "cld3_weights.pth",
    "charcnn_highcap_weights.pth",
]

WEIGHTS_DIR = Path(__file__).parent / "backend" / "weights"


def _progress(filename: str):
    """Return a reporthook that prints a simple progress bar."""
    def hook(count, block_size, total_size):
        if total_size <= 0:
            return
        done = count * block_size
        pct = min(done * 100 // total_size, 100)
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        mb_done = done / 1_048_576
        mb_total = total_size / 1_048_576
        print(
            f"\r  [{bar}] {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MB",
            end="",
            flush=True,
        )
    return hook


def download(force: bool = False) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Destination : {WEIGHTS_DIR}")
    print(f"Source      : {BASE_URL}\n")

    failed = []
    for fname in WEIGHTS:
        dest = WEIGHTS_DIR / fname
        if dest.exists() and not force:
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  SKIP  {fname}  ({size_mb:.1f} MB already present)")
            continue

        url = f"{BASE_URL}/{fname}"
        print(f"  GET   {fname}")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress(fname))
            print()  # newline after progress bar
            size_mb = dest.stat().st_size / 1_048_576
            print(f"        saved  {size_mb:.1f} MB → {dest}")
        except Exception as exc:
            print(f"\n        FAILED: {exc}")
            failed.append(fname)

    print()
    if failed:
        print(f"Some files failed to download: {failed}")
        sys.exit(1)
    else:
        print("All weights downloaded successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DeText model weights from Hugging Face.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files that already exist locally."
    )
    args = parser.parse_args()
    download(force=args.force)
