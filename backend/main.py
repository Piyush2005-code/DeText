import os
import sys
import time
import warnings
import urllib.request
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import LanguageRequest

warnings.filterwarnings("ignore")  # suppress sklearn version mismatch noise

# ---------------------------------------------------------------------------
# Auto-download weights if any are missing
# HuggingFace: pyconfaced/ClassicalNLP-LanguageDetectionModels
# Google Drive (manual): https://drive.google.com/drive/folders/1w3rNWTsXgIP3jEsSvMKHlUBQslWpZJ0m
# ---------------------------------------------------------------------------
_HF_BASE = "https://huggingface.co/pyconfaced/ClassicalNLP-LanguageDetectionModels/resolve/main"
_WEIGHTS_DIR = Path(__file__).parent / "weights"
_REQUIRED_WEIGHTS = [
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

def _download_weights():
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in _REQUIRED_WEIGHTS if not (_WEIGHTS_DIR / f).exists()]
    if not missing:
        return
    print(f"[DeText] {len(missing)} weight file(s) missing — downloading from Hugging Face...")
    for fname in missing:
        url = f"{_HF_BASE}/{fname}"
        dest = _WEIGHTS_DIR / fname
        print(f"[DeText]   Downloading {fname} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            size_mb = dest.stat().st_size / 1_048_576
            print(f"done ({size_mb:.1f} MB)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            print("[DeText] Weight download incomplete. Run download_weights.py manually.")

_download_weights()

from algo_classes import (
    ComplementNB, PassiveAggressive, RidgeClassifier,
    SGDClassifier, LangdetectStyleComplementNB,
    FastText, GlotLID, CLD3, CharCNN
)

app = FastAPI()

# Load models — each wrapped so a corrupt weight file doesn't crash startup
def _load(cls):
    try:
        return cls()
    except Exception as e:
        import logging
        logging.getLogger("uvicorn").warning(f"Could not load {cls.__name__}: {e}")
        return None

clf_complementNB       = _load(ComplementNB)
clf_passive_aggressive = _load(PassiveAggressive)
clf_ridge_classifier   = _load(RidgeClassifier)
clf_sgd_classifier     = _load(SGDClassifier)
clf_langdetect         = _load(LangdetectStyleComplementNB)
clf_fasttext           = _load(FastText)
clf_glotlid            = _load(GlotLID)
clf_cld3               = _load(CLD3)
clf_charcnn            = _load(CharCNN)
# clf_linear_svc omitted — corrupt (EOF); clf_nearest_centroid omitted — weight file missing

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def getMessage():
    return {"status_code": 200, "status": "Up and Running"}


ALGO_MAP = {
    "Complement Naive Bayes": clf_complementNB,
    "Passive Aggressive":     clf_passive_aggressive,
    "Ridge Classifier":       clf_ridge_classifier,
    "SGD Classifier":         clf_sgd_classifier,
    "Lang Detect":            clf_langdetect,
    "FastText":               clf_fasttext,
    "GlotLID":                clf_glotlid,
    "CLD3":                   clf_cld3,
    "CharCNN (High-Cap)":     clf_charcnn,
    # "Linear SVC" omitted — corrupt; "Nearest Centroid" omitted — weight file missing
}


@app.post("/")
def language(request: LanguageRequest):
    model = ALGO_MAP.get(request.detection_algo)
    if model is None:
        return {"status_code": 400, "error": f"Unknown algorithm: {request.detection_algo}"}

    start = time.perf_counter()
    prediction = model(request.prompt)
    end = time.perf_counter()

    return {
        "status_code": 200,
        "execution_time": end - start,
        "detected language": prediction,
    }
