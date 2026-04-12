from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import LanguageRequest
from algo_classes import (
    ComplementNB, PassiveAggressive, RidgeClassifier,
    SGDClassifier, LangdetectStyleComplementNB,
    FastText, GlotLID, CLD3, CharCNN
)
import time

import warnings
warnings.filterwarnings("ignore")  # suppress sklearn version mismatch noise

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
