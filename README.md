# WiLI-2018 Language Identification — Benchmark & Model Export

**Course Project | Pattern Recognition & Machine Learning (PRML)**
**Group 10**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Environment & Hardware](#environment--hardware)
5. [Methodology](#methodology)
6. [Models](#models)
7. [Results & Comparison](#results--comparison)
8. [Analysis & Visualizations](#analysis--visualizations)
9. [Saved Model Inventory](#saved-model-inventory)
10. [How to Run](#how-to-run)
11. [Dependencies](#dependencies)
12. [Key Takeaways](#key-takeaways)

---

## Project Overview

This project benchmarks a comprehensive suite of **language identification (LangID)** models on the **WiLI-2018** dataset, covering **235 languages**. The goal is to evaluate and compare both classical machine learning approaches and modern neural architectures across accuracy, macro F1-score, training time, and inference speed — producing trained, exportable models for downstream use in an inference backend.

**Models evaluated:**

| Category | Models |
|---|---|
| Classical ML | Complement NB, SGD Classifier, Passive Aggressive, Ridge Classifier, Linear SVC |
| LangDetect-style | Complement NB on char 1–3 gram TF-IDF |
| Neural (Shallow) | fastText, GlotLID, CLD3 (PyTorch re-implementations) |
| Neural (Deep) | High-Capacity Character-level CNN |

---

## Dataset

### WiLI-2018 — Wikipedia Language Identification Benchmark

| Property | Value |
|---|---|
| **Full Name** | WiLI-2018 (Wikipedia Language Identification) |
| **Source** | Wikipedia paragraphs |
| **Languages** | **235** |
| **Train Samples** | **117,500** |
| **Test Samples** | **117,500** |
| **Total Samples** | 235,000 |
| **Samples per Language** | 500 train + 500 test (balanced) |
| **Task** | Multi-class text classification (235 classes) |
| **Kaggle Dataset ID** | `mexwell/wili-2018` |

### Files

```
wili-2018/
├── x_train.txt   — Training text samples (one per line)
├── y_train.txt   — Training labels (ISO language codes)
├── x_test.txt    — Test text samples
└── y_test.txt    — Test labels
```

### Data Characteristics

- Text spans a wide variety of scripts: Latin, Cyrillic, Arabic, Devanagari, CJK, and many more.
- The dataset is **perfectly balanced** — each of the 235 languages contributes exactly 500 training and 500 test paragraphs.
- Empty samples are filtered during preprocessing.
- Labels are encoded with `sklearn.LabelEncoder` (fit on train + test labels combined to ensure full class coverage).

---

## Project Structure

```
prml-project-group-10/
├── prml-project-group-10.ipynb     # Main notebook
├── README.md
└── backend/
    └── weights/                    # All exported model files
        ├── label_encoder.pkl
        ├── vectorizer_char_wb_2_4.pkl
        ├── vectorizer_char_wb_1_3_langdetect.pkl
        ├── clf_ComplementNB.pkl
        ├── clf_SGDClassifier.pkl
        ├── clf_PassiveAggressive.pkl
        ├── clf_RidgeClassifier.pkl
        ├── clf_LinearSVC.pkl
        ├── langdetect_style_complement_nb.pkl
        ├── fasttext_weights.pth
        ├── glotlid_weights.pth
        ├── cld3_weights.pth
        └── charcnn_highcap_weights.pth
```

---

## Environment & Hardware

| Property | Value |
|---|---|
| **Platform** | Kaggle Notebooks |
| **GPU** | 2 × NVIDIA Tesla T4 (15.6 GB VRAM each) |
| **Accelerator** | CUDA (multi-GPU via `nn.DataParallel`) |
| **Python** | 3.12.12 |
| **Deep Learning Framework** | PyTorch |

Neural models leverage **mixed precision training (AMP)**, **gradient accumulation**, and **multi-GPU DataParallel** to maximize utilization of the dual T4 setup.

---

## Methodology

### Preprocessing

1. **Text cleaning:** Empty lines are removed from both train and test sets.
2. **Label encoding:** `sklearn.LabelEncoder` fit on the combined label set (235 classes).
3. **TF-IDF Vectorization (for classical models):**
   - Analyzer: `char_wb` (character n-grams with word boundaries)
   - N-gram range: (2, 4) for classical classifiers; (1, 3) for the LangDetect-style model
   - Max features: 50,000
   - Sublinear TF scaling: enabled

### Neural Model Input

All neural models use a **character n-gram hashing** scheme:
- Text is padded with spaces and sliced into n-grams of specified range.
- Each n-gram is hashed into a fixed-size bucket (avoiding vocabulary maintenance).
- Bucket sizes vary per model (1M–2M).

### Training Setup (Neural Models)

| Setting | fastText / GlotLID / CLD3 | CharCNN |
|---|---|---|
| **Optimizer** | SGD | AdamW |
| **LR Schedule** | StepLR | Cosine Annealing |
| **Loss** | Cross Entropy | Cross Entropy + Label Smoothing |
| **Mixed Precision** | ✅ AMP | ✅ AMP |
| **Gradient Accumulation** | 2 steps | 4 steps |
| **Epochs** | 5 | 20 |
| **Batch Size** | 256 | 128 |
| **Multi-GPU** | DataParallel | DataParallel |

---

## Models

### 1. Classical ML Models (TF-IDF + Classifiers)

All classical models share the same TF-IDF feature matrix (`char_wb`, 2–4 gram, 50k features, shape: `117500 × 50000`).

| Model | Description |
|---|---|
| **Complement NB** | Naive Bayes variant designed for imbalanced multi-class text |
| **SGD Classifier** | Linear model trained with stochastic gradient descent (modified Huber loss, balanced class weights) |
| **Passive Aggressive** | Online learning classifier, aggressive updates on misclassified samples |
| **Ridge Classifier** | Multi-class via one-vs-rest with L2 regularization |
| **Linear SVC** | Support Vector Machine with a linear kernel, fast and effective for sparse TF-IDF |

### 2. LangDetect-Style Naïve Bayes

A reimplementation of the classic [langdetect](https://github.com/Mimino666/langdetect) library strategy:
- TF-IDF with **char_wb**, n-gram range **(1, 3)**, `min_df=2`, no accent stripping.
- Complement NB with `alpha=0.1`.
- Intentionally lightweight — serves as a strong baseline.

### 3. fastText (PyTorch)

Mirrors the original [fastText](https://fasttext.cc/) architecture:
- `nn.EmbeddingBag` with **mean pooling** over hashed character n-grams (2–4).
- Followed by **Adaptive Softmax** for efficient large-vocabulary classification.
- Bucket size: 2,000,000 | Embedding dim: 64 | Trained for 5 epochs.

### 4. GlotLID (PyTorch)

Inspired by [GlotLID](https://github.com/cisnlp/GlotLID):
- Wider `nn.EmbeddingBag` with a **linear classification head**.
- Uses a broader n-gram range **(2–5)** for richer subword features.
- Bucket size: 2,000,000 | Embedding dim: 128 | Trained for 5 epochs.

### 5. CLD3 (PyTorch)

Inspired by Google's [Compact Language Detector v3](https://github.com/google/cld3):
- **Three separate** `nn.EmbeddingBag` modules for uni-grams, bi-grams, and tri-grams.
- Concatenated into a **hidden dense layer** (256 units) → output logits.
- Bucket size: 1,000,000 | Embedding dim: 64 | Hidden dim: 256 | Trained for 5 epochs.

### 6. Character-Level CNN (High Capacity)

A custom multi-kernel convolutional architecture operating on raw **Unicode character IDs**:
- `nn.Embedding` over character vocab of size 65,536 (full BMP Unicode).
- **Parallel convolutional banks** with filter sizes `[2, 3, 4, 5, 6, 7]`.
- Max-over-time pooling → concatenated → classification head.
- Embedding dim: 128 | Filters per size: 256 | Max sequence length: 768 | Trained for **20 epochs**.
- Heaviest model in the suite — captures fine-grained orthographic and script-level patterns.

---

## Results & Comparison

### Full Leaderboard (sorted by Macro F1)

| Rank | Model | Accuracy | Macro F1 | Train Time (s) | Infer Time (s) | Type |
|---|---|---|---|---|---|---|
| 🥇 1 | **GlotLID** | **0.9697** | **0.9706** | 833.8 | 146.43 | Neural |
| 🥈 2 | **LinearSVC** | 0.9637 | 0.9643 | 1055.1 | **9.37** | Classical ML |
| 🥉 3 | **Passive Aggressive** | 0.9631 | 0.9634 | **108.1** | 8.56 | Classical ML |
| 4 | CharCNN | 0.9599 | 0.9615 | 1269.1 | 34.43 | Deep CNN |
| 5 | Ridge Classifier | 0.9576 | 0.9592 | 3367.8 | 9.39 | Classical ML |
| 6 | SGD Classifier | 0.9549 | 0.9567 | 78.9 | 8.49 | Classical ML |
| 7 | CLD3 | 0.9545 | 0.9549 | 553.7 | 98.64 | Neural |
| 8 | fastText | 0.9522 | 0.9526 | 579.9 | 105.90 | Neural |
| 9 | LangDetect-NB | 0.9105 | 0.9126 | 8.5 | 7.07 | Classical ML |
| 10 | Complement NB | 0.9186 | 0.9208 | 10.2 | 8.45 | Classical ML |

### Classical ML Summary

| Model | Accuracy | Macro F1 | Train Time | Infer Time |
|---|---|---|---|---|
| Linear SVC | 0.9637 | 0.9643 | 1055.1s | 9.37s |
| Passive Aggressive | 0.9631 | 0.9634 | 108.1s | 8.56s |
| Ridge Classifier | 0.9576 | 0.9592 | 3367.8s | 9.39s |
| SGD Classifier | 0.9549 | 0.9567 | 78.9s | 8.49s |
| Complement NB | 0.9186 | 0.9208 | 10.2s | 8.45s |
| LangDetect-NB | 0.9105 | 0.9126 | 8.5s | 7.07s |

### Neural Model Summary

| Model | Accuracy | Macro F1 | Train Time | Infer Time |
|---|---|---|---|---|
| GlotLID | **0.9697** | **0.9706** | 833.8s | 146.43s |
| CharCNN | 0.9599 | 0.9615 | 1269.1s | 34.43s |
| CLD3 | 0.9545 | 0.9549 | 553.7s | 98.64s |
| fastText | 0.9522 | 0.9526 | 579.9s | 105.90s |

---

## Analysis & Visualizations

The notebook produces 6 diagnostic plots:

### Plot 1 — Accuracy vs. Macro F1 (Grouped Bar Chart)
Grouped bar chart comparing both metrics side by side for all 10 models, ordered by Macro F1. Confirms that accuracy and F1 track closely on this balanced dataset.

### Plot 2 — Training Time vs. Macro F1 (Scatter Plot)
Color-coded scatter (Classical ML = blue, Neural = green, Deep CNN = red). Reveals the efficiency frontier: **Passive Aggressive** achieves near-top F1 in only ~108s, while **Ridge Classifier** takes 56× longer for marginally better performance.

### Plot 3 — Inference Time Comparison (Horizontal Bar)
Highlights that classical TF-IDF models are far faster at inference (7–9s for 117k samples) vs. neural models (35–146s), due to sparse matrix operations vs. GPU tensor batching overhead.

### Plot 4 — Per-Class F1 Distribution (Best Model — GlotLID)
Box/histogram showing the spread of per-class F1 scores across all 235 languages for GlotLID. Most classes score above 0.95; a tail of low-resource or closely related languages drags the minimum.

### Plot 5 — Confusion Matrix (Top 15 Languages)
Heatmap restricted to the 15 most frequent language groups in GlotLID's predictions. Reveals which languages are most commonly confused (typically script-sharing or geographically proximate languages).

### Plot 6 — Macro F1 Heatmap (All Models)
Color-coded summary table (YlGn colormap) of Accuracy and Macro F1 for all models at a glance.

---

## Saved Model Inventory

All models are exported to `./backend/weights/` for use in an inference API.

| File | Size | Description |
|---|---|---|
| `label_encoder.pkl` | < 1 MB | Fitted `LabelEncoder` (235 classes) |
| `vectorizer_char_wb_2_4.pkl` | ~MB | TF-IDF (2–4 gram) for classical models |
| `vectorizer_char_wb_1_3_langdetect.pkl` | ~MB | TF-IDF (1–3 gram) for LangDetect-NB |
| `clf_ComplementNB.pkl` | 188.4 MB | Complement NB classifier |
| `clf_LinearSVC.pkl` | 94.0 MB | Linear SVC classifier |
| `clf_PassiveAggressive.pkl` | 94.0 MB | Passive Aggressive classifier |
| `clf_RidgeClassifier.pkl` | ~MB | Ridge Classifier |
| `clf_SGDClassifier.pkl` | ~MB | SGD Classifier |
| `langdetect_style_complement_nb.pkl` | ~MB | LangDetect-style NB |
| `fasttext_weights.pth` | ~MB | fastText PyTorch weights |
| `glotlid_weights.pth` | ~MB | GlotLID PyTorch weights |
| `cld3_weights.pth` | 768.4 MB | CLD3 PyTorch weights |
| `charcnn_highcap_weights.pth` | 38.6 MB | CharCNN PyTorch weights |

---

## Getting the Weights

Model weights are **not stored in this repository** (total ~2.9 GB). Place all files into `backend/weights/` using one of the two methods below.

---

### Option 1 — Google Drive (Manual, Recommended for Quick Setup)

**Download the entire folder in one click:**

> 🔗 **[Google Drive — DeText Model Weights](https://drive.google.com/drive/folders/1w3rNWTsXgIP3jEsSvMKHlUBQslWpZJ0m?usp=sharing)**

1. Open the link above.
2. Click **"Download all"** (or select individual files).
3. Unzip / move all files into `backend/weights/`.

Your directory should look like:

```
backend/weights/
├── label_encoder.pkl
├── vectorizer_char_wb_2_4.pkl
├── vectorizer_char_wb_1_3_langdetect.pkl
├── clf_ComplementNB.pkl
├── clf_LinearSVC.pkl
├── clf_PassiveAggressive.pkl
├── clf_RidgeClassifier.pkl
├── clf_SGDClassifier.pkl
├── langdetect_style_complement_nb.pkl
├── fasttext_weights.pth
├── glotlid_weights.pth
├── cld3_weights.pth
└── charcnn_highcap_weights.pth
```

---

### Option 2 — Automated Script (Hugging Face Hub)

The included `download_weights.py` script fetches all weights directly from:
> 🤗 **[pyconfaced/ClassicalNLP-LanguageDetectionModels](https://huggingface.co/pyconfaced/ClassicalNLP-LanguageDetectionModels)**

```bash
# From the repo root — downloads only missing files
python3 download_weights.py

# Force re-download everything (overwrites existing files)
python3 download_weights.py --force
```

The backend will also **auto-download any missing weights on first startup** — no manual step needed if you run `uvicorn` right after cloning.

---

## How to Run

### On Kaggle (Training / Notebook)

1. Open the notebook on Kaggle.
2. Add the dataset `mexwell/wili-2018` as a data source.
3. Enable GPU accelerator (Tesla T4 × 2).
4. Run all cells sequentially. Total runtime: ~3–4 hours (dominated by Ridge and CharCNN).

### Running Locally (Backend + Frontend)

We have provided a unified startup script that automatically:
- Checks for Python & Node.js
- Installs `frontend/node_modules/` if missing
- Automatically downloads any missing model weights (~2.9 GB)
- Starts both the FastAPI backend and React frontend concurrently

```bash
# 1. Clone the repo
git clone https://github.com/Piyush2005-code/Language_Detection_PRML_Group_10
cd Language_Detection_PRML_Group_10

# 2. Run the startup script
./start.sh
```

- **Frontend UI:** `http://localhost:5173`
- **Backend API:** `http://localhost:8000`

> Press `Ctrl+C` in the terminal to cleanly shut down both services.


> **Note:** The Ridge Classifier takes ~56 minutes and CharCNN takes ~21 minutes on dual T4 GPUs. On CPU, these will be significantly longer.

---

## Dependencies

```
torch
scikit-learn
matplotlib
seaborn
pandas
numpy
tqdm
joblib
```

Install all at once:

```bash
pip install torch scikit-learn matplotlib seaborn pandas numpy tqdm joblib
```

---

## Key Takeaways

1. **GlotLID achieves the highest Macro F1 (0.9706)** — the wider n-gram range (2–5) and larger embedding dimension (128) give it an edge over fastText and CLD3 despite the same 5-epoch training budget.

2. **Classical ML is surprisingly competitive.** Linear SVC and Passive Aggressive both exceed 0.963 F1 with sub-10-second inference, making them strong production candidates where latency matters.

3. **Passive Aggressive is the best efficiency trade-off** — 0.963 F1 in just 108 seconds of training, vs. 1055s for LinearSVC and 3368s for Ridge at only marginally better scores.

4. **CharCNN underperforms neural bag-of-ngrams models** despite 4× more training epochs and the heaviest architecture. Character-level convolutions need more capacity or longer sequences to surpass hashing-based embeddings on a 235-class task.

5. **CLD3's multi-scale embedding design does not outperform fastText** at 5 epochs — the tri-gram separation adds complexity without a meaningful accuracy gain at this scale.

6. **Ridge Classifier is the worst value proposition** — takes 56× longer than Passive Aggressive for a 0.3% F1 improvement.

7. **Inference speed favors classical models** — TF-IDF sparse classifiers serve 117k predictions in ~9s vs. 35–146s for neural models, even with GPU batching.

---

*PRML Group 10 | WiLI-2018 Language Identification Benchmark*
