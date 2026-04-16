# Project Overview

## AI-Based Early Mental Health Breakdown Detection from Speech Patterns

**Course:** BSDA4001 — Data Science and AI Lab Project
**Institution:** Indian Institute of Technology Madras
**Group:** 6

---

## 1. Project Objective

The primary objective is to build an end-to-end machine learning system that ingests speech or physiological audio signals and classifies the speaker's emotional or mental health state. The system targets four clinically and affectively distinct tasks:

1. **Emotion recognition** — 8-class classification (neutral, calm, happy, sad, angry, fearful, disgust, surprised) using the RAVDESS acted-speech corpus.
2. **Depression detection** — Binary classification (depressed vs. healthy) and PHQ-8 severity regression using the DAIC-WOZ clinical interview corpus.
3. **Major Depressive Disorder detection** — Binary classification (MDD vs. healthy control) using the MODMA real-patient audio corpus.
4. **Physiological stress detection** — 3-class and binary classification of work-related stress states using SWELL HRV/EDA sensor signals.

The system is built on two architectures:

- **Pipeline 1 (Baseline):** Domain-engineered acoustic features (338–2,231 dimensions) combined with classical ML classifiers.
- **Pipeline 2 (Proposed):** Frozen OpenAI Whisper encoder producing rich 1536-dimensional speech embeddings fed into XGBoost or SVM classifiers.

---

## 2. End-to-End Architecture

### Pipeline 1 — Handcrafted Features + Classical ML

```
Raw Audio / Physiological Signal (WAV, CSV)
          │
          ▼
  ┌───────────────────────────────────────┐
  │         Feature Extraction            │
  │  MFCC (40 coeff, mean+std+delta=160)  │
  │  Chroma (24), Mel spectrogram (128)   │
  │  Spectral: centroid, bandwidth,       │
  │  rolloff, contrast×7 (14 features)   │
  │  ZCR (2), RMS (2), Duration (1)       │
  │  COVAREP acoustic statistics          │
  │  HRV: RMSSD, SDNN, LF/HF ratio       │
  └───────────────────────────────────────┘
          │
          ▼
  RobustScaler Normalization
  (fitted on train split only)
          │
          ▼
  ┌───────────────────────────────────────┐
  │         Classifier                    │
  │  XGBoost (hist, shallow depth)        │
  │  SVM (RBF, calibrated)                │
  │  MLP (EmotionMLP / StressMLP)         │
  └───────────────────────────────────────┘
          │
          ▼
  Class Label or Regression Score
```

### Pipeline 2 — Whisper Embeddings + Classical Head

```
Raw Audio (WAV)
          │
          ▼
  ┌──────────────────────────────────────────┐
  │   Whisper Encoder (openai/whisper-small)  │
  │   Frozen — 244M parameters not updated   │
  │   30-second window → encoder output      │
  │   Mean + Std pool over time → 1536-D     │
  └──────────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────────┐
  │   Classifier Head                        │
  │   XGBoost  → RAVDESS emotion (8-class)   │
  │   SVM RBF  → MODMA depression (binary)   │
  └──────────────────────────────────────────┘
          │
          ▼
  Class Label + Confidence Score
```

---

## 3. Data Flow (Detailed)

### 3.1 Audio Input Stage

Raw audio files (WAV/MP3/FLAC) are loaded using `librosa` at 16,000 Hz for Whisper encoding. Silent segments are detected and trimmed. Invalid values (NaN, ±Inf) arising from silent or corrupted segments are replaced with zero.

### 3.2 Feature Extraction Stage

**Whisper embeddings (Pipeline 2 / Deployed App):**
- Audio is loaded and padded/trimmed to a 30-second window using `whisper.pad_or_trim`.
- The 80-bin log mel-spectrogram is computed and passed through the frozen Whisper encoder.
- Mean and standard deviation are pooled over the time dimension to produce a 1536-dimensional embedding.
- This single vector represents the full prosodic, phonetic, and paralinguistic content of the audio.

**Handcrafted features (Pipeline 1):**
- Per-file features extracted using librosa across the entire recording.
- For DAIC-WOZ, per-frame CSV statistics are already provided; aggregation (mean, std) produces session-level vectors.

### 3.3 Normalization Stage

All features are normalized with `RobustScaler` (median + IQR normalization). The scaler is fitted exclusively on the training split and applied unchanged to validation and test splits.

### 3.4 Classification Stage

**XGBoost** — primary model for RAVDESS emotion recognition (Pipeline 2). Gradient-boosted shallow trees with early stopping.

**SVM (RBF, calibrated)** — primary model for MODMA depression screening (Pipeline 2). Isotonic calibration enables probability output.

**EmotionMLP / Fusion MLP** — PyTorch networks used in training experiments (not deployed in the live app).

### 3.5 Output Stage

The deployed app returns predicted class labels with confidence scores and visualisations. For the depression screening task, the model returns calibrated probabilities for both MDD and HC classes. The PHQ-8 self-screener computes a validated clinical score from user responses.

---

## 4. Deployed Application

### 4.1 Modular Application Architecture

The deployed application (`app/`) uses a **4-file modular architecture** — a significant improvement over a monolithic single-file design:

```
app/
├── app.py          — Gradio UI: 5-tab interface, chart helpers, event bindings
├── inference.py    — ML logic: Whisper loading, model loading, predict functions
├── config.py       — API key management (Groq)
├── rag_utils.py    — RAG pipeline: knowledge base, FAISS index, Groq LLM
├── requirements.txt
└── models/
    ├── rav_whisper_xgb.json      — RAVDESS XGBoost model
    └── modma_whisper_svm.pkl     — MODMA SVM pipeline
```

The `hf_space/` folder contains the Hugging Face Spaces deployment version (same files, models loaded from HF Space storage rather than a local `models/` subfolder).

### 4.2 Five-Tab Gradio Interface

The application provides five functional tabs:

| Tab | Function | Model Used |
|---|---|---|
| **Emotion Recognition** | 8-class speech emotion detection | Whisper-small + XGBoost (F1 = 0.974) |
| **Depression Screening** | Binary MDD vs. HC classification | Whisper-small + SVM RBF (F1 = 0.750) |
| **PHQ-8 Self-Screener** | Validated 8-item clinical questionnaire | Rule-based scoring (no ML) |
| **AI Clinical Explanation** | RAG + LLM explanation of results | FAISS + sentence-transformers + Groq LLaMA3 |
| **About & Model Details** | Project documentation and architecture | — |

### 4.3 Inference Module (inference.py)

At startup, `warmup()` pre-loads all models into memory:
1. Whisper-small encoder (~461 MB, downloaded on first run)
2. RAVDESS XGBoost model (from `models/rav_whisper_xgb.json`)
3. MODMA SVM pipeline (from `models/modma_whisper_svm.pkl`, loaded via joblib)

At inference time, `extract_whisper_embedding()` extracts a 1536-D vector from any audio file, which is then passed to `predict_emotion()` or `predict_depression()` as appropriate.

### 4.4 RAG Pipeline (rag_utils.py)

The AI Clinical Explanation tab is powered by a Retrieval-Augmented Generation pipeline:

```
User audio + optional PHQ-8 score
          │
          ▼
Run predict_emotion() + predict_depression()
          │
          ▼
Build retrieval query from results
          │
          ▼
FAISS vector search over 15 clinical knowledge chunks
(sentence-transformers/all-MiniLM-L6-v2 encoder)
          │
          ▼
Top-4 chunks injected into structured clinical prompt
          │
          ▼
Groq API → LLaMA3-70B-Versatile → 350-word explanation
          │
          ▼
Structured 4-paragraph clinical explanation
(results, speech patterns, context, next steps + helplines)
```

The knowledge base covers: WHO depression guidelines, PHQ-8 interpretation, RAVDESS emotion–mental health mappings, acoustic markers of MDD, MODMA dataset context, treatment approaches, and India mental health helplines.

### 4.5 PHQ-8 Self-Screener

Tab 3 implements the validated Patient Health Questionnaire-8 as an interactive radio-button form. Eight clinical questions are presented (each scored 0–3 on frequency). The total score (0–24) is mapped to severity bands: No significant depression (0–4), Mild (5–9), Moderate (10–14), Moderately severe (15–19), Severe (20–24). A visual gauge chart accompanies the score. The clinical threshold of PHQ-8 ≥ 10 is flagged explicitly.

### 4.6 Hosting (Hugging Face Spaces)

The application is hosted on Hugging Face Spaces at:
**[https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis](https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis)**

Hugging Face Spaces manages container build, HTTPS provisioning, REST API generation, and Space sleep/wake lifecycle.

---

## 5. Design Decisions and Rationale

| Decision | Rationale |
|---|---|
| Actor-level splitting (RAVDESS) | Prevents speaker identity leakage — test set is fully unseen speakers |
| Participant-level splitting (DAIC-WOZ, MODMA) | Prevents clinical interview leakage — each participant appears in exactly one split |
| RobustScaler over StandardScaler | Clinical audio features contain extreme outliers from pathological speech segments |
| Frozen Whisper encoder | Avoids fine-tuning 244M parameters on small datasets (N ≤ 189); prevents catastrophic forgetting |
| Whisper-small (1536-D mean+std) over base (512-D mean) | Higher-dimensional embedding captures more prosodic detail; small model still fast on CPU |
| Macro-F1 as primary metric | Penalizes ignoring minority class (depressed patients); accuracy is misleading on imbalanced datasets |
| AUPRC for DAIC-WOZ tuning | More sensitive to minority-class recall than ROC-AUC under 2.57:1 imbalance |
| Threshold sweep on validation set | Default 0.5 threshold is suboptimal for imbalanced datasets; clinical priority is sensitivity |
| RAG over fine-tuned LLM | Keeps knowledge base editable without retraining; grounded in authoritative clinical sources |
| Modular 4-file app structure | Separates UI, inference, config, and RAG concerns for maintainability |
