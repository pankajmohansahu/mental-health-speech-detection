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
- **Pipeline 2 (Proposed):** Frozen OpenAI Whisper encoder producing rich 512-dimensional speech embeddings, optionally fused with handcrafted clinical features before a lightweight MLP classifier.

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

### Pipeline 2 — Whisper Embeddings + Fusion MLP

```
Raw Audio (WAV)
          │
          ▼
  ┌──────────────────────────────────────────┐
  │   Whisper Encoder (openai/whisper-base)   │
  │   Frozen — 80M parameters not updated    │
  │   30-second chunks → 512-dim embedding   │
  │   Mean-pool across chunks per session    │
  └──────────────────────────────────────────┘
          │
          ▼
  [Optional] Concatenate with 448-D clinical
  handcrafted features (DAIC-WOZ only)
  → 960-D fused representation
          │
          ▼
  ┌──────────────────────────────────────────┐
  │   Fusion MLP Head                        │
  │   LayerNorm → Linear → ReLU → Dropout   │
  │   → Softmax / Sigmoid output            │
  └──────────────────────────────────────────┘
          │
          ▼
  Class Label or PHQ-8 Score
```

---

## 3. Data Flow (Detailed)

### 3.1 Audio Input Stage

Raw audio files (WAV/MP3/FLAC) are loaded using `librosa` at 22,050 Hz sample rate for handcrafted feature extraction, or converted to 16,000 Hz for Whisper encoding. Silent segments are detected and trimmed. Invalid values (NaN, ±Inf) arising from silent or corrupted segments are replaced with zero.

### 3.2 Feature Extraction Stage

**Handcrafted features (Pipeline 1):**
- Per-file features are extracted across the entire recording (no windowing for RAVDESS).
- For DAIC-WOZ, per-frame CSV statistics are already provided; aggregation (mean, std) produces session-level vectors.
- MODMA applies per-segment statistics and then subject-level aggregation.

**Whisper embeddings (Pipeline 2):**
- Audio is chunked into 30-second windows with 1-second overlap.
- Each chunk is passed through the frozen Whisper encoder to produce a 512-dimensional representation.
- Chunks within a session are mean-pooled into a single session embedding.

### 3.3 Normalization Stage

All features are normalized with `RobustScaler` (median + IQR normalization), chosen over `StandardScaler` for robustness to the extreme outliers common in clinical acoustic features (e.g., COVAREP values during pathological speech). The scaler is fitted exclusively on the training split and applied unchanged to validation and test splits.

### 3.4 Classification/Regression Stage

**XGBoost** — primary model for all datasets with small-to-medium N. Gradient-boosted shallow trees (`max_depth` ∈ {2,3}) with early stopping on validation loss. Binary and multi-class variants depending on task.

**SVM (RBF, calibrated)** — primary model for MODMA (N=52) where margin maximization provides implicit regularization on very small datasets. Isotonic calibration enables soft-vote ensemble construction.

**EmotionMLP** — 4-layer PyTorch network (338→256→128→64→8) with BatchNorm, ReLU, Dropout(0.3), AdamW optimizer. Used for RAVDESS 8-class classification.

**Fusion MLP** — Combines Whisper 512-D + clinical 448-D features → 960-D → LayerNorm → 512 → 256 → 2. Used for DAIC-WOZ binary classification.

### 3.5 Output Stage

The deployed app returns a predicted class label with a confidence score. In the clinical depression context, a threshold-optimized operating point (tuned on the validation set) is applied before final prediction to maximize sensitivity (recall of depressed cases).

---

## 4. Deployed Components

### 4.1 Gradio Web Interface

The Gradio UI provides:
- An **audio input widget** supporting file upload (WAV, MP3, FLAC) and direct microphone recording.
- A **prediction display panel** showing the predicted mental state or emotion label and the model's confidence.
- A **description panel** explaining the output labels and their clinical meaning.

All user interaction, preprocessing, model inference, and result rendering happen within a single `app.py` monolithic application.

### 4.2 Inference System

At startup, `app.py`:
1. Loads the serialized model artifact (XGBoost `.json`, MLP `.pth`, or Whisper encoder checkpoint).
2. Reconstructs the preprocessing pipeline (scaler, feature extractor).
3. Registers the Gradio interface and binds the `predict()` function to the UI submit event.

At inference time, the `predict()` function:
1. Receives raw audio bytes from the Gradio component.
2. Decodes and resamples the audio.
3. Extracts features (handcrafted or Whisper embeddings).
4. Applies the fitted scaler.
5. Runs inference through the loaded model.
6. Returns the label and confidence to the Gradio output component.

### 4.3 Hosting (Hugging Face Spaces)

The application is hosted on Hugging Face Spaces at:
**[https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis](https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis)**

Hugging Face Spaces manages:
- Container build from `requirements.txt`
- HTTPS endpoint provisioning
- Auto-generated Gradio API endpoint (`/run/predict`)
- Space sleep/wake lifecycle management

---

## 5. Design Decisions and Rationale

| Decision | Rationale |
|---|---|
| Actor-level splitting (RAVDESS) | Prevents speaker identity leakage — test set is fully unseen speakers |
| Participant-level splitting (DAIC-WOZ, MODMA) | Prevents clinical interview leakage — each participant appears in exactly one split |
| RobustScaler over StandardScaler | Clinical audio features contain extreme outliers from pathological speech segments |
| Frozen Whisper encoder | Avoids fine-tuning 80M parameters on small datasets (N ≤ 189); prevents catastrophic forgetting |
| Macro-F1 as primary metric | Penalizes ignoring minority class (depressed patients); accuracy is misleading on imbalanced datasets |
| AUPRC for DAIC-WOZ tuning | More sensitive to minority-class recall than ROC-AUC under 2.57:1 imbalance |
| Threshold sweep on validation set | Default 0.5 threshold is suboptimal for imbalanced datasets; clinical priority is sensitivity |
