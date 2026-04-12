---
title: Mental Health Speech Analysis
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Mental Health Breakdown Detection from Speech

**AI-Based Early Screening | Group 6 — IIT Madras BSDA4001**

## What This App Does

This system uses **OpenAI Whisper** (speech encoder) combined with classical ML classifiers
to perform two tasks from uploaded audio:

| Tab | Task | Model | Performance |
|---|---|---|---|
| Emotion Recognition | 8-class emotion from speech | Whisper + XGBoost | **F1 = 0.974** |
| Depression Screening | MDD vs Healthy Control | Whisper + SVM (RBF) | **F1 = 0.750** |

## Quick Start — Run Locally

```bash
# 1. Clone the repository
git clone <your-hf-space-url>
cd <space-name>

# 2. Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 3. Install ffmpeg (required by openai-whisper)
# Windows: winget install ffmpeg
# macOS:   brew install ffmpeg
# Linux:   sudo apt install ffmpeg

# 4. Launch
python app.py
# Open http://localhost:7860 in your browser
```

> **Note:** First run downloads the Whisper-small model (~461 MB). Subsequent runs use the cached model.

## Input / Output

**Input:** Any `.wav`, `.mp3`, or `.flac` audio file containing speech.
- Recommended: 3–30 seconds of clear speech
- Sample rate: any (auto-resampled to 16kHz by Whisper)

**Output:**
- Emotion Recognition: predicted emotion label + confidence bar chart (8 classes)
- Depression Screening: MDD/HC prediction + probability bar chart

## Model Architecture

```
Input Audio (.wav)
        ↓
OpenAI Whisper-small Encoder  (244M params, frozen)
        ↓
Mean + Std pooling over time dimension
        ↓
1536-D embedding vector
        ↓
XGBoost Classifier  →  Emotion (8 classes)
SVM (RBF, C=10)     →  Depression (binary)
```

## Files

```
app/
├── app.py              # Gradio interface
├── inference.py        # Model loading + prediction logic
├── requirements.txt    # Python dependencies
├── README.md           # This file (also HF Spaces config)
└── models/
    ├── rav_whisper_xgb.json    # RAVDESS XGBoost (F1=0.974)
    └── modma_whisper_svm.pkl   # MODMA SVM pipeline (F1=0.750)
```

## Disclaimer

This application is a **research prototype** for academic purposes only.
It is **NOT a clinical diagnostic tool** and must **NOT** be used for medical decisions.
If you are concerned about mental health, please consult a qualified healthcare professional.
