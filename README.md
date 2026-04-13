# AI-Based Early Mental Health Breakdown Detection from Speech Patterns

**Course:** Data Science and AI Lab Project (BSDA4001)
**Institution:** Indian Institute of Technology Madras
**Group:** 6

---

## Live Demo

**Deployed on Hugging Face Spaces:**
[https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis](https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis)

---

## Team Members

| Name | Roll Number |
|---|---|
| G Hamsini | 22f3000767 |
| Om Aryan | 21f3002286 |
| Drashti Shah | 22f2001483 |
| Pankaj Mohan Sahu | 21f2001203 |
| Mahi Mudgal | 21f3002602 |

---

## Project Overview

This system detects emotional states and screens for depression indicators from speech audio using:
- **OpenAI Whisper** (speech encoder) for deep acoustic feature extraction
- **Classical ML classifiers** (XGBoost, SVM) trained on clinical datasets
- **RAG pipeline** (FAISS + sentence-transformers + Groq LLaMA-3.3-70B) for AI-powered clinical explanation
- **PHQ-8 questionnaire** for validated self-report depression screening

---

## App Features (5 Tabs)

| Tab | Feature | Model | Performance |
|---|---|---|---|
| 1 - Emotion Recognition | 8-class emotion from speech | Whisper + XGBoost | F1 = 0.974 |
| 2 - Depression Screening | MDD vs Healthy Control | Whisper + SVM (RBF) | F1 = 0.750 |
| 3 - PHQ-8 Self-Screener | Validated 8-question questionnaire | Scoring algorithm | Clinical standard |
| 4 - AI Clinical Explanation | RAG + LLM explanation | FAISS + Groq LLaMA | Contextualised output |
| 5 - About & Model Details | Architecture, datasets, limitations | - | - |

---

## Pipeline Architecture

```
Input Audio (.wav / .mp3 / .flac)
        |
OpenAI Whisper-small Encoder (244M params, frozen)
        |
Mean + Std Pooling over time -> 1536-D Embedding
        |
+-----------------------------------------------+
| Tab 1: XGBoost Classifier  -> 8 Emotions       |  F1 = 0.974
| Tab 2: SVM (RBF) Classifier -> MDD / HC        |  F1 = 0.750
+-----------------------------------------------+
        |
Tab 3: PHQ-8 Self-Report Questionnaire
        |
Tab 4: RAG (FAISS + sentence-transformers)
        + Groq LLaMA-3.3-70B -> Clinical Explanation
```

---

## Datasets

| Dataset | Task | Samples | Best F1 |
|---|---|---|---|
| RAVDESS | Emotion Recognition (8-class) | 2,452 clips | 0.974 |
| MODMA | Depression Screening (binary) | 52 subjects | 0.750 |
| DAIC-WOZ | Depression Screening (binary) | 189 sessions | 0.607 |
| WESAD | Stress Detection (binary) | HRV signals | 0.917 |

---

## Key Results

- Whisper encoder embeddings outperform hand-crafted features by **+48.8 pp** on RAVDESS (F1: 0.974 vs 0.486)
- SMOTE class balancing improved minority-class recall across all datasets
- RAG pipeline provides WHO-guideline-grounded explanations with relevant clinical context

---

## Repository Structure

```
.
+-- app/                        # Local Gradio app (5 tabs, RAG, PHQ-8)
|   +-- app.py
|   +-- inference.py
|   +-- rag_utils.py
|   +-- models/
|       +-- rav_whisper_xgb.json
|       +-- modma_whisper_svm.pkl
+-- hf_space/                   # Hugging Face Spaces deployment
|   +-- app.py
|   +-- requirements.txt
|   +-- README.md
+-- Notebooks_Models/           # Experiment notebooks (M4, M5, M6)
+-- Notebooks_EDA/              # EDA notebooks
+-- processed_data/             # Feature-extracted datasets
+-- Reports/                    # Milestone reports
+-- ppts/                       # Presentation slides
```

---

## Run Locally

```bash
# 1. Install dependencies
pip install -r app/requirements.txt

# 2. Install ffmpeg (required by openai-whisper)
# Windows: winget install ffmpeg
# macOS:   brew install ffmpeg
# Linux:   sudo apt install ffmpeg

# 3. Set Groq API key (for AI explanation tab)
export GROQ_API_KEY=your_key_here   # Linux/macOS
set GROQ_API_KEY=your_key_here      # Windows

# 4. Launch
cd app
python app.py
# Open http://localhost:7860
```

---

## Milestones

| Milestone | Description | Status |
|---|---|---|
| M1 | Problem definition, dataset selection | Done |
| M2 | EDA, baseline models | Done |
| M3 | Feature engineering, improved models | Done |
| M4 | Whisper encoder embeddings, fusion models | Done |
| M5 | Full model evaluation, error analysis | Done |
| M6 | Deployment (Gradio + HF Spaces + RAG) | Done |

---

## Disclaimer

This application is a **research prototype** for academic purposes only.
It is **NOT a clinical diagnostic tool** and must **NOT** be used for medical decisions.
If you are concerned about mental health, please consult a qualified healthcare professional.
