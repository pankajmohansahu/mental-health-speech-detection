# AI-Based Early Mental Health Breakdown Detection from Speech Patterns

**Course:** Data Science and AI Lab Project (BSDA4001)
**Institution:** Indian Institute of Technology Madras
**Group:** 6

---

## Live Demo

**Deployed on Hugging Face Spaces:**
[https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis](https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis)

> No installation required. Open the link, upload or record a speech sample, and receive an instant prediction.

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

## Overview

This project develops a machine learning system that analyses speech and physiological audio signals to detect mental health indicators including **depression**, **emotional state**, **Major Depressive Disorder (MDD)**, and **physiological stress**. The system is deployed as an interactive 5-tab web application on Hugging Face Spaces using Gradio, augmented with a RAG-powered clinical explanation engine.

Two distinct pipelines are implemented:

- **Pipeline 1 (Baseline):** Handcrafted acoustic features (MFCC, chroma, spectral statistics, HRV) fed into classical ML classifiers (XGBoost, SVM).
- **Pipeline 2 (Proposed):** Frozen OpenAI Whisper encoder producing 1536-dimensional speech embeddings classified by XGBoost/SVM heads -- **+48.8 pp improvement over baseline on RAVDESS**.

---

## App Features (5 Tabs)

| Tab | Feature | Model | Performance |
|---|---|---|---|
| 1 - Emotion Recognition | 8-class emotion from speech | Whisper + XGBoost | F1 = 0.974 |
| 2 - Depression Screening | MDD vs Healthy Control | Whisper + SVM (RBF) | F1 = 0.750 |
| 3 - PHQ-8 Self-Screener | Validated 8-question questionnaire | Scoring algorithm | Clinical standard |
| 4 - AI Clinical Explanation | RAG + LLM explanation | FAISS + Groq LLaMA-3.3-70B | Contextualised |
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

## Best Reported Results

| Dataset | Task | Best Model | Macro-F1 | Notes |
|---|---|---|---|---|
| RAVDESS | 8-class emotion | Whisper + XGBoost | **0.974** | Near-ceiling performance |
| MODMA | MDD vs HC | Whisper + SVM (RBF) | **0.750** | Subject-level split |
| DAIC-WOZ | Binary depression | Fusion MLP (Whisper + clinical) | **0.607** | Sensitivity = 0.857 |
| WESAD | 3-class stress | XGBoost | **0.917** | HRV signals |

**Key finding:** Whisper encoder embeddings outperform hand-crafted features by **+48.8 pp** on RAVDESS (F1: 0.974 vs 0.486). SMOTE class balancing improved minority-class recall across all datasets.

---

## Demo Instructions

1. Open the [live app](https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis)
2. **Tab 1/2:** Click Record from microphone or upload a .wav/.mp3/.flac file, then click Predict
3. **Tab 3:** Answer all 8 PHQ-8 questions and click Calculate Score
4. **Tab 4:** Upload audio + optionally enter your PHQ-8 score, click Generate AI Explanation
5. Results appear within a few seconds (allow 15-30s on first load -- cold start downloads Whisper ~461MB)

| Sample Scenario | Expected Output |
|---|---|
| Flat, monotone speech with slow rate | Sad / Depressed |
| High-energy, rapid speech | Happy / Excited |
| Trembling voice, irregular pitch | Fearful |
| Calm, steady low-energy speech | Neutral / Calm |

---

## Quick Start -- Run Locally

```bash
git clone https://github.com/21f3002286/Group-6-DS-and-AI-Lab-Project.git
cd Group-6-DS-and-AI-Lab-Project
python -m venv venv && source venv/bin/activate
pip install -r app/requirements.txt
# Windows: winget install ffmpeg  |  macOS: brew install ffmpeg  |  Linux: sudo apt install ffmpeg
export GROQ_API_KEY=your_key_here   # for AI explanation tab
cd app && python app.py
# Open http://localhost:7860
```

---

## Repository Structure

```
Group-6-DS-and-AI-Lab-Project/
+-- app/                     Local Gradio app (5 tabs, RAG, PHQ-8)
+-- hf_space/                Hugging Face Spaces deployment variant
+-- docs/                    Full project documentation (overview, technical, user guide, API, licenses)
+-- Notebooks_Models/        Training and experiment notebooks (M3-M6)
+-- Notebooks_EDA/           Exploratory Data Analysis notebooks
+-- processed_data/          Feature-extracted datasets
+-- Reports/                 Milestone reports (M1-M5)
+-- ppts/                    Presentation slides
```

---

## Documentation

| Document | Description |
|---|---|
| [Overview](docs/overview.md) | Architecture, data flow, deployed components |
| [Technical Documentation](docs/technical_doc.md) | Setup, pipeline, models, deployment details |
| [User Guide](docs/user_guide.md) | Non-technical usage guide |
| [API Documentation](docs/api_doc.md) | Gradio API endpoint reference |
| [Licenses](docs/licenses.md) | Code and dataset license information |

---

## Milestones

| Milestone | Description | Status |
|---|---|---|
| M1 | Problem definition, dataset selection, literature review | Done |
| M2 | EDA, baseline models (handcrafted features) | Done |
| M3 | Feature engineering, improved classical ML models | Done |
| M4 | Whisper encoder embeddings, fusion models, hyperparameter tuning | Done |
| M5 | Full model evaluation, error analysis, threshold optimization | Done |
| M6 | Deployment (Gradio + HF Spaces + RAG + PHQ-8 + documentation) | Done |

---

## License

Code: MIT License. See [docs/licenses.md](docs/licenses.md) for full details including dataset attributions.

---

## Disclaimer

This application is a **research prototype** for academic purposes only.
It is **NOT a clinical diagnostic tool** and must **NOT** be used for medical decisions.
If you are concerned about mental health, please consult a qualified healthcare professional.
