# AI-Based Early Mental Health Breakdown Detection from Speech Patterns

> **Course:** Data Science and AI Lab Project (BSDA4001)
> **Institution:** Indian Institute of Technology Madras
> **Group:** 6

---

## Overview

This project develops a machine learning system that analyzes speech and physiological audio signals to detect mental health indicators including **depression**, **emotion state**, **Major Depressive Disorder (MDD)**, and **physiological stress**. The system is deployed as an interactive web application on Hugging Face Spaces using Gradio.

Two distinct pipelines are implemented:

- **Pipeline 1 (Baseline):** Handcrafted acoustic features (MFCC, chroma, spectral statistics, HRV) fed into classical ML classifiers (XGBoost, SVM).
- **Pipeline 2 (Proposed):** Frozen OpenAI Whisper encoder producing 512-dimensional speech embeddings, optionally fused with clinical handcrafted features, then classified by an MLP head.

---

## Problem Statement

Mental health disorders — particularly depression — are severely under-diagnosed worldwide. Clinical interviews are time-consuming, subjective, and inaccessible at scale. Acoustic biomarkers embedded in speech (pitch variability, vocal energy, speaking rate, prosodic patterns) carry measurable signals of affective and cognitive state. This project investigates whether machine learning applied to speech features can reliably differentiate emotional states and detect depressive episodes in a clinically meaningful way.

---

## Key Features

- **Multi-dataset evaluation** across four independent datasets: RAVDESS (emotion), DAIC-WOZ (depression), MODMA (MDD vs. HC), and SWELL (physiological stress)
- **Two-pipeline architecture**: classical MFCC baseline and Whisper deep-embedding pipeline
- **Clinical-grade evaluation**: Macro-F1, Sensitivity, AUC-ROC, AUPRC — not just accuracy
- **Threshold optimization** for clinical deployment (sensitivity-prioritized operation point)
- **Real-time inference** via Gradio web interface with audio upload or microphone recording
- **Actor-level and participant-level data splits** to prevent identity leakage across train/test

---

## Live Demo

**Deployed on Hugging Face Spaces:**

[https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis](https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis)

> No installation required. Open the link and upload or record a speech sample to receive an instant prediction.

---

## Demo Instructions

1. Navigate to the Hugging Face Space link above.
2. Click **"Record from microphone"** or **"Upload audio file"** (WAV, MP3, or FLAC accepted).
3. If recording, speak naturally for 5–30 seconds, then click **Stop**.
4. Click **Submit / Predict**.
5. The app returns:
   - The predicted **emotional / mental state label**
   - A **confidence score** for the prediction
6. Results appear within a few seconds (allow 10–30 s on first load due to cold start).

### Sample Inputs

| Scenario | Expected Output |
|---|---|
| Flat, monotone speech with slow rate | Depressed / Sad |
| High-energy, rapid speech | Happy / Excited |
| Trembling voice, irregular pitch | Fearful |
| Calm, steady low-energy speech | Neutral / Calm |

> **Note:** This system is a research prototype. It is not a clinical diagnostic tool and should not be used for medical decision-making.

---

## Best Reported Results

| Dataset | Task | Best Model | Macro-F1 | Notes |
|---|---|---|---|---|
| RAVDESS | 8-class emotion | Whisper + XGBoost | **0.9741** | 97.4% — near-ceiling performance |
| DAIC-WOZ | Binary depression | Fusion MLP (Whisper + clinical) | **0.6068** | Sensitivity = 0.857 |
| MODMA | MDD vs. HC | SVM (C=100, RBF) | **0.8730** | Subject-level split |
| SWELL-HRV | 3-class stress | XGBoost | **0.5083** | Balanced classes |

---

## Quick Start — Local Setup

### Prerequisites

- Python 3.10 or higher
- `pip` package manager
- (Recommended) NVIDIA GPU with CUDA for Whisper extraction

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/Group-6-DS-and-AI-Lab-Project.git
cd Group-6-DS-and-AI-Lab-Project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Gradio app locally
python app.py
```

The app will be available at `http://localhost:7860` in your browser.

---

## Repository Structure

```
Group-6-DS-and-AI-Lab-Project/
│
├── app.py                          # Gradio inference app (entry point)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── Notebooks_EDA/                  # Exploratory Data Analysis notebooks
│   ├── DAICWOZ_Feature_Extraction.ipynb
│   ├── MODMA_Audio_EDA_and_Feature_Extraction.ipynb
│   ├── RAVDESS_EDA_and_Feature_Extraction.ipynb
│   └── Stress_EDA_and_Preparation.ipynb
│
├── Notebooks_Models/               # Training and experiment notebooks
│   ├── Milestone3_Model_Architecture_(6)_1.ipynb
│   ├── Milestone4_Experiments (1–15).ipynb
│
├── data/                           # Dataset links and metadata
│   └── dataset_links.txt
│
├── processed_data/                 # Preprocessed feature files
│   ├── daicwoz/
│   ├── modma_audio/
│   ├── ravdess/
│   └── stress/
│
├── Reports/                        # Milestone reports
│   ├── Milestone 1 - Group 6.md
│   ├── Milestone 2 - Group 6.md
│   ├── Milestone 3 - Group 6.md
│   ├── Milestone 4 - Group 6.md
│   └── Milestone 5 - Group6.md
│
└── docs/                           # Full project documentation
    ├── overview.md
    ├── technical_doc.md
    ├── user_guide.md
    ├── api_doc.md
    └── licenses.md
```

---

## Documentation

Full documentation is available in the [`/docs`](docs/) directory:

| Document | Description |
|---|---|
| [Overview](docs/overview.md) | Architecture, data flow, deployed components |
| [Technical Documentation](docs/technical_doc.md) | Setup, pipeline, models, deployment details |
| [User Guide](docs/user_guide.md) | Non-technical usage guide |
| [API Documentation](docs/api_doc.md) | Gradio API endpoint reference |
| [Licenses](docs/licenses.md) | Code and dataset license information |

---

## Team

| Name | Roll Number | 
|---|---|
| G Hamsini | 22f3000767 |
| Om Aryan | 21f3002286 | Model 
| Drashti Shah | 22f2001483 | 
| Pankaj Mohan Sahu | 21f2001203 | 
| Mahi Mudgal | 21f3002602 | 

---

## License

Code: MIT License. See [docs/licenses.md](docs/licenses.md) for full details including dataset attributions.
