# Technical Documentation

## AI-Based Early Mental Health Breakdown Detection from Speech Patterns

**Group 6 | BSDA4001 | IIT Madras**

---

## 1. Environment Setup

### 1.1 Python Version

Python **3.10** is required. The codebase uses match-case syntax and type hints compatible with Python 3.10+.

### 1.2 Dependencies (app/requirements.txt)

```text
gradio==4.44.0
openai-whisper==20240930
xgboost==2.1.3
scikit-learn==1.5.2
joblib>=1.3.0
torch>=2.0.0
numpy>=1.24.0,<2.0
librosa>=0.10.0
matplotlib>=3.7.0
ffmpeg-python>=0.2.0
imageio-ffmpeg>=0.4.9
faiss-cpu>=1.7.4
sentence-transformers==2.7.0
groq>=0.9.0
```

Key additions over a minimal ML stack:
- `faiss-cpu` — vector similarity search for RAG retrieval
- `sentence-transformers` — encodes queries and knowledge chunks for FAISS
- `groq` — Groq API client for LLaMA3-70B generation
- `imageio-ffmpeg` — bundled ffmpeg binary (avoids system PATH dependency on Windows/HF Spaces)

### 1.3 Installation Steps

```bash
# Step 1: Clone repository
git clone https://github.com/your-org/Group-6-DS-and-AI-Lab-Project.git
cd Group-6-DS-and-AI-Lab-Project

# Step 2: Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate.bat       # Windows

# Step 3: Install app dependencies
pip install --upgrade pip
pip install -r app/requirements.txt

# Step 4: Set Groq API key (required for Tab 4 RAG explanation)
export GROQ_API_KEY=gsk_...      # Linux/macOS
set GROQ_API_KEY=gsk_...         # Windows

# Step 5: Verify GPU availability (optional but recommended)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Step 6: Launch app
python app/app.py
```

The application will be available at `http://localhost:7860`.

---

## 2. Data Pipeline

### 2.1 Data Sources

| Dataset | Task | Source | Samples | Notes |
|---|---|---|---|---|
| RAVDESS | 8-class emotion | Ryerson Audio-Visual DB of Emotional Speech and Song | 1,440 WAV files | 24 professional actors, 8 emotions |
| DAIC-WOZ | Depression (binary + PHQ-8) | Distress Analysis Interview Corpus — Wizard of Oz | 189 clinical interview sessions | Psychiatrist-administered PHQ-8 labels |
| MODMA | MDD vs. HC | Multi-modal Open Dataset for Mental-disorder Analysis | 52 subjects (23 MDD, 29 HC) | Real clinical patients |
| SWELL | Physiological stress | SWELL Knowledge Work Dataset | 204,885 HRV windows; 51,741 EDA windows | 3-class stress conditions |

Dataset download links are provided in `data/dataset_links.txt`.

### 2.2 Preprocessing

#### RAVDESS

1. Audio files are loaded with `librosa.load()` at 22,050 Hz.
2. Amplitude normalization and silence trimming are applied.
3. No segmentation — features represent the entire recording.
4. Split strategy: **actor-level** — actors 1–18 (train), 19–22 (validation), 23–24 (test). This ensures that test set speakers are completely unseen during training, preventing speaker identity leakage.

#### DAIC-WOZ

1. Pre-extracted per-frame feature CSVs (COVAREP acoustic, Action Units visual, NLP linguistic) are loaded per session.
2. Session-level aggregation: mean and standard deviation computed per feature across all frames.
3. Extended feature set (up to 2,231 dimensions) adds higher-order statistics (skewness, kurtosis, 10th/25th/75th/90th percentiles) to COVAREP features.
4. Labels are derived from PHQ-8 scores: binary target = (PHQ-8 ≥ 10).
5. Split strategy: **participant-level** — each participant's session appears in exactly one split (train: 107, val: 35, test: 47).
6. Class imbalance: 77 healthy vs. 30 depressed in training set. `scale_pos_weight = 2.567`.

#### MODMA

1. Raw WAV files are loaded per subject and segmented into overlapping windows.
2. Per-segment statistics (mean, std, min, max, median) are computed for all acoustic features.
3. Subject-level aggregation reduces each subject to a single feature vector (~500+ dimensions).
4. Split strategy: **subject-level** — 36 train, 8 validation, 8 test.

#### SWELL (HRV/EDA)

1. Time-domain HRV features: RMSSD, SDNN, pNN50, mean RR interval, LF/HF ratio, total power.
2. EDA features: tonic level, phasic component statistics, rise/decay times.
3. Split strategy: `GroupShuffleSplit` by subject ID (70% train, 15% val, 15% test) to prevent subject leakage.
4. Features clipped to [−10, +10] after `RobustScaler` to prevent extreme outlier gradients.

### 2.3 Feature Extraction

#### Handcrafted Acoustic Features (Pipeline 1)

The 338-dimensional RAVDESS feature vector is composed as follows:

| Feature Group | Dimensions | Extraction Method |
|---|---|---|
| MFCC (coeff 1–40, mean + std) | 80 | `librosa.feature.mfcc(n_mfcc=40)` |
| MFCC delta (mean + std) | 80 | `librosa.feature.delta()` |
| Chroma STFT (mean + std) | 24 | `librosa.feature.chroma_stft(n_chroma=12)` |
| Mel spectrogram (mean, 128 bins) | 128 | `librosa.feature.melspectrogram(n_mels=128)` |
| Spectral centroid (mean + std) | 2 | `librosa.feature.spectral_centroid()` |
| Spectral bandwidth (mean + std) | 2 | `librosa.feature.spectral_bandwidth()` |
| Spectral rolloff (mean + std) | 2 | `librosa.feature.spectral_rolloff()` |
| Spectral contrast × 7 bands (mean + std) | 14 | `librosa.feature.spectral_contrast()` |
| Zero crossing rate (mean + std) | 2 | `librosa.feature.zero_crossing_rate()` |
| RMS energy (mean + std) | 2 | `librosa.feature.rms()` |
| Duration | 1 | `librosa.get_duration()` |
| **Total** | **337** | + 1 overlap correction = **338** |

#### Whisper Embeddings (Pipeline 2)

```python
# Pseudocode — Whisper feature extraction
import whisper
import numpy as np

model = whisper.load_model("base")  # 74M params, frozen

def extract_whisper_embedding(audio_path, chunk_sec=30):
    audio = whisper.load_audio(audio_path)          # resample to 16kHz
    chunk_len = chunk_sec * 16000
    chunks = [audio[i:i+chunk_len]
              for i in range(0, len(audio), chunk_len)
              if len(audio[i:i+chunk_len]) > 16000]
    chunk_embeddings = []
    for chunk in chunks:
        chunk = whisper.pad_or_trim(chunk)           # pad/trim to 30s
        mel = whisper.log_mel_spectrogram(chunk)     # 80-bin mel
        with torch.no_grad():
            enc = model.encoder(mel.unsqueeze(0))    # → [1, T, 512]
        # Mean + std pooling → 1024-D per chunk
        emb = torch.cat([enc.squeeze(0).mean(0),
                         enc.squeeze(0).std(0)]).cpu().numpy()
        chunk_embeddings.append(emb)
    return np.mean(chunk_embeddings, axis=0)         # → 1024-D session vector
```

For classification, only the mean-pool (512-D) is used; the std extension to 1024-D applies to the DAIC-WOZ Whisper extraction path where temporal variability is clinically informative.

---

## 3. Model Architecture

### 3.1 XGBoost

Used as the primary classical classifier for all four datasets.

| Parameter | RAVDESS | DAIC-WOZ | MODMA |
|---|---|---|---|
| `objective` | `multi:softprob` | `binary:logistic` | `binary:logistic` |
| `n_estimators` | 300 (early stop) | 200–300 | 200 |
| `max_depth` | 3 | 2 | 3 |
| `learning_rate` | 0.10 | 0.05 | 0.05 |
| `subsample` | 0.80 | 0.70 | 0.80 |
| `colsample_bytree` | 0.80 | 0.70 | 0.80 |
| `reg_alpha` (L1) | 0.10 | 0.50 | 0.50 |
| `reg_lambda` (L2) | 1.50 | 2.00 | 2.00 |
| `scale_pos_weight` | — | 2.567 | per fold |
| `eval_metric` | `mlogloss` | `aucpr` | `logloss` |
| `early_stopping_rounds` | 30 | 40 | 30 |
| `tree_method` | `hist` | `hist` | `hist` |

Shallow tree depth (`max_depth=2` for DAIC-WOZ) is mandatory on small datasets — deeper trees learn patient-specific artifacts rather than generalizable acoustic patterns.

### 3.2 Support Vector Machine

Used for RAVDESS (secondary) and MODMA (primary).

- Kernel: RBF
- `C`: 100 (MODMA), grid-searched ∈ {0.1, 1, 5, 10, 50, 100} (RAVDESS)
- Calibration: `CalibratedClassifierCV(method='isotonic', cv=5)` — required for soft-vote ensemble
- Input: `RobustScaler`-normalized features
- At `C=100`, MODMA features are near-linearly separable after `SelectKBest` dimensionality reduction

### 3.3 PyTorch MLP — EmotionMLP (RAVDESS)

**Base architecture (Experiments 1–6):**

```
Input (338)
  → Linear(338 → 256) + BatchNorm1d + ReLU + Dropout(0.3)
  → Linear(256 → 128) + BatchNorm1d + ReLU + Dropout(0.3)
  → Linear(128 → 64)  + BatchNorm1d + ReLU + Dropout(0.2)
  → Linear(64 → 8)
  → Softmax (inference) / CrossEntropyLoss (training)
```

**Residual architecture (Experiments 7+):**

```
Input (150 PCA-reduced)
  → Stem: Linear(150→256) + BN + ReLU + Dropout(0.3)
  → ResBlock(256): Linear(256→256) + BN + ReLU + Dropout(0.2) + identity skip
  → Linear(256→128) + BN + ReLU + Dropout(0.3)
  → ResBlock(128): Linear(128→128) + BN + ReLU + Dropout(0.2) + identity skip
  → Linear(128→64) + ReLU + Dropout(0.2)
  → Linear(64→8)
```

Residual skip connections address the vanishing gradient degradation observed in Experiments 2–3 where val F1 dropped from 0.46 to 0.32 between epochs 30 and 100 in deeper networks.

### 3.4 Fusion MLP — Pipeline 2 (DAIC-WOZ)

```
Input: Whisper(512) ⊕ Clinical features(448) = 960-D
  → LayerNorm(960)
  → Linear(960 → 512) + ReLU + Dropout(0.4)
  → Linear(512 → 256) + ReLU + Dropout(0.3)
  → Linear(256 → 2)
  → Sigmoid (binary) / CrossEntropyLoss with pos_weight
```

Class weight: `pos_weight = scale_pos_weight = 2.567` in `BCEWithLogitsLoss`.

### 3.5 StressMLP (SWELL/WESAD)

Three depth configurations compared:

| Configuration | Hidden Layers | Parameters |
|---|---|---|
| Shallow | [64] | ~6K |
| Medium | [128, 64] | ~16K |
| Deep | [256, 128, 64, 32] | ~51K |

Best: `BatchNorm + Dropout(0.3) + AdamW`, Shallow configuration for SWELL-HRV.

---

## 4. Training Summary

### 4.1 Training Setup

| Component | Value |
|---|---|
| Platform | Kaggle Notebooks (NVIDIA Tesla T4, 16 GB VRAM) |
| Language | Python 3.10 |
| Framework | PyTorch 2.0, Scikit-learn 1.3, XGBoost 1.7 |
| Random seed | 42 (all frameworks) |
| Reproducibility | Model artifacts saved: XGBoost `.json`, SVM `.pkl`, MLP `.pth` |

### 4.2 MLP Hyperparameters

| Parameter | Value | Selection Method |
|---|---|---|
| Optimizer | AdamW | Compared vs SGD, Adam, RMSprop — AdamW wins by ~1.5% F1 |
| Learning rate | 1×10⁻³ | Grid search |
| Weight decay | 1×10⁻⁴ | Dropout × weight_decay ablation |
| Dropout | 0.3 (hidden), 0.2 (final) | Ablation grid |
| Loss | `CrossEntropyLoss(label_smoothing=0.05)` | From Experiment 7+ |
| Batch size | 64 (RAVDESS), 256 (SWELL large) | Memory-constrained |
| Max epochs | 100 (RAVDESS), 60 (Fusion MLP), 50 (Stress) | — |
| Early stopping | Patience = 10–20 on val macro-F1 | — |
| LR scheduler | `CosineAnnealingLR` | Grid: no-sched, ReduceLR, Cosine |
| Gradient clipping | `max_norm=1.0` | Applied from Experiment 5 onwards |
| Weight initialization | Kaiming Normal (fan_in) | PyTorch default for ReLU nets |

### 4.3 Loss Functions

| Task | Loss Function | Reason |
|---|---|---|
| Multi-class classification (RAVDESS) | `CrossEntropyLoss` | Standard for 8-class |
| Binary classification with imbalance (DAIC-WOZ) | `BCEWithLogitsLoss(pos_weight=2.567)` | Up-weights positive class |
| PHQ-8 regression | `MSELoss` | Penalizes large errors quadratically |
| PHQ-8 regression (alternative) | Huber loss | Robust to extreme outliers — RMSE=6.843 vs MSE best=6.515 |

---

## 5. Evaluation Summary

### 5.1 Metrics Used

| Task | Primary Metric | Secondary Metrics | Justification |
|---|---|---|---|
| RAVDESS emotion (8-class) | **Macro-F1** | Accuracy, per-class F1 | Equal class weight; no majority-class inflation |
| DAIC-WOZ depression (binary) | **Macro-F1 + Sensitivity** | AUC-ROC, AUPRC, Specificity | Missing a depressed case is clinically costly |
| DAIC-WOZ PHQ-8 (regression) | **RMSE** | MAE | Quadratic penalty appropriate for clinical scores |
| MODMA MDD vs. HC | **Macro-F1** | AUC-ROC | Accuracy unreliable at N=8 test set |
| SWELL stress | **Macro-F1** | Accuracy | Equal class weight for 3-class task |

Accuracy is explicitly avoided as a primary metric for DAIC-WOZ because a majority-class predictor achieves 72% accuracy trivially (80% healthy in test set).

AUPRC (Area Under Precision-Recall Curve) is used for DAIC-WOZ branch tuning (`eval_metric='aucpr'` in XGBoost) because AUC-ROC can appear inflated under severe class imbalance — it remains high even when the model rarely predicts the positive class correctly.

### 5.2 Best Results per Dataset

| Dataset | Task | Best Model | Test Accuracy | Test Macro-F1 | Additional |
|---|---|---|---|---|---|
| RAVDESS | 8-class emotion | Whisper + XGBoost (Pipeline 2) | 0.9722 | **0.9741** | +48.8 pp over MFCC baseline |
| DAIC-WOZ | Binary depression | Fusion MLP (Whisper + clinical) | 0.6739 | **0.6068** | Sensitivity = **0.857** |
| DAIC-WOZ | PHQ-8 regression | XGBoost (2,231 enriched features) | — | — | RMSE = **6.515** |
| MODMA | MDD vs. HC | SVM (C=100, RBF, calibrated) | 0.875 | **0.873** | N=8 test set |
| SWELL-HRV | 3-class stress | XGBoost | 0.5083 | **0.5083** | Best achievable |

### 5.3 Results Interpretation

**RAVDESS:** The Whisper encoder's 512-dimensional embeddings — pretrained on 680,000 hours of speech — encode prosodic, phonetic, and paralinguistic information that 338-dimensional MFCC features cannot represent. This explains the transformative +48.8 percentage-point improvement. Residual errors are concentrated at the Calm/Neutral boundary, where both emotions share identical low-energy, flat-pitch prosody.

**DAIC-WOZ:** The 0.857 sensitivity (correctly detecting 6 of 7 depressed test cases) is the most clinically significant result. The modest macro-F1 of 0.607 reflects the fundamental challenge of predicting a complex psychiatric condition from speech features alone with only 107 training sessions. The Whisper+clinical fusion architecture's complementary signals (deep speech representations + clinical acoustic statistics) outperform either feature set alone.

**MODMA:** SVM's near-perfect separation (F1=0.873) on 8 test subjects demonstrates that MODMA's clinical recordings yield acoustically separable patterns. However, the small test set (N=8) makes these numbers statistically fragile — any single misclassification changes the macro-F1 by 12.5 percentage points.

**SWELL:** All models plateau near 50% macro-F1 on the 3-class stress task, indicating that HRV and EDA features alone are insufficient to reliably distinguish the three stress conditions at the session level. The binary stress task achieves 0.6786 accuracy but only 0.4043 macro-F1, reflecting the class imbalance problem.

---

## 6. Inference Pipeline

### 6.1 Step-by-Step Flow

```
1. User submits audio via Gradio UI
       │
       ▼
2. app.py receives: audio_path (temp file path from Gradio)
       │
       ▼
3. Audio decoding and resampling
   - Load with librosa.load(audio_path, sr=22050)
   - Validate: duration > 0.5s, valid waveform
       │
       ▼
4. Feature extraction (one of two paths):
   PATH A — Handcrafted (XGBoost/SVM models):
     extract_features(y, sr) → numpy array [338]
     
   PATH B — Whisper (MLP models):
     resample to 16kHz
     chunk into 30s windows
     for each chunk: whisper_encoder(pad_or_trim(chunk)) → 512-D
     mean-pool all chunks → 512-D session embedding
       │
       ▼
5. Preprocessing: scaler.transform(features)
       │
       ▼
6. Model inference:
   model.predict_proba(features_scaled) → probabilities
       │
       ▼
7. Threshold application:
   For binary tasks: apply tuned threshold (default 0.45 for depression)
   For multi-class: argmax over class probabilities
       │
       ▼
8. Label mapping: integer class → human-readable label
   e.g., 0 → "Neutral", 1 → "Calm", ..., 4 → "Sad"
       │
       ▼
9. Return: (predicted_label, confidence_score) to Gradio UI
```

### 6.2 Actual Inference Functions (inference.py)

```python
# --- Whisper embedding extraction ---
def extract_whisper_embedding(audio_path: str) -> np.ndarray:
    """
    Extract 1536-D Whisper embedding from an audio file.
    audio → 16kHz mono → pad/trim to 30s → 80-bin mel
    → Whisper encoder → mean+std pool over time → 1536-D
    """
    import librosa
    wmodel, wh = load_whisper()
    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    audio = wh.pad_or_trim(audio.astype(np.float32))
    mel = wh.log_mel_spectrogram(audio, n_mels=80).to(get_device())
    with torch.no_grad():
        enc = wmodel.encoder(mel.unsqueeze(0)).squeeze(0)   # (T, 768)
        emb = torch.cat([enc.mean(0), enc.std(0)])          # (1536,)
    return emb.float().cpu().numpy()

# --- Emotion prediction (Tab 1) ---
def predict_emotion(audio_path: str) -> dict:
    emb = extract_whisper_embedding(audio_path)
    xgb = load_rav_xgb()
    probs = xgb.predict_proba(emb.reshape(1, -1))[0]
    pred_idx = int(np.argmax(probs))
    return {
        "label":         EMOTION_LABELS[pred_idx],
        "emoji":         EMOTION_EMOJI[EMOTION_LABELS[pred_idx]],
        "confidence":    float(probs[pred_idx]),
        "probabilities": {EMOTION_LABELS[i]: float(probs[i]) for i in range(8)},
        "model_info":    "Whisper-small (1536-D) + XGBoost | RAVDESS F1 = 0.974",
    }

# --- Depression screening (Tab 2) ---
def predict_depression(audio_path: str) -> dict:
    emb = extract_whisper_embedding(audio_path)
    svm = load_modma_svm()
    probs = svm.predict_proba(emb.reshape(1, -1))[0]   # [HC, MDD]
    pred_idx = int(np.argmax(probs))
    label = "MDD — Possible Depression Indicators" if pred_idx == 1 \
            else "HC — No Depression Indicators"
    return {
        "label":      label,
        "prob_hc":    float(probs[0]),
        "prob_mdd":   float(probs[1]),
        "confidence": float(probs[pred_idx]),
        "model_info": "Whisper-small (1536-D) + SVM (RBF, C=10) | MODMA F1 = 0.750",
    }
```

---

## 7. Deployment Details

### 7.1 Platform: Hugging Face Spaces

The application is hosted on **Hugging Face Spaces** at:
[https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis](https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis)

Hugging Face Spaces provides:
- Managed container runtime with automatic dependency installation from `requirements.txt`
- HTTPS endpoint with SSL termination
- Auto-generated REST API endpoint via Gradio's built-in API layer
- Space sleep/wake management (space sleeps after ~48 hours of inactivity)
- Free-tier CPU compute (2 vCPU, 16 GB RAM)

### 7.2 Framework: Gradio

Gradio is the UI and API framework. Key components used:

| Gradio Component | Purpose |
|---|---|
| `gr.Audio(source="upload")` | File upload widget |
| `gr.Audio(source="microphone")` | Live recording widget |
| `gr.Label()` | Prediction output with confidence bar chart |
| `gr.Interface` | Wraps `predict()` function into a full web UI |

Gradio automatically generates a `/run/predict` REST API endpoint. Any client that can send a multipart HTTP POST request with audio data can receive predictions without using the browser UI.

### 7.3 Architecture: Modular 4-File Design

The deployment uses a **modular architecture** with clear separation of concerns:

```
app/
├── app.py          — Gradio UI only: 5 tabs, chart helpers, event bindings
├── inference.py    — ML logic only: model loading, Whisper embedding, predict functions
├── config.py       — Configuration only: Groq API key, model name
├── rag_utils.py    — RAG pipeline only: knowledge base, FAISS index, Groq LLM
├── requirements.txt
└── models/
    ├── rav_whisper_xgb.json      — RAVDESS XGBoost model (Whisper features)
    └── modma_whisper_svm.pkl     — MODMA SVM pipeline (joblib, Whisper features)
```

`hf_space/` contains the same four Python files prepared for Hugging Face Spaces deployment (models are loaded from HF Space persistent storage rather than a local `models/` directory).

### 7.4 Model Loading at Runtime

All models are loaded once at startup via `warmup()` (called at module import in `app.py`):

```python
# inference.py — called once at startup
def warmup():
    load_whisper()       # openai-whisper-small, ~461 MB, CPU or CUDA
    load_rav_xgb()       # XGBoost from models/rav_whisper_xgb.json
    load_modma_svm()     # SVM pipeline via joblib from models/modma_whisper_svm.pkl

# app.py — invoked at import time
from inference import warmup
warmup()

# RAG index built lazily on first Tab 4 call
# sentence-transformers encoder + FAISS IndexFlatL2 over 15 knowledge chunks
```

Models are cached in module-level globals (`_whisper_model`, `_rav_xgb`, `_modma_svm`) so subsequent requests reuse the same in-memory objects without re-loading from disk.

### 7.4b RAG Pipeline Architecture

```python
# rag_utils.py
KNOWLEDGE_CHUNKS = [...]   # 15 curated clinical knowledge strings

def _build_index():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode([c[1] for c in KNOWLEDGE_CHUNKS])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

def retrieve(query: str, k=4) -> list:
    q_emb = encoder.encode([query]).astype("float32")
    _, indices = index.search(q_emb, k)
    return [texts[i] for i in indices[0]]

def generate_explanation(emotion_result, depression_result, phq8_score):
    query = build_query(emotion_result, depression_result, phq8_score)
    context = "\n---\n".join(retrieve(query, k=4))
    prompt = build_clinical_prompt(emotion_result, depression_result,
                                   phq8_score, context)
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content
```

### 7.5 API: Auto-Generated by Gradio

Gradio automatically exposes REST API endpoints for all interactive tabs:
- `/run/predict` — Emotion Recognition (Tab 1)
- `/run/predict_1` — Depression Screening (Tab 2)
- `/run/predict_2` — PHQ-8 Scoring (Tab 3)
- `/run/predict_3` — RAG Clinical Explanation (Tab 4)

See [docs/api_doc.md](api_doc.md) for the full API reference including request/response formats and Python examples.

### 7.6 User Interaction Flow (5-Tab Application)

1. User opens the Hugging Face Spaces URL.
2. Gradio serves the 5-tab HTML/JS interface.
3. **Tab 1 / Tab 2:** User uploads audio or records from microphone → browser POSTs to `/run/predict` or `/run/predict_1` → `inference.py` extracts 1536-D Whisper embedding → XGBoost/SVM inference → Gradio renders markdown result + confidence bar chart.
4. **Tab 3:** User selects PHQ-8 responses via radio buttons → browser POSTs to `/run/predict_2` → `score_phq8()` in `inference.py` computes total → Gradio renders score gauge chart.
5. **Tab 4:** User uploads audio (optionally enters PHQ-8 score) → `run_rag_explanation()` in `app.py` calls both `predict_emotion()` and `predict_depression()` → RAG retrieves 4 knowledge chunks → Groq LLaMA3-70B generates 350-word explanation → Gradio renders markdown.

### 7.7 Deployment Limitations

| Limitation | Details |
|---|---|
| **Cold start latency** | The Space sleeps after ~48h inactivity. First request after sleep takes 30–60s to wake, load models, and respond. |
| **CPU-only inference** | Free-tier Spaces run on CPU. Whisper embedding extraction is slow (~5–15s per 30s audio clip) compared to GPU. |
| **No persistent storage** | Uploaded audio files are not stored beyond the inference request. No user session management. |
| **Not production-scaled** | Single container, no load balancing, no auto-scaling. Concurrent request handling is limited to Gradio's built-in queue. |
| **Memory constraints** | Loading both Whisper encoder and XGBoost model simultaneously may approach the 16 GB RAM limit on large audio inputs. |

---

## 8. System Design Considerations

### 8.1 Scalability

The current monolithic Gradio deployment is not horizontally scalable. For production deployment, the following decomposition is recommended:

- **Separate feature extraction service**: REST microservice wrapping the Whisper encoder, deployed on GPU hardware.
- **Separate inference service**: REST microservice wrapping the trained classifier (XGBoost/MLP).
- **Queue-based processing**: Async audio processing queue (e.g., Celery + Redis) for handling concurrent users without blocking.
- **Stateless containers**: Model weights stored in object storage (S3/GCS) and loaded at container startup.

### 8.2 Modularity

The existing codebase maintains clear separation between:
- **Data pipeline notebooks** (`Notebooks_EDA/`) — feature extraction and preprocessing
- **Model training notebooks** (`Notebooks_Models/`) — training, ablation, and evaluation
- **Inference application** (`app.py`) — runtime inference only

Feature extraction functions are duplicated between training notebooks and `app.py`. A production refactor would move them into a shared `features.py` module imported by both.

### 8.3 Performance Trade-offs

| Trade-off | Choice Made | Rationale |
|---|---|---|
| Whisper `base` vs. `large` | `base` (74M params) | `large` (1.5B params) is 20× slower; base provides sufficient quality for classification |
| Frozen vs. fine-tuned Whisper | Frozen | Prevents overfitting on N ≤ 189; avoids GPU memory requirements |
| XGBoost vs. MLP on small datasets | XGBoost for N < 200 | XGBoost's inductive bias (shallow trees) outperforms MLP on small, high-dimensional feature sets |
| SMOTE oversampling | Avoided for DAIC-WOZ | At N=107, SMOTE generates unrealistic synthetic points in 448-D clinical feature space |

---

## 9. Error Handling and Monitoring

### 9.1 Input Validation

The `predict()` function validates inputs before processing:

```python
# Duration check
if len(y) < sr * 0.5:
    return "Error: Audio too short (< 0.5 seconds).", 0.0

# Silence check
if np.abs(y).max() < 1e-4:
    return "Error: Audio appears silent. Check microphone or file.", 0.0

# NaN/Inf in features
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
```

### 9.2 Graceful Failure

- If the Whisper encoder fails (e.g., corrupted audio), the system falls back to the handcrafted MFCC feature path.
- If model loading fails at startup, Gradio displays an error message on the UI rather than crashing silently.
- All exceptions in `predict()` are caught and returned as user-readable error strings.

### 9.3 Monitoring

On Hugging Face Spaces, basic monitoring is available through:
- **Space Logs** — stdout/stderr output visible in the HF UI under "Logs" tab
- **Space Metrics** — request volume and uptime visible under "Settings"

No custom logging or alerting infrastructure is deployed in the current prototype.

---

## 10. Reproducibility Checklist

To reproduce all experiments and the deployed application from scratch:

| Step | File / Action | Notes |
|---|---|---|
| 1 | `app/requirements.txt` | Install all dependencies exactly as specified |
| 2 | `data/dataset_links.txt` | Download RAVDESS, DAIC-WOZ, MODMA, SWELL datasets |
| 3 | `Notebooks_EDA/*.ipynb` | Run EDA notebooks to generate `processed_data/` |
| 4 | `Notebooks_Models/Milestone4_Experiments (15).ipynb` | Run full training and hyperparameter search |
| 5 | Export model artifacts | Copy `rav_whisper_xgb.json` and `modma_whisper_svm.pkl` to `app/models/` |
| 6 | Set `GROQ_API_KEY` | Required for Tab 4 RAG explanation |
| 7 | `python app/app.py` | Launch local inference server at `http://localhost:7860` |
| 8 | `hf_space/` | For Hugging Face Spaces deployment: push `hf_space/` contents to the HF Space repo |

All random seeds are fixed at 42 across NumPy, PyTorch (`torch.manual_seed(42)`), and scikit-learn (`random_state=42`). Results may vary by ±0.001–0.005 F1 due to floating-point non-determinism across hardware.
