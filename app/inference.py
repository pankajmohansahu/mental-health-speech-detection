"""
inference.py — Model loading and prediction logic
AI-Based Early Mental Health Breakdown Detection from Speech Patterns
Group 6 | IIT Madras — BSDA4001
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Ensure ffmpeg is findable via imageio-ffmpeg (works without system PATH) ──
try:
    import imageio_ffmpeg
    _ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass  # fall back to system ffmpeg if imageio-ffmpeg not installed

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"
RAV_XGB_PATH  = MODELS_DIR / "rav_whisper_xgb.json"
MODMA_SVM_PATH = MODELS_DIR / "modma_whisper_svm.pkl"

# ── RAVDESS emotion label map (0-indexed, matches training) ────────────────────
EMOTION_LABELS = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised",
}

EMOTION_EMOJI = {
    "Neutral":   "😐",
    "Calm":      "😌",
    "Happy":     "😄",
    "Sad":       "😢",
    "Angry":     "😠",
    "Fearful":   "😨",
    "Disgust":   "🤢",
    "Surprised": "😲",
}

# ── Global model cache (loaded once, reused across calls) ──────────────────────
_whisper_model  = None
_whisper_module = None
_rav_xgb        = None
_modma_svm      = None
_device         = None


def get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def load_whisper():
    """Load openai-whisper-small model (downloads ~461 MB on first run)."""
    global _whisper_model, _whisper_module
    if _whisper_model is None:
        import whisper
        _whisper_module = whisper
        dev = get_device()
        print(f"[INFO] Loading Whisper-small on {dev} ...")
        _whisper_model = whisper.load_model("small", device=dev)
        _whisper_model.eval()
        print("[INFO] Whisper-small loaded.")
    return _whisper_model, _whisper_module


def load_rav_xgb():
    """Load RAVDESS Whisper+XGBoost model (F1=0.974)."""
    global _rav_xgb
    if _rav_xgb is None:
        from xgboost import XGBClassifier
        _rav_xgb = XGBClassifier()
        _rav_xgb.load_model(str(RAV_XGB_PATH))
        print("[INFO] RAVDESS XGBoost model loaded.")
    return _rav_xgb


def load_modma_svm():
    """Load MODMA Whisper+SVM pipeline (StandardScaler+SVC, F1=0.750).
    Saved with joblib on Kaggle — must be loaded with joblib."""
    global _modma_svm
    if _modma_svm is None:
        import joblib
        _modma_svm = joblib.load(str(MODMA_SVM_PATH))
        print("[INFO] MODMA SVM pipeline loaded.")
    return _modma_svm


def extract_whisper_embedding(audio_path: str) -> np.ndarray:
    """
    Extract 1536-D Whisper embedding from an audio file.

    Process:
        audio → pad/trim to 30s → mel spectrogram (80 bins)
        → Whisper encoder → mean+std pool over time → 1536-D vector

    Args:
        audio_path: path to .wav / .mp3 / .flac audio file

    Returns:
        numpy array of shape (1536,)
    """
    wmodel, wh = load_whisper()
    dev = get_device()

    # Use librosa to load audio — avoids ffmpeg subprocess dependency on Windows.
    # librosa reads .wav/.flac natively; resamples to 16kHz to match Whisper.
    import librosa
    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    audio = audio.astype(np.float32)

    audio = wh.pad_or_trim(audio)                              # 30s window
    mel   = wh.log_mel_spectrogram(audio, n_mels=80).to(dev)  # (80, 3000)

    with torch.no_grad():
        enc = wmodel.encoder(mel.unsqueeze(0)).squeeze(0)      # (T, 768)
        emb = torch.cat([enc.mean(0), enc.std(0)])             # (1536,)
        emb = emb.float().cpu().numpy()

    return emb


def predict_emotion(audio_path: str):
    """
    Predict emotion from speech using Whisper+XGBoost (RAVDESS, 8 classes).

    Returns:
        dict with keys:
            - 'label'       : predicted emotion string
            - 'emoji'       : emoji for the emotion
            - 'confidence'  : confidence score for top prediction (0-1)
            - 'probabilities': dict {emotion_name: probability}
            - 'model_info'  : description string
    """
    emb  = extract_whisper_embedding(audio_path)
    xgb  = load_rav_xgb()

    probs_arr = xgb.predict_proba(emb.reshape(1, -1))[0]   # shape (8,)
    pred_idx  = int(np.argmax(probs_arr))
    pred_label = EMOTION_LABELS[pred_idx]

    prob_dict = {
        EMOTION_LABELS[i]: float(probs_arr[i])
        for i in range(len(EMOTION_LABELS))
    }

    return {
        "label":         pred_label,
        "emoji":         EMOTION_EMOJI[pred_label],
        "confidence":    float(probs_arr[pred_idx]),
        "probabilities": prob_dict,
        "model_info":    "Whisper-small encoder (1536-D) + XGBoost | RAVDESS Test F1 = 0.974",
    }


def predict_depression(audio_path: str):
    """
    Screen for depression indicators using Whisper+SVM (MODMA, binary: MDD vs HC).

    DISCLAIMER: This is a research prototype trained on Mandarin clinical audio.
    It is NOT a clinical diagnostic tool. Results must not be used for medical decisions.

    Returns:
        dict with keys:
            - 'label'      : 'MDD (Depressed)' or 'HC (Healthy Control)'
            - 'prob_mdd'   : probability of MDD class (0-1)
            - 'prob_hc'    : probability of HC class (0-1)
            - 'confidence' : confidence in prediction
            - 'model_info' : description string
    """
    emb = extract_whisper_embedding(audio_path)
    svm = load_modma_svm()

    probs_arr = svm.predict_proba(emb.reshape(1, -1))[0]   # shape (2,) — [HC, MDD]
    pred_idx  = int(np.argmax(probs_arr))

    label = "MDD — Possible Depression Indicators" if pred_idx == 1 else "HC — No Depression Indicators"
    confidence = float(probs_arr[pred_idx])

    return {
        "label":      label,
        "prob_hc":    float(probs_arr[0]),
        "prob_mdd":   float(probs_arr[1]),
        "confidence": confidence,
        "model_info": "Whisper-small encoder (1536-D) + SVM (RBF, C=10) | MODMA Test F1 = 0.750",
    }


def warmup():
    """Pre-load all models at startup so first inference is fast."""
    print("[WARMUP] Loading all models...")
    load_whisper()
    load_rav_xgb()
    load_modma_svm()
    print("[WARMUP] All models ready.")
