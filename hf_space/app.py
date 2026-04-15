# Install openai-whisper at runtime (bypasses pip build-isolation bug on HF Spaces)
import subprocess, sys
try:
    import whisper  # noqa: F401
except ImportError:
    print("[INSTALL] openai-whisper not found - installing with --no-build-isolation ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--no-build-isolation", "openai-whisper==20240930",
    ])
    print("[INSTALL] openai-whisper installed.")

# Patch gradio_client 1.3.0 bug: _json_schema_to_python_type() crashes on boolean schemas
# Root cause: gr.Number(value=None) produces additionalProperties=True in the JSON Schema.
# _json_schema_to_python_type(True, defs) reaches the final `raise APIInfoParseError`
# because it has no branch for boolean inputs. This propagates up the ASGI stack and
# kills the server. Patching to return "str" for any boolean schema input prevents the crash.
# The recursive calls inside the function look up the name through module globals (__dict__),
# so replacing _gcu._json_schema_to_python_type also fixes all recursive invocations.
try:
    import gradio_client.utils as _gcu
    _orig_inner = _gcu._json_schema_to_python_type
    def _patched_inner(schema, defs=None):
        if isinstance(schema, bool):
            return "str"
        return _orig_inner(schema, defs)
    _gcu._json_schema_to_python_type = _patched_inner
    print("[PATCH] gradio_client.utils._json_schema_to_python_type patched (boolean schema fix).")
except Exception as _pe:
    print(f"[PATCH] Could not patch gradio_client: {_pe}")

"""
app.py -- Gradio Web Interface
AI-Based Early Mental Health Breakdown Detection from Speech Patterns
Group 6 | IIT Madras — BSDA4001

Tabs:
  1. Emotion Recognition  (RAVDESS Whisper+XGBoost, F1=0.974)
  2. Depression Screening (MODMA  Whisper+SVM,     F1=0.750)
  3. PHQ-8 Self-Screener  (validated questionnaire, 8 items)
  4. AI Clinical Explanation (RAG + Groq LLM)
  5. About & Model Details

Deployment: Hugging Face Spaces (Gradio SDK)
Run locally: python app.py
"""

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inference import (
    predict_emotion, predict_depression, warmup,
    EMOTION_EMOJI, PHQ8_QUESTIONS, PHQ8_OPTIONS, score_phq8,
)

# ── Pre-load ML models at startup ──────────────────────────────────────────────
warmup()

# ── Pre-load RAG encoder + FAISS index at startup ─────────────────────────────
try:
    from rag_utils import generate_explanation, retrieve
    _rag_available = True
    print("[STARTUP] RAG utilities imported.")
except Exception as _rag_err:
    _rag_available = False
    print(f"[STARTUP] RAG not available: {_rag_err}")


# ══════════════════════════════════════════════════════════════════════════════
#  Helper — bar charts
# ══════════════════════════════════════════════════════════════════════════════
def make_emotion_bar_chart(prob_dict: dict, predicted_label: str):
    emotions = list(prob_dict.keys())
    probs    = [prob_dict[e] * 100 for e in emotions]
    labels   = [f"{EMOTION_EMOJI[e]} {e}" for e in emotions]
    colors   = ["#2ecc71" if e == predicted_label else "#95a5a6" for e in emotions]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, probs, color=colors, edgecolor="white", height=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_title("Emotion Prediction — Confidence per Class", fontsize=12, fontweight="bold")

    for bar, val in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def make_depression_bar_chart(prob_hc: float, prob_mdd: float):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    labels = ["HC (Healthy)", "MDD (Depressed)"]
    probs  = [prob_hc * 100, prob_mdd * 100]
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax.barh(labels, probs, color=colors, edgecolor="white", height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", fontsize=11)
    ax.set_title("Depression Screening — Class Probabilities", fontsize=11, fontweight="bold")

    for bar, val in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def make_phq8_gauge(score: int, severity: str, color: str):
    fig, ax = plt.subplots(figsize=(6, 1.8))
    # Background bar (full range)
    ax.barh(["PHQ-8 Score"], [24], color="#ecf0f1", edgecolor="white", height=0.5)
    # Score bar
    ax.barh(["PHQ-8 Score"], [score], color=color, edgecolor="white", height=0.5)
    ax.set_xlim(0, 24)
    ax.set_xlabel("Score (0–24)", fontsize=10)
    ax.set_title(f"PHQ-8 Score: {score}/24 — {severity}", fontsize=11, fontweight="bold")

    # Threshold markers
    for thresh, label in [(5, "Mild"), (10, "Mod."), (15, "Sev.")]:
        ax.axvline(thresh, color="#7f8c8d", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.text(score + 0.3, 0, f"{score}", va="center", fontsize=11, fontweight="bold", color=color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 1 — Emotion Recognition
# ══════════════════════════════════════════════════════════════════════════════
def run_emotion(audio_path):
    if audio_path is None:
        return "Please upload or record an audio file.", None, ""
    try:
        result = predict_emotion(audio_path)
        emoji  = result["emoji"]
        label  = result["label"]
        conf   = result["confidence"] * 100
        summary = (
            f"### {emoji} Predicted Emotion: **{label}**\n\n"
            f"**Confidence:** {conf:.1f}%\n\n"
            f"_{result['model_info']}_"
        )
        chart = make_emotion_bar_chart(result["probabilities"], label)
        return summary, chart, ""
    except Exception as e:
        return f"Error during prediction: {str(e)}", None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 2 — Depression Screening
# ══════════════════════════════════════════════════════════════════════════════
def run_depression(audio_path):
    if audio_path is None:
        return "Please upload or record an audio file.", None, ""
    try:
        result = predict_depression(audio_path)
        label  = result["label"]
        conf   = result["confidence"] * 100
        summary = (
            f"### Screening Result: **{label}**\n\n"
            f"**Confidence:** {conf:.1f}%\n\n"
            f"- P(Healthy): {result['prob_hc']*100:.1f}%\n"
            f"- P(Depressed): {result['prob_mdd']*100:.1f}%\n\n"
            f"_{result['model_info']}_"
        )
        chart = make_depression_bar_chart(result["prob_hc"], result["prob_mdd"])
        return summary, chart, ""
    except Exception as e:
        return f"Error during prediction: {str(e)}", None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 3 — PHQ-8 Self-Screener
# ══════════════════════════════════════════════════════════════════════════════
def run_phq8(*responses):
    """
    Accepts 8 radio-button values like "Not at all (0)" → extracts integer → scores.
    """
    scores = []
    for r in responses:
        if r is None:
            return "Please answer all 8 questions before calculating.", None
        # Extract the integer from "Not at all (0)" → 0
        scores.append(int(r.split("(")[-1].rstrip(")")))

    result = score_phq8(scores)
    score    = result["score"]
    severity = result["severity"]
    color    = result["color"]
    rec      = result["recommendation"]
    thresh   = "**Above clinical threshold** (≥10) — professional evaluation recommended." \
               if result["above_threshold"] else \
               "Below clinical threshold (<10)."

    summary = (
        f"## PHQ-8 Result\n\n"
        f"**Total Score: {score} / 24**\n\n"
        f"**Severity: {severity}**\n\n"
        f"{thresh}\n\n"
        f"{rec}\n\n"
        f"> _This is a self-report screening tool, not a clinical diagnosis. "
        f"Please consult a qualified mental health professional for evaluation._"
    )
    chart = make_phq8_gauge(score, severity, color)
    return summary, chart


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 4 — AI Clinical Explanation (RAG + Groq)
# ══════════════════════════════════════════════════════════════════════════════
def run_rag_explanation(audio_path, phq8_score_input):
    if not _rag_available:
        return (
            "RAG module not available. Please ensure `faiss-cpu`, "
            "`sentence-transformers`, and `groq` are installed."
        )
    if audio_path is None:
        return "Please upload an audio file to generate an explanation."

    try:
        # Run both models on the audio
        emo_result = predict_emotion(audio_path)
        dep_result = predict_depression(audio_path)

        # Parse optional PHQ-8 score
        phq8_score = None
        if phq8_score_input is not None and str(phq8_score_input).strip() != "":
            try:
                phq8_score = int(phq8_score_input)
                phq8_score = max(0, min(24, phq8_score))
            except ValueError:
                pass

        explanation = generate_explanation(emo_result, dep_result, phq8_score)

        header = (
            f"**Audio Analysis Summary**\n"
            f"- Emotion: {emo_result['emoji']} {emo_result['label']} "
            f"({emo_result['confidence']*100:.1f}%)\n"
            f"- Depression screen: {dep_result['label']} "
            f"(P(MDD)={dep_result['prob_mdd']*100:.1f}%)\n"
            + (f"- PHQ-8 score provided: {phq8_score}/24\n" if phq8_score is not None else "")
            + "\n---\n\n"
        )

        return header + explanation

    except Exception as e:
        return f"Error generating explanation: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ══════════════════════════════════════════════════════════════════════════════
DISCLAIMER = """
> **IMPORTANT DISCLAIMER:** This application is a **research prototype** developed as part of an
> academic project (IIT Madras, BSDA4001). It is **NOT a clinical diagnostic tool** and must **NOT**
> be used for medical decisions. If you are concerned about your mental health, please consult a
> qualified healthcare professional.
"""

with gr.Blocks(
    title="Mental Health Speech Analysis",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=".gradio-container { max-width: 960px; margin: auto; }",
) as demo:

    gr.Markdown(
        """
        # Mental Health Breakdown Detection from Speech
        ### AI-Based Early Screening | Group 6 — IIT Madras BSDA4001

        This system uses **OpenAI Whisper** (speech encoder) combined with classical ML models
        and a **RAG-powered AI explainer** to detect emotional states and screen for depression
        indicators from speech audio.
        """
    )
    gr.Markdown(DISCLAIMER)

    with gr.Tabs():

        # ── Tab 1: Emotion Recognition ──────────────────────────────────────
        with gr.TabItem("Emotion Recognition"):
            gr.Markdown(
                """
                ### Speech Emotion Recognition
                Upload a `.wav` / `.mp3` / `.flac` file. Detects **8 emotions**:
                Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.

                **Model:** Whisper-small encoder (1536-D) + XGBoost | **F1 = 0.974**
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    emo_audio = gr.Audio(
                        label="Upload Speech Audio",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    emo_btn = gr.Button("Detect Emotion", variant="primary", size="lg")
                with gr.Column(scale=2):
                    emo_result = gr.Markdown(label="Prediction")
                    emo_chart  = gr.Plot(label="Confidence per Emotion")
                    emo_err    = gr.Textbox(label="Error (if any)", visible=False)

            emo_btn.click(
                fn=run_emotion,
                inputs=[emo_audio],
                outputs=[emo_result, emo_chart, emo_err],
            )

        # ── Tab 2: Depression Screening ─────────────────────────────────────
        with gr.TabItem("Depression Screening"):
            gr.Markdown(
                """
                ### Depression Screening from Speech
                Binary classification: MDD (Depressed) vs HC (Healthy Control).

                **Model:** Whisper-small encoder (1536-D) + SVM (RBF, C=10) | **F1 = 0.750**

                > Trained on MODMA (Mandarin clinical audio). Domain mismatch may affect accuracy.
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    dep_audio = gr.Audio(
                        label="Upload Speech Audio",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    dep_btn = gr.Button("Screen for Depression", variant="primary", size="lg")
                with gr.Column(scale=2):
                    dep_result = gr.Markdown(label="Screening Result")
                    dep_chart  = gr.Plot(label="Class Probabilities")
                    dep_err    = gr.Textbox(label="Error (if any)", visible=False)

            dep_btn.click(
                fn=run_depression,
                inputs=[dep_audio],
                outputs=[dep_result, dep_chart, dep_err],
            )

        # ── Tab 3: PHQ-8 Self-Screener ───────────────────────────────────────
        with gr.TabItem("PHQ-8 Self-Screener"):
            gr.Markdown(
                """
                ### PHQ-8 Depression Self-Assessment
                The **PHQ-8** (Patient Health Questionnaire-8) is a validated clinical screening tool
                used worldwide. Rate how often you have been bothered by each problem **over the last
                2 weeks**.

                > _This is a self-report screening tool and does NOT replace clinical diagnosis._
                """
            )

            phq_radios = []
            for i, question in enumerate(PHQ8_QUESTIONS):
                r = gr.Radio(
                    choices=PHQ8_OPTIONS,
                    label=f"Q{i+1}. {question}",
                    value=None,
                )
                phq_radios.append(r)

            phq_btn    = gr.Button("Calculate PHQ-8 Score", variant="primary", size="lg")
            phq_result = gr.Markdown(label="PHQ-8 Result")
            phq_chart  = gr.Plot(label="Score Gauge")

            phq_btn.click(
                fn=run_phq8,
                inputs=phq_radios,
                outputs=[phq_result, phq_chart],
            )

        # ── Tab 4: AI Clinical Explanation (RAG) ─────────────────────────────
        with gr.TabItem("AI Clinical Explanation (RAG)"):
            gr.Markdown(
                """
                ### AI-Powered Clinical Explanation
                Upload your audio. The system will:
                1. Run **both** emotion and depression models on your speech
                2. **Retrieve** relevant clinical knowledge (WHO guidelines, PubMed research, acoustic markers)
                3. **Generate** a personalised, clinically-grounded explanation using an LLM

                Optionally, enter your **PHQ-8 score** from Tab 3 to enrich the explanation.

                **Powered by:** FAISS vector search + sentence-transformers + Groq (LLaMA3-8B)

                > _All explanations are educational only. Not a clinical diagnosis._
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    rag_audio = gr.Audio(
                        label="Upload Speech Audio",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    rag_phq8 = gr.Number(
                        label="PHQ-8 Score (optional — from Tab 3)",
                        minimum=0,
                        maximum=24,
                        step=1,
                        value=None,
                    )
                    rag_btn = gr.Button(
                        "Generate AI Explanation", variant="primary", size="lg"
                    )
                with gr.Column(scale=2):
                    rag_output = gr.Markdown(label="AI Clinical Explanation")

            rag_btn.click(
                fn=run_rag_explanation,
                inputs=[rag_audio, rag_phq8],
                outputs=[rag_output],
            )

        # ── Tab 5: About ─────────────────────────────────────────────────────
        with gr.TabItem("About & Model Details"):
            gr.Markdown(
                """
                ## About This Project

                **Course:** Data Science and AI Lab Project (BSDA4001)
                **Institution:** Indian Institute of Technology Madras
                **Group:** 6

                ### Team Members
                | Name | Roll Number |
                |---|---|
                | G Hamsini | 22f3000767 |
                | Om Aryan | 21f3002286 |
                | Drashti Shah | 22f2001483 |
                | Pankaj Mohan Sahu | 21f2001203 |
                | Mahi Mudgal | 21f3002602 |

                ---

                ### Pipeline Architecture

                ```
                Input Audio (.wav / .mp3 / .flac)
                        ↓
                OpenAI Whisper-small Encoder (244M params, frozen)
                        ↓
                Mean + Std Pooling over time → 1536-D Embedding
                        ↓
                ┌──────────────────────────────────────────────┐
                │  Tab 1: XGBoost Classifier  → 8 Emotions     │  F1 = 0.974
                │  Tab 2: SVM (RBF) Classifier → MDD / HC      │  F1 = 0.750
                └──────────────────────────────────────────────┘
                        ↓
                Tab 3: PHQ-8 Self-Report Questionnaire
                        ↓
                Tab 4: RAG (FAISS + sentence-transformers)
                        + Groq LLaMA3-8B → Clinical Explanation
                ```

                ### Why Whisper?
                Whisper was pretrained on **680,000 hours** of multilingual speech.
                Its encoder captures deep temporal and prosodic patterns that hand-crafted
                features (MFCC, Mel, Chroma) cannot represent. This is why Whisper+XGBoost
                achieves **F1 = 0.974** vs F1 = 0.486 for the best hand-crafted baseline —
                a **+48.8 percentage point improvement**.

                ### RAG Knowledge Sources
                The AI explainer retrieves from a curated knowledge base covering:
                - WHO depression guidelines and clinical criteria
                - PubMed research on speech markers of depression
                - PHQ-8 score interpretation guidelines
                - MFCC / pitch / prosody clinical meanings
                - RAVDESS emotion-to-mental-health mappings
                - India mental health helplines and resources

                ### Datasets
                | Dataset | Task | Samples | Best F1 |
                |---|---|---|---|
                | RAVDESS | Emotion Recognition (8-class) | 2,452 clips | 0.974 |
                | MODMA | Depression Screening (binary) | 52 subjects | 0.750 |
                | DAIC-WOZ | Depression Screening (binary) | 189 sessions | 0.607 |
                | WESAD | Stress Detection (binary) | HRV signals | 0.917 |

                ### Limitations
                - MODMA model trained on Mandarin clinical audio — may vary on other languages
                - All models are research prototypes, not clinical tools
                - RAG explanations are educational only
                """
            )

    gr.Markdown(
        """
        ---
        *Built with [Gradio](https://gradio.app) | Powered by [OpenAI Whisper](https://github.com/openai/whisper) + [Groq](https://groq.com)*
        """
    )


# ── Launch ────────────────────────────────────────────────────────────────────
demo.launch(server_name="0.0.0.0", show_error=True)
