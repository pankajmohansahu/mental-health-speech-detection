"""
app.py — Gradio Web Interface
AI-Based Early Mental Health Breakdown Detection from Speech Patterns
Group 6 | IIT Madras — BSDA4001

Deployment: Hugging Face Spaces (Gradio SDK)
Run locally: python app.py
"""

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inference import predict_emotion, predict_depression, warmup, EMOTION_EMOJI

# ── Pre-load models at startup ─────────────────────────────────────────────────
warmup()


# ── Helper: create probability bar chart ──────────────────────────────────────
def make_emotion_bar_chart(prob_dict: dict, predicted_label: str):
    emotions = list(prob_dict.keys())
    probs    = [prob_dict[e] * 100 for e in emotions]
    emojis   = [EMOTION_EMOJI[e] for e in emotions]
    labels   = [f"{EMOTION_EMOJI[e]} {e}" for e in emotions]

    colors = ["#2ecc71" if e == predicted_label else "#95a5a6" for e in emotions]

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


# ── Tab 1: Emotion Recognition ─────────────────────────────────────────────────
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


# ── Tab 2: Depression Screening ────────────────────────────────────────────────
def run_depression(audio_path):
    if audio_path is None:
        return "Please upload or record an audio file.", None, ""

    try:
        result = predict_depression(audio_path)
        label  = result["label"]
        conf   = result["confidence"] * 100
        color  = "red" if "MDD" in label else "green"

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


# ── Build Gradio Interface ─────────────────────────────────────────────────────
DISCLAIMER = """
> **IMPORTANT DISCLAIMER:** This application is a **research prototype** developed as part of an
> academic project (IIT Madras, BSDA4001). It is **NOT a clinical diagnostic tool** and must **NOT**
> be used for medical decisions. If you are concerned about your mental health, please consult a
> qualified healthcare professional.
"""

with gr.Blocks(
    title="Mental Health Speech Analysis",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=".gradio-container { max-width: 900px; margin: auto; }",
) as demo:

    gr.Markdown(
        """
        # Mental Health Breakdown Detection from Speech
        ### AI-Based Early Screening | Group 6 — IIT Madras BSDA4001

        This system uses **OpenAI Whisper** (speech encoder) combined with classical ML models
        to detect emotional states and screen for depression indicators from speech audio.
        """
    )
    gr.Markdown(DISCLAIMER)

    with gr.Tabs():

        # ── Tab 1: Emotion Recognition ──────────────────────────────────────
        with gr.TabItem("Emotion Recognition (RAVDESS)"):
            gr.Markdown(
                """
                ### Speech Emotion Recognition
                Upload a `.wav` speech file. The model detects one of **8 emotions**:
                Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.

                **Model:** Whisper-small encoder (1536-D embeddings) + XGBoost
                **Performance:** Macro-F1 = **0.974** on RAVDESS test set (432 samples)
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    emo_audio = gr.Audio(
                        label="Upload Speech Audio (.wav / .mp3 / .flac)",
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

            gr.Examples(
                examples=[],
                inputs=[emo_audio],
                label="Example audio files (upload your own .wav)",
            )

        # ── Tab 2: Depression Screening ─────────────────────────────────────
        with gr.TabItem("Depression Screening (MODMA)"):
            gr.Markdown(
                """
                ### Depression Screening from Speech
                Upload a `.wav` speech file for binary depression screening (MDD vs Healthy Control).

                **Model:** Whisper-small encoder (1536-D embeddings) + SVM (RBF kernel, C=10)
                **Performance:** Macro-F1 = **0.750** on MODMA test set (Mandarin clinical audio)

                > **Note:** This model was trained on Mandarin clinical interview audio (MODMA dataset).
                > Domain mismatch may reduce accuracy for non-Mandarin or non-clinical speech.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    dep_audio = gr.Audio(
                        label="Upload Speech Audio (.wav / .mp3 / .flac)",
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

        # ── Tab 3: About ─────────────────────────────────────────────────────
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
                Input Audio (.wav)
                        ↓
                OpenAI Whisper-small Encoder
                (244M parameters, frozen)
                        ↓
                Mean + Std Pooling over time
                        ↓
                1536-D Embedding Vector
                        ↓
                ┌──────────────────────────────┐
                │  Tab 1: XGBoost Classifier   │  → 8 Emotions (F1=0.974)
                │  Tab 2: SVM (RBF) Classifier │  → MDD / HC  (F1=0.750)
                └──────────────────────────────┘
                ```

                ### Why Whisper?
                Whisper was pretrained on **680,000 hours** of multilingual speech.
                Its encoder captures deep temporal and prosodic patterns that hand-crafted
                features (MFCC, Mel, Chroma) cannot represent. This is why Whisper+XGBoost
                achieves **F1=0.974** vs F1=0.486 for the best hand-crafted baseline —
                a **+48.8 percentage point improvement**.

                ### Datasets
                | Dataset | Task | Samples | Best F1 |
                |---|---|---|---|
                | RAVDESS | Emotion Recognition (8-class) | 2,452 audio clips | 0.974 |
                | MODMA | Depression Screening (binary) | 52 subjects | 0.750 |
                | DAIC-WOZ | Depression Screening (binary) | 189 sessions | 0.607 |
                | WESAD | Stress Detection (binary) | HRV signals | 0.917 |

                ### Limitations
                - **MODMA model** was trained on Mandarin clinical audio — accuracy may vary on other languages
                - **DAIC-WOZ** depression model (not deployed here) has limited generalizability due to small dataset size (~189 samples)
                - All models are research prototypes and must not be used for clinical decisions
                """
            )

    gr.Markdown(
        """
        ---
        *Built with [Gradio](https://gradio.app) | Powered by [OpenAI Whisper](https://github.com/openai/whisper)*
        """
    )


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
