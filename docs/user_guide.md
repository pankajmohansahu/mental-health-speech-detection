# User Guide

## Mental Health Speech Analysis — How to Use the App

**For non-technical users**

---

## What Does This App Do?

This application listens to a speech sample and uses artificial intelligence to analyse vocal patterns that may indicate emotional or mental health states. It provides five tools in a single interface:

1. **Emotion Recognition** — detects one of 8 emotions from speech
2. **Depression Screening** — screens speech for depression indicators
3. **PHQ-8 Self-Screener** — a validated clinical questionnaire you fill in yourself
4. **AI Clinical Explanation** — an AI-generated explanation of your results grounded in clinical research
5. **About & Model Details** — project background and technical information

> **Important:** This is a research tool built for academic study at IIT Madras. It is **not** a medical diagnostic tool and **cannot** be used to diagnose or treat any mental health condition. If you are concerned about your mental health, please speak to a qualified healthcare professional.

---

## How to Access the App

You do not need to install anything. Open your web browser and go to:

**[https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis](https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis)**

The app works on any modern browser (Chrome, Firefox, Safari, Edge) on desktop or mobile.

---

## Tab 1 — Emotion Recognition

### What it does
Analyses your speech and classifies it into one of 8 emotional states: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, or Surprised.

**Model:** Whisper-small encoder + XGBoost — Test F1 = 0.974

### How to use

**Option A — Upload a file**
1. Click **"Upload file"** or drag and drop a WAV / MP3 / FLAC file onto the audio box.
2. The file should be at least 2 seconds long; 5–30 seconds works best.
3. Click **"Detect Emotion"**.
4. Results appear on the right: the predicted emotion label, confidence percentage, and a bar chart showing confidence for all 8 classes.

**Option B — Record from microphone**
1. Click the microphone icon in the audio box.
2. Allow microphone access when prompted by your browser.
3. Speak naturally for 5–30 seconds.
4. Click **Stop**.
5. Click **"Detect Emotion"**.

### Understanding the results

| Label | What it means |
|---|---|
| **Neutral** | Flat, unexpressive speech |
| **Calm** | Low-energy, steady, relaxed speech |
| **Happy** | High-energy, upbeat vocal patterns |
| **Sad** | Low pitch, slow rate, reduced energy |
| **Angry** | High energy, tense vocal quality |
| **Fearful** | Irregular pitch, faster rate, tension |
| **Disgust** | Strong negative vocal expression |
| **Surprised** | Sudden pitch changes, exclamatory patterns |

The **confidence bar chart** shows what the model thinks across all 8 emotions. A single tall bar means high certainty; several similar bars mean the model is unsure.

---

## Tab 2 — Depression Screening

### What it does
Screens your speech for acoustic patterns associated with depression (MDD) versus a healthy control (HC). Returns a binary result with probability scores for both classes.

**Model:** Whisper-small encoder + SVM (RBF kernel) — trained on MODMA clinical audio. Test F1 = 0.750.

> This model was trained on Mandarin clinical recordings. Results on other languages may vary.

### How to use
1. Upload a speech audio file or record from microphone (same steps as Tab 1).
2. Click **"Screen for Depression"**.
3. Results show: screening label (HC or MDD), confidence, and a bar chart of P(Healthy) vs P(Depressed).

### Understanding the results

- **HC — No Depression Indicators**: Speech patterns are consistent with healthy controls in the training data.
- **MDD — Possible Depression Indicators**: Speech patterns show similarities to clinically diagnosed MDD patients.

> A single speech recording cannot diagnose depression. This tool is a research prototype and must not be used for clinical decisions.

---

## Tab 3 — PHQ-8 Self-Screener

### What it does
The **PHQ-8 (Patient Health Questionnaire-8)** is a validated clinical screening tool used worldwide by healthcare professionals. It asks 8 questions about how you have been feeling over the past two weeks.

This tab has no audio input — you answer the questions directly.

### How to use
1. Read each of the 8 questions carefully.
2. For each question, select how often you have been bothered by that problem over the **last 2 weeks**:
   - Not at all (0)
   - Several days (1)
   - More than half the days (2)
   - Nearly every day (3)
3. Answer all 8 questions.
4. Click **"Calculate PHQ-8 Score"**.
5. Your total score (0–24), severity band, and a score gauge chart will appear.

### Score interpretation

| Score | Severity | Recommendation |
|---|---|---|
| 0–4 | No significant depression | No action needed |
| 5–9 | Mild depression | Monitor symptoms over coming weeks |
| 10–14 | Moderate depression | Consider speaking with a counsellor or GP |
| 15–19 | Moderately severe depression | Consult a mental health professional |
| 20–24 | Severe depression | Seek professional help as soon as possible |

**Clinical threshold:** A score of **10 or above** is the standard cut-off used by clinicians to flag significant depression.

> The PHQ-8 has 88% sensitivity and 88% specificity for major depressive disorder in clinical validation studies. It is a screening tool — not a diagnosis.

---

## Tab 4 — AI Clinical Explanation (RAG)

### What it does
This tab combines your audio analysis with a clinical knowledge base and an AI language model to generate a personalised, research-grounded explanation of your results. It is powered by:
- **FAISS vector search** over 15 curated clinical knowledge chunks
- **sentence-transformers** (all-MiniLM-L6-v2) for semantic retrieval
- **Groq LLaMA3-70B** for natural language generation

### How to use
1. Upload a speech audio file or record from microphone.
2. Optionally, enter your **PHQ-8 score** from Tab 3 (type the number in the box).
3. Click **"Generate AI Explanation"**.
4. A 4-paragraph explanation appears covering:
   - **What the results show** — what the AI detected and its clinical meaning
   - **Speech patterns** — which acoustic features may have contributed
   - **Context and perspective** — limitations of AI screening, how common these patterns are
   - **Next steps** — recommendation to consult a professional, India helplines included

### Important notes
- The explanation is educational only. It is grounded in published clinical research but is **not a clinical diagnosis**.
- If the RAG module is unavailable (missing API key or dependencies), an error message will appear.
- The Groq API key is required for LLM generation. It is pre-configured for the Hugging Face Space deployment.

### Helplines included in explanations
- **iCall:** 9152987821 (Mon–Sat, 8am–10pm)
- **Vandrevala Foundation:** 1860-2662-345 (24/7, free)
- **AASRA:** 9820466627 (24/7)

---

## Tab 5 — About & Model Details

Contains project background, team information, pipeline architecture, dataset summary, and model performance metrics. No interaction required.

---

## Tips for Best Results (Audio Tabs)

- **Speak clearly** and directly toward the microphone.
- **Avoid background noise** — a quiet room gives more accurate results.
- **Speak for at least 5 seconds** — very short recordings reduce accuracy.
- **Avoid music or sounds** — models are trained on speech only.
- **Use WAV format** when possible — no compression artifacts.
- **One speaker at a time** — models are designed for single-speaker input.

---

## Troubleshooting

### The app is not loading / shows a blank page
The Hugging Face Space may be in sleep mode after a period of inactivity.

**Fix:** Wait 30–60 seconds and refresh the page. The Space wakes up automatically.

### The prediction is taking a long time
The first prediction after the Space wakes up may take 15–30 seconds as Whisper loads into memory.

**Fix:** Be patient — subsequent predictions in the same session will be much faster.

### "Error: Audio too short"
The uploaded audio is less than 0.5 seconds.

**Fix:** Ensure your recording or file is at least 2 seconds long.

### "Error: Audio appears silent"
The audio file has no detectable sound.

**Fix:** Check microphone volume, ensure the microphone is not muted, or verify the uploaded file contains actual audio.

### The RAG explanation says "RAG module not available"
The RAG dependencies (faiss-cpu, sentence-transformers, groq) are not installed or the Groq API key is missing.

**Fix:** Ensure all dependencies in `requirements.txt` are installed and `GROQ_API_KEY` is set as an environment variable or in `config.py`.

### The browser asks for microphone permission but I cannot allow it
Some browsers block microphone access in private/incognito mode.

**Fix:** Use a regular (non-private) browser tab, or upload a pre-recorded file instead.

### I uploaded a file but the waveform looks wrong
Very large files (over 50 MB) or unusual sample rates may cause display issues.

**Fix:** Convert audio to WAV at 16,000 Hz mono before uploading. Audacity (free) can do this.

---

## Privacy Notice

Audio submitted through this app is used only for the current prediction and is not stored, logged, or shared beyond the duration of the inference request. The application does not collect personal data. Do not submit audio containing sensitive personal, medical, or confidential information through this research prototype.

---

## Contact and Feedback

This application was developed by Group 6 as part of the BSDA4001 Data Science and AI Lab Project at IIT Madras.

For technical issues or research inquiries, contact the team through the course portal.
