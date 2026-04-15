# User Guide

## Mental Health Speech Analysis — How to Use the App

**For non-technical users**

---

## What Does This App Do?

This application listens to a short speech sample and uses artificial intelligence to analyze vocal patterns that may indicate emotional or mental health states. Based on acoustic characteristics — such as the pitch, energy, rhythm, and quality of your voice — the system predicts one of the following:

- **Emotional state**: neutral, calm, happy, sad, angry, fearful, disgust, or surprised
- **Depressive indicators**: whether speech patterns are consistent with a depressed or healthy state

> **Important:** This is a research tool built for academic study at IIT Madras. It is **not** a medical diagnostic tool and **cannot** be used to diagnose or treat any mental health condition. If you are concerned about your mental health, please speak to a qualified healthcare professional.

---

## How to Access the App

You do not need to install anything. Simply open your web browser and go to:

**[https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis](https://huggingface.co/spaces/PANKAJ-MOHAN/mental-health-speech-analysis)**

The app works on any modern browser (Chrome, Firefox, Safari, Edge) on desktop or mobile.

---

## Step-by-Step Usage

### Option A — Upload an Audio File

1. On the app page, locate the **Audio Input** box at the top.
2. Click **"Upload file"** or drag and drop an audio file onto the box.
3. Supported formats: **WAV, MP3, FLAC, OGG**
4. The file should be **at least 2 seconds** long; 5–30 seconds works best.
5. Once the file is loaded, you will see a waveform preview and a playback button.
6. Click the **Submit** button.
7. Wait a few seconds. The result will appear in the **Output** panel on the right.

### Option B — Record Directly from Microphone

1. On the app page, click the **microphone icon** or the **"Record from microphone"** button.
2. Your browser may ask for microphone permission — click **Allow**.
3. Speak naturally for 5–30 seconds. You can say anything: read a passage, describe your day, or just talk.
4. Click **Stop Recording** when finished.
5. The recording will appear in the audio box with a waveform preview.
6. Click **Submit**.
7. Wait a few seconds. The result will appear in the **Output** panel.

---

## What Do the Results Mean?

The app returns two pieces of information:

### 1. Predicted Label

This is the detected emotional or mental health state. Examples:

| Label | Meaning |
|---|---|
| **Neutral** | Flat, unexpressive speech with no strong emotion |
| **Calm** | Low-energy, steady speech — relaxed state |
| **Happy** | High-energy, upbeat vocal patterns |
| **Sad** | Low pitch, slow rate, reduced energy |
| **Angry** | High energy, tense vocal quality |
| **Fearful** | Irregular pitch, faster rate, tension |
| **Disgust** | Strong negative vocal expression |
| **Surprised** | Sudden pitch changes, exclamatory patterns |
| **Depressed** | Reduced energy, monotone pitch, slower speech rate |
| **Healthy** | Normal speech energy and variability |

### 2. Confidence Score

This number (between 0% and 100%) indicates how certain the model is about its prediction:

- **80–100%**: High confidence — the speech strongly matches this pattern
- **60–80%**: Moderate confidence — a plausible match with some uncertainty
- **Below 60%**: Low confidence — treat the result with caution; the audio may be unclear, too short, or noisy

---

## Example Scenarios

### Scenario 1: Checking an audio clip from a patient interview
- Upload a WAV recording of a clinical interview.
- The app analyzes the acoustic patterns and returns "Depressed" or "Healthy" with a confidence score.
- Useful for researchers studying automated depression screening.

### Scenario 2: Emotion recognition research
- Upload RAVDESS-format acted speech clips (e.g., an actor expressing anger).
- The app returns one of 8 emotion labels.
- Useful for testing the model's emotion detection accuracy.

### Scenario 3: Live demonstration
- Use the microphone recording feature to demonstrate the system in real time.
- Speak in different emotional tones and observe how the predicted label changes.

---

## Tips for Best Results

- **Speak clearly** and directly toward the microphone.
- **Avoid background noise** — a quiet room gives more accurate results.
- **Speak for at least 5 seconds** — very short recordings reduce accuracy.
- **Avoid music or sounds** — the model is trained on speech only.
- **Use WAV format** when possible — it provides the best audio quality without compression artifacts.
- **One speaker at a time** — the model is designed for single-speaker input.

---

## Troubleshooting

### The app is not loading / shows a blank page

The Hugging Face Space may be in "sleep" mode after a period of inactivity. This is normal for free-tier hosting.

**Fix:** Wait 30–60 seconds and refresh the page. The Space will wake up and load automatically.

### The prediction is taking a long time

The first prediction after the Space wakes up may take 15–30 seconds as the model is loaded into memory.

**Fix:** Be patient — subsequent predictions on the same session will be much faster.

### "Error: Audio too short"

The uploaded audio is less than 0.5 seconds.

**Fix:** Ensure your recording or file is at least 2 seconds long.

### "Error: Audio appears silent"

The audio file has no detectable sound.

**Fix:** Check your microphone volume, ensure the microphone is not muted, or verify that the uploaded file contains actual audio.

### The browser asks for microphone permission but I cannot allow it

Some browsers block microphone access on certain websites or in private/incognito mode.

**Fix:** Use a regular (non-private) browser tab, or switch to uploading a pre-recorded file instead.

### I uploaded a file but the waveform looks wrong

Very large files (over 50 MB) or unusual sample rates may cause display issues.

**Fix:** Convert your audio to WAV at 16,000 Hz or 22,050 Hz mono before uploading. Tools like Audacity (free) can do this conversion.

---

## Privacy Notice

Audio submitted through this app is used only for the current prediction and is not stored, logged, or shared beyond the duration of the inference request. The application does not collect personal data. Do not submit audio containing sensitive personal, medical, or confidential information through this research prototype.

---

## Contact and Feedback

This application was developed by Group 6 as part of the BSDA4001 Data Science and AI Lab Project at IIT Madras.

For technical issues or research inquiries, contact the team through the course portal.
