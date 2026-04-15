"""
rag_utils.py — RAG knowledge base, FAISS retrieval, and Groq LLM generation
AI-Based Early Mental Health Breakdown Detection from Speech Patterns
Group 6 | IIT Madras — BSDA4001
"""

import numpy as np
from config import GROQ_API_KEY, GROQ_MODEL

# ── Knowledge Base ─────────────────────────────────────────────────────────────
KNOWLEDGE_CHUNKS = [
    (
        "depression_speech_1",
        "Research studies have identified key acoustic speech markers associated with Major Depressive "
        "Disorder (MDD). These include: reduced speech rate and longer pauses, lower pitch (fundamental "
        "frequency F0) and reduced pitch variability, decreased speech energy and volume, monotonous "
        "prosody, increased vowel space reduction, and longer response latency. A 2019 meta-analysis "
        "found that depressed individuals show significantly reduced vocal activity and flattened "
        "prosodic contours compared to healthy controls.",
    ),
    (
        "depression_speech_2",
        "MFCC (Mel-Frequency Cepstral Coefficients) features capture the spectral envelope of speech, "
        "which changes in depression. Depressed speech typically shows lower MFCC variance, reduced "
        "spectral brightness, and compressed dynamic range. Pitch features (F0 mean, F0 range) are "
        "consistently lower in MDD patients compared to healthy controls across multiple languages and "
        "cultures.",
    ),
    (
        "who_depression_guidelines",
        "According to WHO guidelines, depression affects over 280 million people worldwide. Clinical "
        "depression (MDD) is characterized by persistent sadness, loss of interest, and at least 5 "
        "symptoms from: depressed mood, loss of pleasure, weight change, sleep disturbance, psychomotor "
        "changes, fatigue, worthlessness, concentration difficulties, and suicidal ideation. Speech-based "
        "screening can serve as a non-invasive early indicator but must always be followed by a thorough "
        "clinical assessment by a qualified professional.",
    ),
    (
        "phq8_interpretation",
        "The PHQ-8 (Patient Health Questionnaire-8) is a validated clinical tool for depression "
        "screening. Score ranges: 0–4 = No significant depression; 5–9 = Mild depression; 10–14 = "
        "Moderate depression; 15–19 = Moderately severe depression; 20–24 = Severe depression. A score "
        "of 10 or above is the standard clinical threshold for major depression. The PHQ-8 has a "
        "sensitivity of 88% and specificity of 88% for major depressive disorder.",
    ),
    (
        "phq8_questions",
        "The PHQ-8 assesses 8 symptoms over the past two weeks: (1) Little interest or pleasure in "
        "doing things; (2) Feeling down, depressed, or hopeless; (3) Trouble falling or staying asleep, "
        "or sleeping too much; (4) Feeling tired or having little energy; (5) Poor appetite or "
        "overeating; (6) Feeling bad about yourself or that you are a failure; (7) Trouble concentrating "
        "on things; (8) Moving or speaking so slowly that others notice, or being fidgety and restless. "
        "Each item is scored 0 (not at all) to 3 (nearly every day).",
    ),
    (
        "emotion_mental_health",
        "Emotional states detected through speech are closely linked to mental health conditions. "
        "Persistent sadness or flat affect (Neutral or Calm with low energy) may indicate depression. "
        "Chronic fear or anxiety patterns in speech correlate with anxiety disorders. Inappropriate "
        "emotional expression (sudden angry or disgust outbursts) may relate to mood disorders. "
        "Emotion recognition from speech has been used in clinical settings as a supplementary "
        "diagnostic marker alongside standardized questionnaires.",
    ),
    (
        "ravdess_emotions",
        "The RAVDESS dataset contains 8 emotional categories: Neutral, Calm, Happy, Sad, Angry, "
        "Fearful, Disgust, and Surprised. In mental health contexts: Sad emotion corresponds to "
        "depressive affect; Fearful corresponds to anxiety-related states; Angry may indicate mood "
        "dysregulation; Neutral or Calm with reduced prosodic energy may indicate emotional blunting "
        "seen in depression; Happy indicates positive affect and emotional resilience.",
    ),
    (
        "whisper_features",
        "OpenAI Whisper's encoder, pretrained on 680,000 hours of multilingual speech, captures deep "
        "temporal and prosodic representations. The 1536-dimensional embedding (mean+std pooling over "
        "encoder output) encodes speaking rate, pause patterns, pitch variation, voice quality, "
        "articulation clarity, and emotional prosody. These representations are far richer than "
        "hand-crafted MFCC features, explaining why Whisper-based models achieve significantly higher "
        "accuracy in emotion and depression detection tasks.",
    ),
    (
        "mdd_vs_hc",
        "In binary depression classification (MDD vs Healthy Control), key differentiating speech "
        "features are: pause duration (longer in MDD), speech rate (slower in MDD), fundamental "
        "frequency variability (lower in MDD), jitter and shimmer (higher in MDD indicating vocal "
        "instability), and energy/loudness (lower in MDD). The MODMA dataset used in this study "
        "contains clinical interviews in Mandarin from diagnosed MDD patients and age-matched healthy "
        "controls confirmed by DSM-5 criteria.",
    ),
    (
        "india_help_resources",
        "If you or someone you know is experiencing signs of depression or emotional distress, please "
        "seek professional help immediately. Helplines in India: iCall — 9152987821 (Mon–Sat, 8am–10pm); "
        "Vandrevala Foundation — 1860-2662-345 (24/7, free); AASRA — 9820466627 (24/7); iCall "
        "WhatsApp — 9152987821. This AI screening tool is NOT a substitute for professional medical "
        "diagnosis. Always consult a qualified psychiatrist or psychologist for clinical evaluation.",
    ),
    (
        "depression_prevalence_india",
        "Depression is a leading cause of disability worldwide, affecting approximately 5% of adults "
        "globally. In India, approximately 56 million people suffer from depression, yet only 10–20% "
        "receive adequate treatment due to stigma and limited access to care. Early detection is "
        "critical. AI-based speech analysis offers a scalable, non-invasive screening approach that "
        "could help bridge the mental health treatment gap, especially in resource-limited settings.",
    ),
    (
        "treatment_approaches",
        "Evidence-based treatments for depression include: Cognitive Behavioral Therapy (CBT) — highly "
        "effective for mild to moderate depression; SSRI/SNRI antidepressants — effective for moderate "
        "to severe cases; Interpersonal Therapy (IPT); Mindfulness-based cognitive therapy (MBCT); "
        "Regular aerobic exercise — shown equivalent to antidepressants for mild depression; Combined "
        "therapy and medication for severe cases. Early intervention significantly improves outcomes "
        "and reduces risk of recurrence.",
    ),
    (
        "acoustic_clinical_markers",
        "Clinical acoustic analysis in mental health assessment covers: Prosody (rhythm, stress, "
        "intonation) — flattened in depression; Voice quality (breathiness, harshness) — increased "
        "breathiness in MDD; Temporal features (speech rate, pause ratio) — slowed in depression; "
        "Spectral features (formant frequencies, spectral tilt) — altered in MDD. These markers are "
        "consistent across multiple international studies and form the scientific basis of computational "
        "depression detection systems.",
    ),
    (
        "ai_screening_limitations",
        "AI-based mental health screening tools have important limitations: Training data bias (models "
        "trained on specific populations may not generalize to all groups); Language and cultural "
        "differences affect speech prosody; Recording noise and microphone quality affect predictions; "
        "These tools cannot assess suicidality, psychosis, or complex psychiatric conditions. They "
        "should be used only as a first-line screening aid, never as a standalone diagnostic tool. "
        "Always combine AI screening output with clinical judgment from a trained professional.",
    ),
    (
        "sad_emotion_clinical",
        "Sadness detected in speech is associated with reduced pitch, slower tempo, lower intensity, "
        "and longer pauses. Clinically, persistent sadness is a core symptom of MDD. When AI detects "
        "sad affect in speech with high confidence, it may reflect transient situational sadness OR "
        "persistent depressive affect — context and frequency matter. A single recording cannot "
        "distinguish between these. PHQ-8 self-report and clinical interview are essential for "
        "differential diagnosis.",
    ),
    (
        "fearful_anxiety_clinical",
        "Fear and anxiety detected in speech manifest as elevated pitch, increased speech rate, voice "
        "tremor, and irregular breathing patterns captured in acoustic features. Clinically, these "
        "patterns may indicate Generalized Anxiety Disorder (GAD), social anxiety, panic disorder, or "
        "PTSD. Anxiety and depression frequently co-occur (comorbidity rate ~50%). If speech-based "
        "analysis detects fearful affect alongside depression indicators, evaluation for comorbid "
        "anxiety disorder is recommended.",
    ),
]

# ── FAISS Index (built once, reused) ──────────────────────────────────────────
_encoder  = None
_index    = None
_texts    = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        print("[RAG] Loading sentence-transformer encoder...")
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[RAG] Encoder loaded.")
    return _encoder


def _build_index():
    global _index, _texts
    import faiss
    enc = _get_encoder()
    _texts = [chunk[1] for chunk in KNOWLEDGE_CHUNKS]
    embeddings = enc.encode(_texts, show_progress_bar=False).astype("float32")
    dim = embeddings.shape[1]
    _index = faiss.IndexFlatL2(dim)
    _index.add(embeddings)
    print(f"[RAG] FAISS index built — {len(_texts)} chunks, dim={dim}.")


def retrieve(query: str, k: int = 4) -> list:
    """Return top-k relevant knowledge chunks for a query."""
    global _index, _texts
    if _index is None:
        _build_index()
    enc = _get_encoder()
    q_emb = enc.encode([query]).astype("float32")
    _, indices = _index.search(q_emb, k)
    return [_texts[i] for i in indices[0]]


# ── Groq LLM generation ───────────────────────────────────────────────────────
def generate_explanation(
    emotion_result: dict | None,
    depression_result: dict | None,
    phq8_score: int | None = None,
) -> str:
    """
    Generate a clinically-grounded explanation using RAG + Groq LLM.

    Args:
        emotion_result:    dict from predict_emotion(), or None
        depression_result: dict from predict_depression(), or None
        phq8_score:        PHQ-8 total score (0-24), or None if not provided
    Returns:
        Plain-language explanation string
    """
    from groq import Groq

    # Build retrieval query from results
    query_parts = []
    if emotion_result:
        query_parts.append(f"{emotion_result['label']} emotion speech markers mental health depression")
    if depression_result:
        if "MDD" in depression_result["label"]:
            query_parts.append("MDD depression speech acoustic markers clinical indicators")
        else:
            query_parts.append("healthy control speech mental health no depression")
    if phq8_score is not None:
        query_parts.append("PHQ-8 score depression severity clinical threshold")

    query = " ".join(query_parts) if query_parts else "mental health speech analysis depression"
    context_chunks = retrieve(query, k=4)
    context = "\n\n---\n\n".join(context_chunks)

    # Format result strings
    emotion_str = (
        f"Detected emotion: **{emotion_result['label']}** "
        f"(confidence {emotion_result['confidence']*100:.1f}%)"
        if emotion_result else "Emotion analysis: not performed"
    )
    dep_str = (
        f"Depression screening: **{depression_result['label']}** "
        f"(P(MDD)={depression_result['prob_mdd']*100:.1f}%, "
        f"P(HC)={depression_result['prob_hc']*100:.1f}%)"
        if depression_result else "Depression screening: not performed"
    )
    phq8_str = (
        f"PHQ-8 self-report score: **{phq8_score}/24**"
        if phq8_score is not None else "PHQ-8 self-report: not provided"
    )

    prompt = f"""You are a clinical AI assistant providing educational explanations of speech-based mental health screening results.
Your role is to explain results clearly, ground them in clinical evidence, and guide users toward professional help.
You must NOT diagnose, prescribe, or make definitive clinical judgments.

## Screening Results
- {emotion_str}
- {dep_str}
- {phq8_str}

## Retrieved Clinical Knowledge
{context}

## Your Task
Write a supportive, factual explanation in 4 short paragraphs:

**Paragraph 1 — What the results show:** Explain what the AI detected and what it may suggest based on clinical knowledge.

**Paragraph 2 — Speech patterns:** Describe what acoustic/speech patterns may have contributed to these results (pitch, energy, pace, prosody etc.).

**Paragraph 3 — Context and perspective:** Provide context — how common these patterns are, what they may or may not mean, and the limitations of AI screening.

**Paragraph 4 — Next steps:** Recommend consulting a mental health professional. Include at least one India helpline (iCall: 9152987821 or Vandrevala Foundation: 1860-2662-345).

Keep the tone warm, non-alarmist, and empowering. Use plain language. Maximum 350 words total."""

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content
