## **AI-Based Early Mental Health Breakdown Detection from Speech Patterns**


Data Science and AI Lab Project (BSDA4001)


**Milestone 2 Report -- Dataset Preparation**


*Submitted By:*
Group 6


Om Aryan (21f3002286)
Pankaj Mohan Sahu (21f2001203)
Drashti Shah (22f2001483)
Mahi Mudgal (21f3002602)
G Hamsini (22f3000767)


---


## Contents


1. Introduction
2. Data Sources
3. Dataset Description
4. Dataset Quality Assessment
5. Dataset Adequacy for the Problem
6. Training / Validation / Test Split Strategy
7. Alignment Between Speech and Transcript Data
8. Dataset Requirements for Model Training
9. Speech Dataset Requirements
10. LLM Fine-Tuning Dataset Requirements
11. RAG Dataset Preparation
12. Preprocessing Pipeline
13. Reproducibility


---


## **1. Introduction**


Early detection of mental health deterioration remains a critical challenge in clinical practice. Traditional screening relies heavily on self-reporting instruments such as the PHQ-8 questionnaire, which are susceptible to subjective bias, recall errors, and social desirability effects. This project develops a multimodal AI screening system that detects early signs of mental health breakdown from speech and conversational patterns, providing an objective, non-invasive complement to existing clinical assessment tools.


The system architecture implements a three-component pipeline:


1. **Speech Analysis Pipeline.** Raw audio recordings from clinical interviews are processed to extract acoustic features that encode paralinguistic markers of depression, including prosodic flattening, atypical pause patterns, reduced energy envelopes, and speech rate abnormalities. The audio branch uses a frozen OpenAI Whisper encoder to extract fixed-dimensional audio embeddings, preserving depression-relevant vocal biomarkers without task-specific feature engineering.


2. **Transcript-Based Analysis.** Interview audio is accompanied by time-stamped transcripts that capture the linguistic content of clinical conversations. These transcripts are processed through a frozen BioClinical-ModernBERT encoder, which leverages an extended 8,192-token context window to capture long-range semantic dependencies, PHQ-style symptom descriptions, negative self-referential language, and affective shifts across the full dialogue.


3. **RAG-Based Reasoning System.** A Retrieval-Augmented Generation (RAG) module operates as a post-hoc interpretability layer. A sentence-embedding model (MiniLM) retrieves relevant clinical guideline snippets from a curated knowledge base of APA and WHO standards. An instruction-tuned LLM then synthesises extracted acoustic and linguistic signals into a constrained, non-diagnostic clinical summary that cites retrieved guidelines for traceability.


These three components are unified through a Gated Dual-Modal Fusion architecture. A sigmoid-based gating mechanism adaptively weights the contributions of the audio and text branches - prioritising linguistic content when audio quality is poor and acoustic paralinguistics when the speaker is laconic. The fused representation feeds a compact classifier head trained with Weighted Cross-Entropy Loss to address inherent class imbalance and ensure high screening sensitivity.


This Milestone 2 report documents the complete dataset preparation pipeline: data source identification and verification, exploratory data analysis, quality assessment, split strategy, cross-modal alignment, and preprocessing - establishing the data foundation for model training in subsequent milestones.


---


## **2. Data Sources**


Three primary datasets and one planned dataset support the multimodal pipeline. Each serves a distinct role in the architecture.


### 2.1 Speech Datasets


#### DAIC-WOZ (Primary Clinical Benchmark)


| Item | Detail |
|---|---|
| **Full Name** | Distress Analysis Interview Corpus -- Wizard of Oz |
| **Source** | University of Southern California (USC) Institute for Creative Technologies |
| **Ownership** | USC ICT; distributed under the AVEC 2017 Depression Sub-Challenge |
| **Access** | Requires signed Data Use Agreement (DUA); application submitted, approval pending |
| **License** | Research-only; no redistribution; no commercial use |
| **Citation** | Gratch et al. (2014), LREC; DeVault et al. (2014); AVEC 2017 Challenge |
| **Format** | Audio (16 kHz mono WAV), transcripts (tab-separated CSV with per-utterance timestamps), pre-extracted visual features (CLNF/OpenFace) |
| **IRB Status** | Original data collection IRB-approved; all participant IDs are de-identified numeric codes |
| **PII** | None -- no personally identifiable information |
| **Reliability** | Gold-standard benchmark; used in 200+ published studies; PHQ-8 labels validated by licensed clinicians |
| **Approval Status** | DUA submitted to USC ICT; awaiting approval. The feature extraction pipeline has been built and validated using publicly available Kaggle dataset. |


**Role in pipeline:** The only dataset providing synchronized transcript-audio pairs with clinical PHQ-8 depression labels. Central to training the Gated Dual-Modal Fusion model.


#### MODMA (Supplementary Clinical Audio Dataset)


| Item | Detail |
|---|---|
| **Full Name** | Multi-modal Open Dataset for Mental-disorder Analysis |
| **Source** | Institute of Automation, Chinese Academy of Sciences |
| **Ownership** | Institute of Automation, Chinese Academy of Sciences |
| **Access** | Access granted after submitting and signing the End User License Agreement (EULA); application approved |
| **License** | Research-only; redistribution prohibited; no commercial use without permission |
| **Citation** | Yang et al. (2020), *MODMA: A Multimodal Dataset for Depression Detection* |
| **Format** | Audio recordings (WAV format) collected from structured speech tasks and interviews; accompanying metadata including clinical diagnosis labels |
| **IRB Status** | Original data collection approved by the institutional ethics committee; participants provided informed consent |
| **PII** | None — all participant identities anonymized using numeric identifiers |
| **Reliability** | Clinically validated dataset used in multiple speech-based depression detection studies |
| **Approval Status** | EULA submitted and approved; dataset access granted |


**Role in pipeline:** Used as a supplementary audio dataset to enhance the robustness of speech feature extraction and support evaluation of the depression detection model on an additional clinically labeled speech corpus.


#### RAVDESS (Acoustic Emotion Baseline)


| Item | Detail |
|---|---|
| **Full Name** | Ryerson Audio-Visual Database of Emotional Speech and Song |
| **Source** | Zenodo / Kaggle (archived from Ryerson University) |
| **Ownership** | Ryerson University; created by Livingstone & Russo |
| **License** | CC BY-NC-SA 4.0 -- free for non-commercial research with attribution |
| **Citation** | Livingstone & Russo (2018), PLoS ONE 13(5) |
| **Format** | 48 kHz, 16-bit, mono WAV files |
| **IRB Status** | Professional actors; no clinical subjects |
| **PII** | None |
| **Reliability** | Peer-reviewed publication; studio-recorded under controlled acoustic conditions; widely used in emotion recognition research |
| **Approval Requirement** | None. Publicly available under CC BY-NC-SA 4.0. Only proper citation is required. |


**Role in pipeline:** Emotion pre-training dataset. The classifier head learns basic prosodic discrimination across 8 emotion classes before being fine-tuned on clinical depression markers from DAIC-WOZ. Depression-relevant emotions (sad, fearful, disgust) serve as depression-risk proxies during transfer learning.


### 2.2 Transcript and Clinical Datasets


#### DAIC-WOZ Transcripts


DAIC-WOZ transcripts are embedded within the primary dataset described above. Each session includes a tab-separated transcript file with columns: `start_time`, `stop_time`, `speaker` (Participant or Interviewer), and `value` (utterance text). These transcripts are manually verified and time-aligned to the corresponding audio, enabling precise audio-text pairing for the Gated Fusion model.


#### MODMA Dataset
The MODMA dataset does not natively provide speech transcripts for its audio recordings. To address this, our pipeline introduces an additional **automatic transcription sub-module** that converts the speech recordings into text using a speech-to-text model. The generated transcripts are time-aligned with the corresponding audio segments and stored in structured tabular format. This enables the dataset to be incorporated into the **audio–text multimodal processing framework**, allowing consistent feature extraction and fusion with the DAIC-WOZ transcript-audio pairs.


### 2.3 Knowledge Base Datasets for RAG (Planned)


| Item | Detail |
|---|---|
| **Document Types** | APA Practice Guidelines, WHO mhGAP Intervention Guide, PHQ-8/PHQ-9 scoring manuals, DSM-5 diagnostic criteria for Major Depressive Disorder |
| **Purpose** | Curated knowledge base for the Retrieval-Augmented Reasoning Layer |
| **Current Status** | Not yet assembled; planned for Milestone 4-5 (RAG Integration phase) |
| **Format** | PDF/text documents to be chunked, embedded, and indexed in a vector database |
| **Rationale for Deferral** | The RAG module is a post-hoc inference component. Building the knowledge base is meaningful only after the core fusion model is trained. |


### 2.4 Contextual Stress Dataset (Planned - Custom Collection)


| Item | Detail |
|---|---|
| **Described In** | Milestone 1 Report, Section 5 |
| **Planned Scope** | ~500 samples of high-stress (exam periods) vs. baseline speech |
| **Estimated Size** | 1.30 GB |
| **Purpose** | Testing the Signal Extraction and RAG-Grounded Reasoning Layer with non-clinical, high-pressure speech patterns |
| **Current Status** | Not yet collected; deferred to Milestone 4-5 |
| **Rationale** | Collection protocol will be defined alongside RAG knowledge-base setup, as this dataset tests post-hoc reasoning outputs rather than core model training |


---


## **3. Dataset Description**


### 3.1 Speech Dataset -- DAIC-WOZ


| Attribute | Value |
|---|---|
| **Total recordings** | 189 clinical interview sessions |
| **Total audio duration** | 50+ hours |
| **Dataset size** | 126 GB (raw and processed) |
| **Audio format** | 16 kHz, mono WAV |
| **Sampling rate** | 16,000 Hz |
| **Number of speakers** | 189 unique participants (one session per participant) |
| **Interviewer** | Animated virtual agent ("Ellie") |
| **Total extracted features** | 448 per participant |
| **Feature modalities** | COVAREP acoustics (300), Formants (23), Facial Action Units (36), Head Pose (26), Eye Gaze (49), Transcript NLP (14) |
| **Labels** | PHQ-8 Binary (0/1 at threshold >= 10), PHQ-8 Score (0-24 continuous) |


**Class distribution (labelled participants, N=142):**


| Class | Count | Percentage |
|---|---|---|
| Healthy (PHQ8_Binary = 0) | 100 | 70.4% |
| Depressed (PHQ8_Binary = 1) | 42 | 29.6% |


Class imbalance ratio: 2.4:1 (healthy : depressed).


**PHQ-8 severity band distribution (N=142):**


| Severity Band | PHQ-8 Range | Count | Percentage |
|---|---|---|---|
| Minimal | 0-4 | 64 | 45.1% |
| Mild | 5-9 | 35 | 24.6% |
| Moderate | 10-14 | 25 | 17.6% |
| Moderately Severe | 15-19 | 13 | 9.2% |
| Severe | 20-24 | 5 | 3.5% |


**PHQ-8 score statistics:**


| Statistic | Value |
|---|---|
| Mean | 6.67 |
| Standard deviation | 5.73 |
| Minimum | 0 |
| Maximum | 23 |


**Gender distribution (labelled, N=142):**


| Gender | Healthy | Depressed | Total |
|---|---|---|---|
| Female | 39 | 24 | 63 |
| Male | 61 | 18 | 79 |


### 3.2 Speech Dataset -- RAVDESS


| Attribute | Value |
|---|---|
| **Total recordings** | 1,440 speech-only audio clips |
| **Dataset size** | 1.09 GB |
| **Audio format** | 48 kHz, 16-bit, mono WAV (studio-recorded) |
| **Native sampling rate** | 48,000 Hz |
| **Resampled to** | 22,050 Hz for feature extraction |
| **Number of speakers** | 24 professional actors (12 male, 12 female) |
| **Emotions** | 8 classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| **Intensity levels** | Normal and Strong (neutral has normal only) |
| **Statements** | 2 scripted statements ("Kids are talking by the door" / "Dogs are sitting by the door") |
| **Total extracted features** | 338 per audio clip |


**Audio duration statistics:**


| Statistic | Value |
|---|---|
| Mean | 3.70 s |
| Std | 0.34 s |
| Min | 2.94 s |
| Max | 5.27 s |
| Median | 3.67 s |


**Class distribution (perfectly balanced):**


| Emotion | File Count |
|---|---|
| Neutral | 96 |
| Calm | 192 |
| Happy | 192 |
| Sad | 192 |
| Angry | 192 |
| Fearful | 192 |
| Disgust | 192 |
| Surprised | 192 |


Neutral has fewer files because it has no "strong intensity" variant.


### 3.3 MODMA Audio Dataset


| Attribute | Value |
|---|---|
| **Total subjects** | 52 |
| **Dataset size** | 2.52 GB |
| **Class distribution** | 23 MDD (44.2%), 29 HC (55.8%) |
| **Gender** | 36 Male, 16 Female |
| **Speech tasks** | Structured vocal tasks including interview responses, passage reading, vocabulary reading, and picture description |
| **Audio format** | WAV |
| **Sampling rate** | 44.1 kHz |
| **Bit depth** | 24-bit |
| **Samples per subject** | ~29 speech recordings per participant |
| **Transcript availability** | Not provided in the original dataset |
| **Transcript generation** | Generated using an automatic speech-to-text module in the preprocessing pipeline |


The MODMA speech dataset consists of recordings collected from clinically diagnosed Major Depressive Disorder (MDD) patients and healthy control (HC) participants. Speech data were recorded under controlled conditions using multiple vocal tasks designed to elicit both spontaneous and structured speech. These recordings are widely used for **speech-based depression detection research**.


Since the dataset does not include textual transcripts, our pipeline introduces an **automatic transcription module** that converts each speech recording into text using a speech recognition model. The generated transcripts enable extraction of **linguistic features**, allowing MODMA speech data to be integrated into the **audio–text multimodal processing framework** used in this study.


---


### Extracted Speech Features (Pipeline Generated)


The following features are **not part of the original dataset** but are extracted during preprocessing.


| Feature Group | Description |
|---|---|
| **Prosodic features** | Pitch statistics, speech rate, pause duration, energy statistics |
| **Spectral features** | Mel-frequency cepstral coefficients (MFCCs) capturing vocal tract characteristics |
| **Voice quality features** | Jitter, shimmer, harmonic-to-noise ratio (HNR) |
| **Linguistic features** | Word count, lexical diversity, sentiment polarity, average sentence length (derived from generated transcripts) |
| **Engineered speech features** | Pitch variability indices, pause frequency metrics, normalized energy ratios |


These acoustic and linguistic features are used to identify **speech biomarkers associated with depressive symptoms**. The extracted feature vectors are incorporated into the **audio branch of the multimodal depression detection model** and used for **cross-dataset evaluation alongside DAIC-WOZ**.


### 3.4 Transcript Dataset -- DAIC-WOZ


| Attribute | Value |
|---|---|
| **Total transcripts** | 189 (one per session) |
| **Format** | Tab-separated CSV with `start_time`, `stop_time`, `speaker`, `value` |
| **Speaker roles** | Participant and Ellie (virtual interviewer) |
| **Alignment** | Per-utterance timestamps enable precise mapping to audio segments |
| **NLP features extracted** | 14 hand-crafted features per participant |


**Transcript NLP feature set:**


| Feature | Description |
|---|---|
| `num_responses` | Number of participant responses |
| `total_words` | Total word count (participant only) |
| `avg_words_per_response` | Mean words per turn |
| `word_rate_per_min` | Speaking rate |
| `total_speaking_duration_s` | Total participant speaking time |
| `pct_silence` | Percentage of session that is silence |
| `first_person_density` | Frequency of first-person pronouns (i, me, my, myself, mine, i'm, i've, i'd, i'll) |
| `negative_word_density` | Frequency of negative-affect words (43-word LIWC/PHQ-derived lexicon) |
| `phq_keyword_density` | Frequency of PHQ-9 symptom keywords (31-word lexicon) |
| `filler_density` | Frequency of disfluency markers (um, uh, hmm, like, you know) |
| `avg_response_gap_s` | Mean inter-turn silence |
| `std_response_gap_s` | Variability in inter-turn silence |
| `lexical_diversity` | Type-token ratio |
| `avg_response_duration_s` | Mean turn duration |


### 3.5 RAG Knowledge Dataset (Planned)


| Attribute | Planned Value |
|---|---|
| **Document count** | To be determined (target: 50-100 guideline documents) |
| **Document types** | APA Practice Guidelines, WHO mhGAP Guide, PHQ manuals, DSM-5 MDD criteria, relevant clinical literature |
| **Average document length** | Variable (will be chunked to 512-token segments for embedding) |
| **Status** | Not yet assembled; planned for Milestone 4-5 |


---


## **4. Dataset Quality Assessment**


### 4.1 DAIC-WOZ Quality Assessment


The DAIC-WOZ notebook includes a dedicated quality assessment cell (Section 5b) that performs five systematic checks:


| Check | Method | Outcome | Action Taken |
|---|---|---|---|
| **Duplicate feature rows** | `DataFrame.duplicated(subset=feat_cols)` | 0 duplicates found | None required |
| **Audio quality proxy** | COVAREP voiced-frame ratio analysis (`covarep_voiced_ratio` -- mean, std, min, max) | All sessions above 10% voiced threshold | Sessions with <10% voiced frames would be flagged; none found |
| **Transcript consistency** | Checked `nlp_num_responses` and `nlp_total_words` for zero values | No empty transcripts | Transcript NLP features computed only from participant turns (interviewer excluded) |
| **Near-zero variance** | Features with variance < 1e-10 flagged | No problematic features identified | All features carry meaningful signal |
| **Cross-modality completeness** | Per-modality feature availability (non-NaN rate before imputation) | Some participants missing CLNF/visual data | Columns with >50% missing dropped; remainder imputed |


**Additional quality measures applied during feature extraction:**


| Issue | Detection | Resolution |
|---|---|---|
| **Missing values** | Column-level NaN percentage computed | Columns >50% missing: dropped. Remaining: imputed with training-split mean |
| **Infinite values** | +/-inf detected in COVAREP statistics | Replaced with training-split median |
| **CLNF confidence** | OpenFace confidence scores vary per frame | Frames with confidence < 0.5 or success != 1 excluded before feature aggregation |
| **Formant artifacts** | Zero and negative formant values (measurement errors) | Replaced with NaN before aggregation |
| **Corrupted audio** | COVAREP voiced-frame ratio as proxy indicator | All files passed |
| **Metadata inconsistency** | Cross-referencing split CSVs with available files | 189 participants present across all splits |


### 4.2 MODMA Audio Quality Assessment


| Check | Method | Outcome | Action Taken |
|---|---|---|---|
| **Subject metadata validation** | Verification of subject IDs and metadata consistency across recordings | Dataset contains **52 subjects (23 MDD, 29 HC)** with consistent labeling across files | None required |
| **Duplicate subject recordings** | Folder-level subject ID audit | Each subject stored in a dedicated folder with **~29 recordings per participant** | None required |
| **Audio format verification** | WAV header inspection (sampling rate and bit depth) | All recordings stored in **WAV format, 44.1 kHz sampling rate, 24-bit depth** | None required |
| **Recording completeness** | File presence check across subject directories | Each subject contains **29 voice samples corresponding to different speech tasks** | None required |
| **Speech task diversity** | Inspection of task types in dataset documentation | Recordings include **interview responses, passage reading, vocabulary reading, and picture description tasks** | All tasks retained for feature extraction |
| **Dataset class balance** | Class distribution analysis | **23 MDD vs 29 HC** (~44% vs 56%), indicating mild class imbalance | Stratified training splits applied |
| **Recording environment quality** | Review of dataset collection protocol | Speech recordings collected under **controlled experimental conditions with professional recording setup** | No additional filtering required |
| **Clinical label validation** | Cross-check of depression diagnosis labels | Participants clinically diagnosed and labeled using validated psychiatric assessment scales | Labels retained as provided |




### 4.3 RAVDESS Quality Assessment


| Check | Method | Outcome | Action Taken |
|---|---|---|---|
| **Missing features** | Column-level NaN check in feature matrix | 0 missing values | None required |
| **Duplicate files** | File hash / row duplication check | 0 duplicates | None required |
| **Audio quality** | Studio recording conditions | Professional acoustic environment; no noise artefacts | None required |
| **Near-zero variance** | Features with variance < 1e-10 | Flagged and reported; none problematic | None required |
| **Sampling consistency** | Native sample rate verification | All files at 48 kHz | Uniformly resampled to 22,050 Hz for feature extraction |
| **Class balance** | File count per emotion class | Perfectly balanced (192 per class; 96 for neutral) | None required |
| **Corrupted audio** | librosa load success check | All 1,440 files loaded successfully | None required |

---


## **5. Dataset Adequacy for the Problem**


### 5.1 Adequacy Evaluation


| Dataset | Sample Size | Adequate? | Key Considerations |
|---|---|---|---|
| **DAIC-WOZ** | 189 participants | Marginal for deep learning | Small dataset; however, both Whisper and ModernBERT encoders remain frozen -- only the lightweight fusion head (~50K parameters) is trained. Transfer learning from RAVDESS provides additional initialisation benefit. |
| **MODMA** | 55 subjects | Small | Provides Audio based clinical interview modules for late fusion. Frozen encoder strategy mitigates overfitting risk. |
| **RAVDESS** | 1,440 files | Adequate | Perfectly balanced 8-class emotion set. Sufficient for pre-training the classifier head on prosodic discrimination. |


### 5.2 Diversity Assessment


| Dimension | DAIC-WOZ | MODMA | RAVDESS |
|---|---|---|---|
| **Speaker diversity** | 189 unique participants (clinical population) | 55 subjects (university sample) | 24 professional actors |
| **Gender balance** | 63F / 79M (labelled) | 25F / 43M | 12F / 12M (perfect) |
| **Age range** | Adults (clinical intake) | 16-56 years (mean 31.1) | Professional actors (adult) |
| **Speech style** | Spontaneous clinical interview | Clinical Interviews | Scripted emotional statements |
| **Emotional coverage** | Natural depression / healthy | MDD / HC (clinical) | 8 discrete emotion classes |
| **Linguistic diversity** | Free-form clinical dialogue | N/A | 2 scripted sentences |


### 5.3 Limitations and Mitigation Strategies


| Limitation | Impact | Proposed Solution |
|---|---|---|
| **Small DAIC-WOZ sample** (N=189) | Risk of overfitting deep models | Frozen encoders; only train fusion head. Audio augmentation (time-stretch, pitch-shift, noise injection). LOSO cross-validation for robustness. |
| **Class imbalance** (DAIC-WOZ 70:30) | Model bias toward majority class | Weighted Cross-Entropy Loss prioritising recall for depressed class. Stratified splits. |
| **Limited speaker diversity** | Demographic bias | Cross-dataset transfer (RAVDESS actors + DAIC-WOZ clinical + MODMA university) provides broader coverage. |
| **DAIC-WOZ DUA pending** | Cannot process raw audio until approved | Pipeline validated on pre-extracted features. Full processing begins upon approval. |
| **No real-world stress data** | Cannot test RAG on non-clinical speech | Custom Contextual Stress Dataset (~500 samples) planned for Milestone 4-5. |


### 5.4 Augmentation Plan


| Modality | Technique | Target Dataset | Parameters | Rationale |
|---|---|---|---|---|
| Audio | Time-stretch | DAIC-WOZ | 0.9x - 1.1x | Simulate speaking rate variation |
| Audio | Pitch-shift | DAIC-WOZ | +-2 semitones | Simulate inter-speaker pitch variability |
| Audio | Additive Gaussian noise | DAIC-WOZ | SNR 15-25 dB | Test robustness to recording quality variation |
| Transcript | Synonym replacement | DAIC-WOZ | Context-aware | Increase linguistic diversity without altering clinical meaning |
| Transcript | Random word dropout | DAIC-WOZ | 5% | Simulate ASR transcription errors |


---


## **6. Training / Validation / Test Split Strategy**


### 6.1 Speech Data Split -- DAIC-WOZ


| Attribute | Detail |
|---|---|
| **Strategy** | Official AVEC 2017 participant-level splits |
| **Source** | Pre-defined CSV files (`train_split_Depression_AVEC2017.csv`, `dev_split_Depression_AVEC2017.csv`, `test_split_Depression_AVEC2017.csv`) |
| **Split sizes** | Train: 107. Validation: 35. Test: 47. |
| **Granularity** | Subject-level -- no single participant appears in more than one split |
| **Test labels** | Withheld per AVEC protocol (PHQ-8 values are NaN for test set) |
| **Leakage prevention** | Speaker-level splitting ensures the model cannot memorise participant-specific vocal characteristics |


**Class distribution per split (labelled only):**


| Split | Healthy | Depressed | % Depressed |
|---|---|---|---|
| Train | 77 | 30 | 28.0% |
| Validation | 23 | 12 | 34.3% |
| Test | -- | -- | Labels withheld |


### 6.2 Speech Data Split -- RAVDESS


| Attribute | Detail |
|---|---|
| **Strategy** | Actor-level gender-stratified split |
| **Split sizes** | Train: 14 actors (840 files). Validation: 5 actors (300 files). Test: 5 actors (300 files). |
| **Ratio** | 58.3% / 20.8% / 20.8% (~60/20/20) |
| **Gender balance** | Train: 7M + 7F. Val: 2M + 3F. Test: 3M + 2F. |
| **Random seed** | `np.random.seed(42)` |
| **Leakage prevention** | Actor-level splitting -- same actor's voice never appears across splits |


**Actor assignments:**


| Split | Actor IDs |
|---|---|
| Train | 1, 2, 3, 4, 5, 8, 11, 14, 17, 18, 19, 20, 21, 22 |
| Validation | 6, 9, 10, 12, 23 |
| Test | 7, 13, 15, 16, 24 |

### 6.4 RAG Knowledge Base Preparation


The RAG knowledge base does not follow a traditional train/val/test split. Instead:


- Clinical guideline documents will be chunked into 512-token segments
- Segments will be embedded using a sentence-transformer model (MiniLM)
- Embeddings will be indexed in a vector database for retrieval at inference time
- The entire knowledge base is used at inference -- no splitting is required




### 6.5 Leakage Prevention Measures


All three datasets implement the following safeguards, verified programmatically:


1. **Subject/Actor-level splits** -- No individual appears in more than one split
2. **Set-intersection assertions** -- Zero overlap verified via `assert len(train_set & test_set) == 0`
3. **Scaler fitted on training data only** -- StandardScaler (DAIC-WOZ) and RobustScaler (MODMA, RAVDESS) fit exclusively on the training split, then `transform()` applied to validation and test
4. **Imputation from training statistics only** -- Missing values filled using training-set mean/median, preventing information leakage from validation/test sets
5. **PHQ-9 exclusion** -- In MODMA, PHQ-9 (the diagnostic criterion defining the MDD label) is excluded from the feature set along with its derived `severity_composite`. Hard assertions block execution if these columns reappear.


### 6.6 LOSO Cross-Validation (Planned)


Leave-One-Subject-Out (LOSO) cross-validation is planned for the model evaluation phase (Milestone 3-4) as a robustness check alongside the primary fixed splits:


```python
from sklearn.model_selection import LeaveOneGroupOut
loso = LeaveOneGroupOut()
for train_idx, test_idx in loso.split(X, y, groups=participant_ids):
    # Train model on X[train_idx], evaluate on X[test_idx]
```


LOSO is the de facto standard for DAIC-WOZ evaluation and prevents the model from memorising speaker-specific vocal characteristics.


---


## **7. Alignment Between Speech and Transcript Data**


### 7.1 Current Alignment (Milestone 2)


At present, speech and transcript data are aligned at the **session level**. Each DAIC-WOZ participant has:


```
Audio files (COVAREP, FORMANT) --> Acoustic features (one aggregate vector)
Transcript file (CSV) ---------> NLP features (one aggregate vector)
Visual files (CLNF) -----------> Facial/pose/gaze features (one aggregate vector)


All three merged into a single 448-dimensional feature vector per participant.
Label: PHQ-8 binary classification + PHQ-8 continuous score
```


The alignment is maintained through consistent participant IDs across all modality files (e.g., Participant 300 has files `300_COVAREP.csv`, `300_TRANSCRIPT.csv`, `300_CLNF_AUs.txt`, etc.).


### 7.2 Alignment Verification


- Participant IDs are cross-referenced across all modality files and the split CSV files
- Any participant missing a modality file has that modality's features marked as NaN and handled during imputation
- The final merged DataFrame is verified to have exactly 189 rows (one per participant)


### 7.3 Planned Chunk-Level Alignment (Milestone 3)


The Gated Dual-Modal Fusion model requires temporally aligned audio-text pairs rather than session averages. The planned pipeline:


**Step 1 -- Audio segmentation.** Each DAIC-WOZ session is divided into 20-second non-overlapping windows.


**Step 2 -- Whisper embedding extraction.**
```
Audio chunk (20s, 16 kHz) --> Whisper encoder (frozen) --> hidden states [T x d]
--> Temporal Statistics Pooling (mean + std) --> 512-d audio embedding
```


**Step 3 -- Transcript alignment.** DAIC-WOZ transcripts include per-utterance `start_time` and `stop_time`. For each 20-second window [t, t+20], all participant turns with `start_time` falling within that window are concatenated into one text segment.


**Step 4 -- ModernBERT embedding.**
```
Aligned text segment --> BioClinical-ModernBERT (frozen, 8192 tokens)
--> [CLS] embedding --> 768-d text embedding
```


**Step 5 -- Paired output.** Each chunk yields: `(audio_emb_512d, text_emb_768d, PHQ8_binary, PHQ8_score, participant_id)`. Labels are inherited from session-level PHQ-8 ground truth.


### 7.4 RAVDESS Alignment


RAVDESS does not require speech-transcript alignment. The dataset uses two scripted statements; emotion is conveyed entirely through prosody, not linguistic content. Each audio file maps directly to a single emotion label.


### 7.5 MODMA Alignment


MODMA provides speech recordings only (no transcripts). The transcription sub-module will generate transcripts which would be used within the pipeline.


---


## **8. Dataset Requirements for Model Training**


| Requirement | Source Dataset | Status | Notes |
|---|---|---|---|
| Synchronised audio + transcript with clinical labels | DAIC-WOZ | Available | Per-utterance timestamps in transcript CSV enable precise pairing |
| PHQ-8 severity scores (regression target) | DAIC-WOZ | Available | Continuous 0-24 scale |
| Binary depression labels | DAIC-WOZ + MODMA | Available | PHQ-8 >= 10 (DAIC-WOZ); PHQ-9 >= 10 (MODMA) |
| Emotion labels for classifier pre-training | RAVDESS | Available | 8-class, perfectly balanced |
| Long-context clinical transcripts | DAIC-WOZ | Available | Full interview transcripts fit ModernBERT's 8,192-token window |
| Subject-level train/val/test splits | All three datasets | Implemented | Zero-overlap verified programmatically |
| Scalers fit on training data only | All three datasets | Implemented | Saved as `.pkl` for reproducible transforms |
| No target leakage in features | All three datasets | Verified | PHQ-9 excluded from MODMA features with hard assertion |
| Class imbalance handling | DAIC-WOZ, MODMA | Planned | Weighted Cross-Entropy Loss to be implemented during training |
| Minimum audio quality | DAIC-WOZ | Verified | COVAREP voiced-frame ratio check; CLNF confidence >= 0.5 filtering |
| Consistent sampling rates | DAIC-WOZ, RAVDESS | Verified | DAIC-WOZ: 16 kHz native; RAVDESS: 48 kHz resampled to 22,050 Hz |
| Sufficient transcript length | DAIC-WOZ | Verified | No zero-word transcripts; all participants have valid responses |
| Balanced emotional classes | RAVDESS | Verified | 192 files per class (96 for neutral) |


---


## **9. Speech Dataset Requirements**


### 9.1 Sampling Rate Consistency


| Dataset | Native Rate | Working Rate | Whisper Requirement | Action |
|---|---|---|---|---|
| DAIC-WOZ | 16 kHz | 16 kHz | 16 kHz | No resampling needed -- native rate matches Whisper input |
| RAVDESS | 48 kHz | 22,050 Hz (librosa) | 16 kHz (if Whisper used) | Currently resampled to 22,050 Hz for feature extraction. Will be resampled to 16 kHz for Whisper embedding extraction in Milestone 3. |


### 9.2 Noise Filtering and Quality Control


**DAIC-WOZ:**
- COVAREP voiced-frame ratio computed per session: measures the proportion of frames classified as voiced speech. Sessions with <10% voiced frames flagged as potential silence/noise issues (none found).
- CLNF (OpenFace) confidence filtering applied at the frame level: only frames with `confidence >= 0.5` and `success == 1` are included in visual feature aggregation.
- Formant values: zero and negative formant frequencies (measurement artefacts) replaced with NaN before aggregation.


**RAVDESS:**
- No noise filtering required. All recordings are studio-quality under controlled acoustic conditions.


### 9.3 Segmentation


**Current state (Milestone 2):** Session-level aggregation. Each DAIC-WOZ session is summarised as a single feature vector (448 dimensions) using mean, std, min, and max statistics across all frames.


**Planned (Milestone 3):** 20-second non-overlapping window segmentation of DAIC-WOZ sessions. Each window produces one audio-text embedding pair. This is required for the Gated Dual-Modal Fusion architecture.


### 9.4 Feature Extraction Preparation


**DAIC-WOZ feature extraction pipeline (6 modalities, 448 total features):**


| Modality | Source | Raw Columns | Aggregation | Output Features |
|---|---|---|---|---|
| COVAREP Acoustics | `{ID}_COVAREP.csv` | 74 (F0, VUV, NAQ, QOQ, H1H2, PSP, MDQ, peakSlope, Rd, MCEP_00-24, HMPDM_00-23, HMPDD_00-14) | mean, std, min, max per column + voiced F0 stats + voiced ratio | 300 |
| Formants | `{ID}_FORMANT.csv` | 5 (F1-F5 Hz) | mean, std, min, max + dispersion ratios (F2-F1, F3-F1) | 23 |
| Facial Action Units | `{ID}_CLNF_AUs.txt` | 14 AU_r + 6 AU_c + confidence | mean+std (AU_r), occurrence rate (AU_c), avg_confidence, pct_success | 36 |
| Head Pose | `{ID}_CLNF_pose.txt` | 6 (Tx,Ty,Tz,Rx,Ry,Rz) | mean, std, min, max + rot_energy + trans_energy | 26 |
| Eye Gaze | `{ID}_CLNF_gaze.txt` | 12 gaze vectors | mean, std, min, max + gaze_variability | 49 |
| Transcript NLP | `{ID}_TRANSCRIPT.csv` | Raw text | 14 hand-crafted linguistic features | 14 |


**RAVDESS feature extraction (338 features per clip):**


| Feature Group | Count | Method |
|---|---|---|
| MFCC mean (40 coefficients) | 40 | `librosa.feature.mfcc(n_mfcc=40)`, mean across time |
| MFCC std (40 coefficients) | 40 | std across time |
| Delta MFCC mean | 40 | 1st-order derivative, mean |
| Delta MFCC std | 40 | 1st-order derivative, std |
| Chroma mean (12 bins) | 12 | `librosa.feature.chroma_stft`, mean |
| Chroma std (12 bins) | 12 | std |
| Mel spectrogram mean (128 bands) | 128 | `librosa.feature.melspectrogram(n_mels=128)`, mean |
| Spectral features | 20 | centroid, bandwidth, rolloff (mean+std) + contrast (7 bands, mean+std) |
| Zero crossing rate | 2 | mean + std |
| RMS energy | 2 | mean + std |
| Tempo | 1 | `librosa.beat.tempo` |
| Duration | 1 | Clip duration in seconds |


All features extracted at SR=22,050 Hz using `librosa`.


---


## **10. LLM Fine-Tuning Dataset Requirements**


### 10.1 Transcript Formatting for BioClinical-ModernBERT


The BioClinical-ModernBERT encoder (`lindvalllab/BioClinical-ModernBERT-base`) processes DAIC-WOZ interview transcripts with the following considerations:


**Token length.** ModernBERT supports an extended 8,192-token context window via Flash Attention and Rotary Positional Embeddings (RoPE). Most DAIC-WOZ interview transcripts fall within this limit. For any transcripts exceeding 8,192 tokens, a chunking strategy with segment-level embedding aggregation will be applied.


**Text normalisation.** Transcripts are cleaned to remove interviewer (Ellie) turns, retaining only participant speech. Filler words and disfluencies (um, uh, hmm) are preserved as these serve as clinically relevant paralinguistic markers.


**Conversational context.** The full interview transcript is provided as a single contiguous document to the [CLS] token extraction pipeline, preserving conversational flow and long-range semantic dependencies. No instruction-response reformatting is applied -- the encoder processes raw clinical dialogue.


### 10.2 Supplementary Linguistic Feature Vector


In addition to ModernBERT embeddings, a lightweight Linguistic Feature Vector is extracted independently from transcripts:


- **Sentiment polarity** (positive/negative affect word density)
- **First-person pronoun density** (frequency of I, me, my, myself, etc.)
- **PHQ keyword density** (31 symptom-related terms)
- **Lexical diversity** (type-token ratio)
- **Filler/disfluency density** (um, uh, like, you know)


These features are concatenated with the [CLS] embedding to ground the model in established psycholinguistic markers.


### 10.3 Current State


ModernBERT tokenisation and embedding extraction are documented but not yet implemented. The current pipeline extracts 14 hand-crafted NLP features from transcripts using vanilla Python string processing. The transformer-based embedding pipeline is planned for Milestone 3.


---


## **11. RAG Dataset Preparation**


### 11.1 Knowledge Base Design (Planned)


The RAG module requires a curated knowledge base of clinical guidelines. The planned preparation pipeline:


**Document sources:**
- APA Practice Guidelines for the Treatment of Major Depressive Disorder
- WHO mhGAP Intervention Guide for Mental Health
- PHQ-8 and PHQ-9 scoring and interpretation manuals
- DSM-5 diagnostic criteria for Major Depressive Disorder
- Relevant published literature on vocal biomarkers and depression screening


**Document cleaning:**
- Remove headers, footers, page numbers, and non-content elements from PDF/text sources
- Standardise formatting and encoding
- Verify completeness and remove duplicates


**Chunking strategy:**
- Fixed-size chunks of 512 tokens with 64-token overlap between adjacent chunks
- Chunk boundaries respect sentence boundaries where possible
- Each chunk retains metadata (source document, section, page number)


**Embedding generation:**
- Sentence-transformer model (e.g., `all-MiniLM-L6-v2`) encodes each chunk into a dense vector
- Embeddings stored with associated metadata for retrieval traceability


**Vector database:**
- Embeddings indexed in a vector store (e.g., FAISS or ChromaDB)
- Retrieval at inference: top-k nearest neighbours to the query (extracted signal descriptions + risk score)
- Retrieved chunks fed to a small instruction-tuned LLM for constrained clinical summary generation


### 11.2 Current Status


The RAG knowledge base has not yet been assembled. This component is deferred to Milestone 4-5 (Forensic Evaluation and RAG Integration phase), as it is a post-hoc inference-time module that is meaningful only after the core Gated Fusion model is trained.


---


## **12. Preprocessing Pipeline**


### 12.1 Audio Preprocessing


**DAIC-WOZ:**
```
Raw COVAREP/FORMANT/CLNF files (per participant)
  --> Load temporal feature files (74-column COVAREP, 5-column Formant, etc.)
  --> CLNF confidence filtering (confidence >= 0.5, success == 1)
  --> Replace zero/negative formant values with NaN
  --> Compute aggregation statistics (mean, std, min, max) per feature column
  --> Compute special features (voiced F0 stats, voiced ratio, gaze variability, etc.)
  --> Merge all modalities into one 448-d feature vector per participant
  --> Drop columns with >50% missing values
  --> Replace +/-inf with NaN
  --> Impute remaining NaN with training-split mean/median
  --> StandardScaler fit on train, transform all splits
  --> Save as .npy arrays (float32)
```


**RAVDESS:**
```
Raw WAV files (48 kHz, 16-bit, mono)
  --> Resample to 22,050 Hz via librosa
  --> Extract 338 features per clip (MFCCs, Delta MFCCs, chroma, mel spectrogram,
      spectral features, ZCR, RMS, tempo, duration)
  --> Construct feature DataFrame with emotion and actor metadata
  --> Actor-level split (14/5/5 actors)
  --> RobustScaler fit on train, transform all splits
  --> Save as .npy arrays (float64)
```


### 12.2 Text Preprocessing


**DAIC-WOZ Transcripts:**
```
Raw transcript CSV (tab-separated: start_time, stop_time, speaker, value)
  --> Filter to participant turns only (exclude interviewer "Ellie")
  --> Compute 14 NLP features:
      - Word counts, speaking rate, turn duration
      - First-person pronoun density (9-word lexicon)
      - Negative affect word density (43-word LIWC/PHQ lexicon)
      - PHQ-9 symptom keyword density (31-word lexicon)
      - Filler/disfluency density (9-word lexicon)
      - Lexical diversity (type-token ratio)
      - Response gap statistics (mean, std of inter-turn silences)
  --> Merge with acoustic features by participant ID
```


### 12.3 RAG Preprocessing (Planned)


```
Clinical guideline documents (PDF/text)
  --> Clean: remove headers, footers, page numbers, non-content elements
  --> Chunk: 512-token fixed-size segments with 64-token overlap
  --> Embed: Sentence-transformer (MiniLM) per chunk
  --> Index: Store in vector database (FAISS / ChromaDB) with source metadata
```


---


## **13. Reproducibility**


### 13.1 Reproducibility Guarantees


| Aspect | Implementation |
|---|---|
| **Random seeds** | `random_state=42` (MODMA split), `np.random.seed(42)` (RAVDESS actor assignment) |
| **Deterministic splits** | DAIC-WOZ uses AVEC 2017 pre-defined CSVs (deterministic, no randomness) |
| **Scaler persistence** | Fitted StandardScaler and RobustScaler objects saved as `.pkl` files. Applying the same scaler to new data produces identical transforms. |
| **Feature column order** | Saved as `.txt` files (`daicwoz_feature_cols.txt`, `feature_names.txt`, `ravdess_feature_cols.txt`). Exact column ordering is preserved and can be loaded for inference. |
| **Split manifests** | `daicwoz_participant_splits.csv`, `subject_splits.csv`, `ravdess_actor_splits.csv` record the exact assignment of each participant/subject/actor to train, validation, or test. |
| **Self-contained notebooks** | All three notebooks in `Notebooks_EDA/` are self-contained. Re-running any notebook with the same input data produces identical output files. |
| **Version control** | All code, reports, and configuration files tracked in Git. |


### 13.2 Complete Artifact Inventory


**DAIC-WOZ artifacts (`processed_data/daicwoz_*`):**


| File | Shape / Content | Size |
|---|---|---|
| `daicwoz_X_train.npy` | (107, 448) float32 | 187.4 KB |
| `daicwoz_X_val.npy` | (35, 448) float32 | 61.4 KB |
| `daicwoz_X_test.npy` | (47, 448) float32 | 82.4 KB |
| `daicwoz_y_train_bin.npy` | (107,) int32 -- binary labels | 0.5 KB |
| `daicwoz_y_val_bin.npy` | (35,) int32 | 0.3 KB |
| `daicwoz_y_test_bin.npy` | (47,) float64 -- NaN (unlabelled) | 0.5 KB |
| `daicwoz_y_train_score.npy` | (107,) float32 -- PHQ-8 scores | 0.5 KB |
| `daicwoz_y_val_score.npy` | (35,) float32 | 0.3 KB |
| `daicwoz_y_test_score.npy` | (47,) float64 -- NaN | 0.5 KB |
| `daicwoz_feature_cols.txt` | 448 feature names | -- |
| `daicwoz_features.csv` | 190 rows (full unscaled features) | 796.2 KB |
| `daicwoz_participant_splits.csv` | Participant-to-split mapping | 3.4 KB |
| `daicwoz_scaler.pkl` | Fitted StandardScaler | 10.9 KB |


**RAVDESS artifacts (`processed_data/ravdess_*`):**


| File | Shape / Content | Size |
|---|---|---|
| `ravdess_X_train.npy` | (840, 338) float64 | 2,218.2 KB |
| `ravdess_X_val.npy` | (300, 338) float64 | 792.3 KB |
| `ravdess_X_test.npy` | (300, 338) float64 | 792.3 KB |
| `ravdess_y_train.npy` | (840,) int64 | 6.7 KB |
| `ravdess_y_val.npy` | (300,) int64 | 2.5 KB |
| `ravdess_y_test.npy` | (300,) int64 | 2.5 KB |
| `ravdess_scaler.pkl` | Fitted RobustScaler | 5.7 KB |
| `ravdess_emotion_label_map.json` | `{angry:0, calm:1, ..., surprised:7}` | -- |
| `ravdess_actor_splits.csv` | Actor-to-split mapping | 0.2 KB |
| `ravdess_features.csv` | Full feature dataset (1,441 rows) | 5,530.1 KB |
| `ravdess_feature_cols.txt` | 338 feature names | -- |
| `ravdess_metadata.csv` | Metadata DataFrame (1,441 rows) | 276.3 KB |


**EDA visualisation artifacts (PNG):**


| File | Description | Size |
|---|---|---|
| `eda_e_clnf_analysis.png` | DAIC-WOZ CLNF confidence analysis | 265 KB |
| `eda_f_transcript_nlp.png` | DAIC-WOZ transcript NLP distributions | 272 KB |
| `ravdess_waveforms_spectrograms.png` | RAVDESS waveform and spectrogram samples | 426 KB |
| `ravdess_mfcc_by_emotion.png` | MFCC distributions across emotions | 222 KB |
| `ravdess_key_features_by_emotion.png` | Key feature box plots by emotion | 123 KB |
| `ravdess_feature_correlation.png` | Feature correlation heatmap | 99 KB |
| `ravdess_actor_emotion_heatmap.png` | Actor-emotion distribution | 92 KB |
| `ravdess_gender_emotion_features.png` | Gender-emotion feature interactions | 63 KB |
| `ravdess_mfcc_profile_per_emotion.png` | Mean MFCC profile per emotion | 62 KB |
| `ravdess_metadata_distributions.png` | Emotion/gender/intensity distributions | 51 KB |
| `ravdess_duration_analysis.png` | Duration distribution analysis | 48 KB |


### 13.3 How to Reproduce


1. **Obtain raw datasets.** Download DAIC-WOZ (via DUA from USC ICT), RAVDESS (from Zenodo/Kaggle under CC BY-NC-SA 4.0), and MODMA (via EULA from official website).


2. **Update base paths.** In each notebook, update the `BASE` variable to point to the local directory containing the raw dataset files:
   - `DAICWOZ_Feature_Extraction.ipynb`: Line 64 (`BASE = r"path/to/daicwoz"`)


   - `RAVDESS_EDA_and_Feature_Extraction.ipynb`: Line 62 (`BASE = Path('path/to/project')`)


3. **Run notebooks sequentially.** Execute all cells in each notebook. Each notebook is self-contained and will produce the corresponding artifacts in `processed_data/`.


4. **Verify outputs.** Compare output `.npy` shapes and `.csv` row counts against the artifact inventory in Section 13.2.


---


*All preprocessing notebooks are located in `Notebooks_EDA/` and all saved artifacts in `processed_data/`.*
*Re-running any notebook with the same input data and the same random seeds produces identical outputs.*



