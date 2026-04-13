# **Model Architecture and Methodology for Multimodal AI Systems**

## **AI-Based Early Mental Health Breakdown Detection from Speech Patterns**

### **Group 6 | DS & AI Lab Project (BSDA4001)**

**Team Members:** Om Aryan · Pankaj Mohan Sahu · Drashti Shah · Mahi Mudgal · G Hamsini

---

# **1\. Dataset Organization**

## **1.1 Directory Structure (Raw & Processed)**

The project repository is organized to separate raw source material from preprocessed feature arrays clearly, and to isolate each dataset's train/validation/test splits. The dataset is stored locally due to its large size and is not stored on GitHub. The links for the dataset are present in the *data/dataset\_links.txt*

## **1.2 Dataset Statistics**

### **DAIC-WOZ Split Summary**

| Split | Samples |
| ----- | ----- |
| Train | 107 |
| Validation | 35 |
| Test | 47 |

### **Label Distribution (Train)**

| Class | Count |
| ----- | ----- |
| Healthy (PHQ-8 \< 10\) | 77 |
| Depressed (PHQ-8 ≥ 10\) | 30 |

### **PHQ-8 Score Statistics (Train)**

| Statistic | Value |
| ----- | ----- |
| Range | 0 – 20 |
| Mean | 6.42 |

---

# **2\. Preprocessing Pipeline**

Before any model sees the data, each modality passes through a dedicated preprocessing chain. The table below summarizes the transformation from raw input to model-ready tensors.

| Modality | Raw Input | Preprocessing Steps | Model Input Format |
| ----- | ----- | ----- | ----- |
| **Tabular — Acoustic** (Pipeline 1\) | Session-level COVAREP \+ formant CSVs from raw `.zip` archives | (1) Session extraction; (2) per-feature aggregation (mean/std over time); (3) missing-value imputation; (4) prefix-based branch assignment; (5) StandardScaler normalization; (6) train/val/test split export | `X_ac: (N, 323)` float32 NumPy array |
| **Tabular — Linguistic** (Pipeline 1\) | Transcript-derived NLP features (first-person density, PHQ keyword rate, pause rate, lexical diversity, etc.) | Same pipeline as acoustic; 14-feature subset selected by prefix `nlp_` | `X_nlp: (N, 14)` float32 NumPy array |
| **Tabular — Visual** (Pipeline 1\) | OpenFace Action Units, head pose, gaze vectors from video CSVs | Same pipeline; 111-feature subset selected by prefixes `au_`, `pose_`, `gaze_` | `X_vis: (N, 111)` float32 NumPy array |
| **Audio** (Pipeline 2\) | Raw `.wav` (variable length, variable sample rate) | (1) Resample to 16 kHz mono via `librosa`; (2) pad/truncate to exactly 30 s (480,000 samples); (3) log-mel spectrogram extraction using Whisper's `WhisperFeatureExtractor` (80 mel bins, 25 ms window, 10 ms hop → 3,000 time frames); (4) forward pass through frozen Whisper-small encoder → shape `(N_frames, 768)`; (5) temporal statistics pooling: mean \+ std across the time axis → flattened to 1,536-D; (6) linear projection 1,536 → 512 | `audio_emb: (N, 512)` float32 tensor |
| **Text** (Pipeline 2\) | Raw participant transcript / scripted emotion sentence | (1) BioClinicalBERT tokenizer (`emilyalsentzer/Bio_ClinicalBERT`); (2) truncation and padding to max 512 tokens; (3) forward pass through frozen BioClinicalBERT → `(512, 768)` sequence output; (4) `[CLS]` token embedding extracted → 768-D; (5) linear projection 768 → 512 | `text_emb: (N, 512)` float32 tensor |

### **Class Imbalance Handling**

The training set is imbalanced (77 healthy vs 30 depressed). Rather than oversampling (which can introduce artefacts on a 107-sample dataset), class imbalance is addressed via weighted cross-entropy. The positive class weight is computed as:

pos\_weight \= n\_negative / n\_positive \= 77 / 30 ≈ 2.567

This is passed directly as `scale_pos_weight` to XGBoost (Pipeline 1\) and as `pos_weight` to `BCEWithLogitsLoss` (Pipeline 2).

---

# **3\. Model Architecture**

The project implements two complementary pipelines that share the same clinical task but differ in the type of input and the modeling strategy.

## **3.1 Pipeline 1 — Classical ML Branch Fusion (CPU)**

### **Overview**

Three independent XGBoost classifiers are trained on the three modality-specific feature subsets. Their soft probability outputs are fed as meta-features into a logistic regression meta-learner (stacking ensemble). A separate XGBoost regressor handles the PHQ-8 score prediction task.

### **Major Components**

**Acoustic Branch** takes the 323 COVAREP \+ formant features and trains an XGBoost classifier. These features encode pitch (F0), glottal properties (NAQ, H1H2), mel-cepstral coefficients (MCEP), and vocal tract resonances (formants F1–F5) — all well-validated acoustic correlates of depression.

**Linguistic Branch** takes the 14 NLP features and trains a lighter XGBoost classifier (shallower trees, full column sampling) to account for the very small feature space.

**Visual Branch** takes the 111 Action Unit, pose, and gaze features and trains an XGBoost classifier. This modality is the weakest due to the absence of temporal sequence modeling.

**Meta-Learner** receives a 3-dimensional input vector `[P(dep|acoustic), P(dep|linguistic), P(dep|visual)]` and learns a logistic regression weighting over these branch probabilities. This is the stacking fusion step.

**PHQ-8 Regressor** is an independent XGBoost regressor trained on all 448 features to predict the continuous severity score (regression task). Predictions are clipped to \[0, 24\] and bucketed into DSM-5 severity bands (Minimal/Mild/Moderate/Mod-Severe/Severe).

### **XGBoost Hyperparameters (per branch)**

| Parameter | Value | Rationale |
| ----- | ----- | ----- |
| n\_estimators | 500 | More trees compensate for shallow depth |
| max\_depth | 3 | Very shallow — prevents overfitting on N=107 |
| learning\_rate | 0.03 | Low LR \+ many trees \= better generalization |
| subsample | 0.8 | Stochastic row sampling per tree |
| colsample\_bytree | 0.7 | Stochastic feature sampling per tree |
| min\_child\_weight | 5 | ≥5 samples per leaf — prevents tiny splits |
| reg\_alpha | 0.5 | L1 regularization → feature sparsity |
| reg\_lambda | 1.5 | L2 regularization |
| eval\_metric | aucpr | PR-AUC is better than logloss for imbalanced data |
| early\_stopping\_rounds | 40 | Stops when validation PR-AUC plateaus |
| scale\_pos\_weight | 2.567 | Weighted cross-entropy for class imbalance |

## **3.2 Pipeline 2 — Foundation Models with Gated Dual-Modal Fusion (GPU)**

### **Overview**

Two large pretrained encoders (frozen) independently embed raw audio and raw text. A trainable sigmoid gate learns to dynamically weight the contribution of each modality before passing the fused representation into a 3-layer MLP head with two output heads (binary classification \+ regression).

### **Major Components**

**Whisper Encoder** (openai/whisper-small, \~244M parameters, frozen). Receives log-mel spectrograms and outputs a sequence of hidden states. Temporal mean \+ std pooling produces a 1,536-D embedding, which is linearly projected to 512-D.

**BioClinicalBERT Encoder** (\~110M parameters, frozen). Domain-adapted BERT trained on clinical notes. The `[CLS]` token embedding (768-D) is linearly projected to 512-D.

**Sigmoid Gate** receives the concatenated 1,024-D vector `[audio_proj; text_proj]` and outputs a scalar gate value `g = σ(W · [audio; text] + b)`. The fused representation is `g · audio_proj + (1 - g) · text_proj`, allowing the model to learn which modality is more reliable per sample.

**MLP Head** (3-layer trainable). Architecture: `512 → 256 → 128 → {1 (binary), 1 (score)}`. Uses ReLU activations and dropout between layers.

---

# **4\. Data Flow Diagrams**

## **4.1 Pipeline 1 — Classical ML Fusion**

╔══════════════════════════════════════════════════════════════════╗  
║              PIPELINE 1  ·  Classical ML Fusion (CPU)            ║  
║                                                                  ║  
║  DAIC-WOZ 448 pre-extracted features (processed\_data/daicwoz)    ║  
║       │                                                          ║  
║  ┌────▼──────┐  ┌────────────┐  ┌───────────────┐                ║  
║  │ Acoustic   │  │ Linguistic │  │   Visual      │ ← Branches   ║  
║  │ Branch     │  │ Branch     │  │   Branch      ║              ║  
║  │ COVAREP \+  │  │ NLP 14     │  │  AU+Pose+Gaze ║              ║  
║  │ Formants   │  │ features   │  │  111 features ║              ║  
║  │ 323 feats  │  │ XGBoost    │  │  XGBoost      ║              ║  
║  │ XGBoost    │  └─────┬──────┘  └──────┬────────╢              ║  
║  └────┬──────┘        │                 │        ║              ║  
║       └───────────────┼─────────────────┘        ║              ║  
║               ┌───────▼────────┐                  ║              ║  
║               │  Meta-Learner  │ ← Stacking (Gating analog)    ║  
║               │ Logistic Regr. │                  ║              ║  
║               └───────┬────────┘                  ║              ║  
║          ┌────────────┴────────────┐               ║              ║  
║     Binary Risk              PHQ-8 Score           ║              ║  
║    (F1 \+ Sensitivity)        (RMSE \+ Band Acc)     ║              ║  
║          │                        │                ║              ║  
║     SHAP Explanations        Feature Importance    ║              ║  
╚══════════════════════════════════════════════════════════════════╝

## **4.2 Pipeline 2 — Foundation Models with Gated Fusion**

 Raw .wav Audio                  Raw Transcript Text  
        │                                 		│  
       ▼                                                             ▼  
  Resample 16kHz mono           BioClinicalBERT Tokenizer  
  Pad/truncate to 30s           Truncate/pad → 512 tokens  
  Whisper FeatureExtractor      input\_ids: (N, 512\)  
  log-mel: (80, 3000\)                        │  
       │                                               ▼  
       ▼                     BioClinicalBERT Encoder \[FROZEN\]  
  Whisper Encoder \[FROZEN\]    \~110M params  
  \~244M params                output: (N, 512, 768\)  
  output: (N, 1500, 768\)               │  
       │                               ▼  
       ▼                     \[CLS\] token → (N, 768\)  
  Temporal Stats Pool                  │  
  mean \+ std → (N, 1536\)               ▼  
       │                     Linear Projection 768→512  
       ▼                     text\_proj: (N, 512\)  
  Linear Projection 1536→512          │  
  audio\_proj: (N, 512\)                 │  
       │                               │  
       └──────────┬────────────────────┘  
                  ▼  
         Concatenate → (N, 1024\)  
                  │  
                  ▼  
          Sigmoid Gate Layer  
          g \= σ(W·\[audio;text\])  
          fused \= g·audio \+ (1-g)·text  
          fused: (N, 512\)  
                  │  
                  ▼  
         3-Layer MLP Head  
         512 → 256 → 128  
              │  
    ┌─────────┴──────────┐  
    ▼                    ▼  
Binary Head          Regression Head  
128 → 1              128 → 1  
BCEWithLogitsLoss    HuberLoss (δ=1.0)  
    │                    │  
    ▼                    ▼  
P(depressed)         PHQ-8 Score  
0 or 1               0.0 – 24.0

---

# **5\. Input Format Specification**

## **5.1 Pipeline 1 — Input Shapes and Formats**

| Branch | NumPy Array | dtype | Notes |
| ----- | ----- | ----- | ----- |
| Full feature matrix | `(N, 448)` | float32 | Loaded from `daicwoz_X_*.npy` |
| Acoustic slice | `(N, 323)` | float32 | Columns with prefix `covarep_` or `formant_` |
| Linguistic slice | `(N, 14)` | float32 | Columns with prefix `nlp_` |
| Visual slice | `(N, 111)` | float32 | Columns with prefix `au_`, `pose_`, `gaze_` |
| Binary label | `(N,)` | int (0/1) | Threshold: PHQ-8 ≥ 10 → depressed |
| PHQ-8 score | `(N,)` | float32 | Continuous score 0–24 |
| Meta-learner input | `(N, 3)` | float32 | Stack of 3 branch probabilities |

## **5.2 Pipeline 2 — Tensor Formats**

| Stage | Tensor Shape | dtype | Description |
| ----- | ----- | ----- | ----- |
| Raw audio waveform | `(480,000,)` | float32 | 30 s × 16,000 Hz mono |
| Whisper log-mel | `(80, 3000)` | float32 | 80 mel bins, 3000 time frames |
| Whisper encoder output | `(N_frames, 768)` | float32 | Contextualized frame embeddings |
| Audio embedding (pooled) | `(N, 1536)` | float32 | mean \+ std across time |
| Audio projection | `(N, 512)` | float32 | After linear 1536→512 |
| Tokenized text | `(N, 512)` | int64 | Padded to max 512 tokens |
| BERT `[CLS]` embedding | `(N, 768)` | float32 | First token of last hidden state |
| Text projection | `(N, 512)` | float32 | After linear 768→512 |
| Fused representation | `(N, 512)` | float32 | After sigmoid gate |
| Binary logit | `(N, 1)` | float32 | Pre-sigmoid classification output |
| PHQ-8 prediction | `(N, 1)` | float32 | Continuous score |

---

# **6\. Architectural Justification**

## **6.1 Why XGBoost for Pipeline 1?**

The DAIC-WOZ training set contains only 107 samples with 448 features. This extreme data scarcity makes deep learning models highly susceptible to overfitting. XGBoost is chosen because:

* Gradient-boosted trees with `max_depth=3` and `min_child_weight=5` provide strong regularization for small datasets.  
* L1 (`reg_alpha`) and L2 (`reg_lambda`) penalties control feature sparsity across the 323-dimensional acoustic space.  
* `early_stopping_rounds=40` prevents overfitting without manual epoch tuning.  
* The model is fully interpretable via SHAP values, a clinical requirement.

Compared to alternatives such as Random Forests (which lack built-in regularization and sequential boosting), SVMs (which require careful kernel tuning and do not natively output calibrated probabilities), and shallow neural networks (which overfit severely at N=107), XGBoost provides the best bias-variance tradeoff.

## **6.2 Why Frozen Pretrained Encoders for Pipeline 2?**

Whisper and BioClinicalBERT are frozen during training. Fine-tuning them would require hundreds of labeled DAIC-WOZ samples at minimum; our 107-sample dataset would cause catastrophic overfitting. By freezing the encoders, we leverage their representations learned from millions of hours of speech (Whisper) and clinical text corpora (BioClinicalBERT) while only training the lightweight MLP head.

## **6.3 Why a Sigmoid Gate Instead of Concatenation?**

Simple concatenation (`[audio; text]`) gives equal weight to both modalities regardless of their reliability for a given sample. A sigmoid gate learns a dynamic, sample-specific weighting. If the acoustic features of a sample are noisy (e.g., background noise), the gate can shift reliance to the text branch. This is more robust than fixed-weight averaging.

## **6.4 Strengths and Limitations**

| Aspect | Pipeline 1 (Classical ML) | Pipeline 2 (Foundation Models) |
| ----- | ----- | ----- |
| Interpretability |  SHAP explanations |  Black-box encoders |
| Latency |  \~50–100 ms |  \~500 ms – 2 s |
| Data efficiency |  Works on N=107 |  Needs GPU; subset run only |
| Representation quality |  Hand-crafted features |  Deep contextualized embeddings |
| Temporal modeling |  Aggregated statistics |  Whisper captures temporal dynamics |
| Clinical deployment |  Readily deployable |  High compute cost |

---

# **7\. End-to-End Pipeline Verification**

## **7.1 Small-Scale Subset Run (Pipeline 2\)**

To verify that all pipeline components connect correctly before scaling, a small-scale integration test was run using the RAVDESS dataset (or a synthetic WAV fallback when RAVDESS is unavailable). This ensures the full path from raw `.wav` \+ raw text → frozen encoders → gated fusion → loss computation works correctly as a unit.

**Subset configuration:**

* Total samples: 160 (train=120, val=40)  
* Class distribution: balanced (60 healthy / 60 at-risk)  
* Epochs: 5 (with early stopping)

**Training log (subset run):**

Epoch | Train Loss | Val Loss | Val F1 | Val RMSE  
\------+------------+----------+--------+---------  
    1 |     2.6125 |   2.0395 | 0.3333 |   5.5776  
    2 |     2.1258 |   2.0408 | 0.3333 |   5.4966  
    3 |     2.1504 |   2.0481 | 0.3333 |   5.0373  
    4 |     2.0293 |   2.0414 | 0.3333 |   5.2114  
    5 |     2.0502 |   2.0421 | 0.3333 |   5.3013

- Early stopping at epoch 5  
- Pipeline 2 raw-data subset run completed.  
Best val Macro-F1 : 0.3333 (at epoch 1\)  
Best val RMSE     : 5.5776  
Raw-input path verified: WAV \+ text → frozen encoders → gated fusion → dual-task outputs

The low F1 on the synthetic subset is expected — the data has no clinically meaningful acoustic or linguistic variation. The purpose of this run is integration verification, not performance benchmarking.

## **7.2 Pipeline 1 Integration Verification**

Pipeline 1 is fully end-to-end verified on the actual DAIC-WOZ validation set. All 27 critical data files were confirmed present before training. All three branch models and the meta-learner were trained and saved to disk:

branch\_acoustic.joblib      (46.0 KB)  
branch\_linguistic.joblib    (74.6 KB)  
branch\_visual.joblib       (172.6 KB)  
meta\_learner\_logreg.joblib   (0.8 KB)  
phq8\_regressor.joblib       (50.9 KB)  
baseline\_xgb.joblib         (98.0 KB)  
modma\_lightgbm.joblib       (61.5 KB)  
feature\_config.json         (15.3 KB)

---

# **8\. Model Outputs — Examples**

## **8.1 Pipeline 1 — Clinical Inference Demo**

The following table shows Pipeline 1's output on a sample of 5 test-set participants. Each row represents one participant-level inference call to `pipeline1_predict(X_new)`.

\#   True Label  True PHQ-8  Pred Label     P(dep)  PHQ-8 Est  Band     Acoustic  Linguistic  Visual  
──────────────────────────────────────────────────────────────────────────────────────────────────  
1   Healthy     2           Depressed    0.5024     6.76     Mild      0.4830     0.4341    0.2743  
2   Healthy     3           Healthy      0.4940     6.33     Mild      0.4595     0.6648    0.4404  
3   Healthy     0           Depressed    0.5116     6.58     Mild      0.4894     0.5363    0.2440  
4   Depressed   22          Healthy      0.4972     6.72     Mild      0.5027     0.5719    0.3679  
5   Depressed   15          Healthy      0.4870     6.46     Mild      0.4886     0.3960    0.3765

Each output dict returned by `pipeline1_predict()` contains:

{  
    "binary\_risk":    0 or 1,          \# Healthy / Depressed  
    "risk\_prob":      float,           \# P(depressed) in \[0, 1\]  
    "phq8\_score":     float,           \# Predicted severity, clipped to \[0, 24\]  
    "severity\_band":  str,             \# Minimal / Mild / Moderate / Mod-Severe / Severe  
    "branch\_probs": {  
        "acoustic":   float,           \# P(depressed | acoustic features)  
        "linguistic": float,           \# P(depressed | linguistic features)  
        "visual":     float,           \# P(depressed | visual features)  
    }  
}

## **8.2 Pipeline 2 — Output Format**

Pipeline 2 produces two simultaneous outputs from the MLP head for each sample:

\# Classification head (before sigmoid)  
binary\_logit:  tensor of shape (N, 1), dtype=float32  
binary\_pred:   (binary\_logit \> 0.0).int()   →  0 \= Healthy, 1 \= Depressed

\# Regression head  
phq8\_pred:     tensor of shape (N, 1), dtype=float32, clipped to \[0, 24\]

---

# **9\. Loss Functions and Evaluation Metrics**

## **9.1 Loss Functions**

### **Pipeline 1 — Classification (XGBoost)**

**Weighted Binary Cross-Entropy:**

L \= \-\[pos\_weight × y · log(p) \+ (1 \- y) · log(1 \- p)\]

where `pos_weight = 2.567`. The eval metric during training is **AUPRC** (Area Under the Precision-Recall Curve), which is more appropriate than log-loss for imbalanced datasets.

### **Pipeline 1 — Regression (XGBoost)**

**Mean Squared Error (MSE):**

L \= (1/N) Σ (y\_score \- ŷ\_score)²

### **Pipeline 2 — Classification**

**BCEWithLogitsLoss** with `pos_weight=2.567`:

L\_clf \= BCEWithLogitsLoss(logit, y\_bin, pos\_weight=2.567)

### **Pipeline 2 — Regression**

**Huber Loss** (δ \= 1.0). Huber loss is chosen over MSE because it is less sensitive to outlier PHQ-8 scores (e.g., participants with extreme scores of 20+):

L\_reg \= HuberLoss(ŷ\_score, y\_score, delta=1.0)

### **Pipeline 2 — Combined Loss**

L\_total \= L\_clf \+ λ · L\_reg     (λ \= 1.0 by default)

## **9.2 Evaluation Metrics**

| Metric | Definition | Relevance |
| ----- | ----- | ----- |
| **Macro F1** | Mean of per-class F1 scores | Primary classification metric; robust to class imbalance |
| **Sensitivity (Recall)** | TP / (TP \+ FN) | Clinical priority: must not miss true depression cases |
| **Specificity** | TN / (TN \+ FP) | Measures false-alarm rate among healthy individuals |
| **ROC-AUC** | Area under ROC curve | Overall ranking quality across thresholds |
| **AUPRC** | Area under Precision-Recall curve | Better than AUC-ROC for imbalanced classes |
| **RMSE** | √(MSE) | Regression error in PHQ-8 units |
| **MAE** | Mean Absolute Error | Robust regression error |
| **Band Accuracy** | Exact severity band match | Clinical relevance metric |
| **Within-1-Band Accuracy** | Correct or off-by-one band | Practical clinical usability |

---

# **10\. Branch-Wise Performance Results (Validation Set)**

## **10.1 Acoustic Branch**

| Metric | Value |
| ----- | ----- |
| Macro F1 | **0.6023** |
| Depressed F1 | 0.4545 |
| Sensitivity | 0.4167 |
| Specificity | 0.7826 |
| ROC-AUC | 0.5743 |
| AUPRC | 0.5202 |

## **10.2 Linguistic Branch**

| Metric | Value |
| ----- | ----- |
| Macro F1 | **0.4898** |
| Depressed F1 | 0.2857 |
| Sensitivity | 0.2500 |
| Specificity | 0.7391 |
| ROC-AUC | 0.4891 |
| AUPRC | 0.4189 |

## **10.3 Visual Branch**

| Metric | Value |
| ----- | ----- |
| Macro F1 | **0.4329** |
| Depressed F1 | 0.1250 |
| Sensitivity | 0.0833 |
| Specificity | 0.8696 |
| ROC-AUC | 0.5978 |

## **10.4 Branch Comparison**

| Modality | Macro F1 | Sensitivity | ROC-AUC |
| ----- | ----- | ----- | ----- |
| Acoustic | **0.6023** | 0.4167 | 0.5743 |
| Linguistic | 0.4898 | 0.2500 | 0.4891 |
| Visual | 0.4329 | 0.0833 | 0.5978 |

## **10.5 Fusion Model (Stacking)**

| Metric | Value |
| ----- | ----- |
| Macro F1 | **0.3881** |
| Depressed F1 | 0.2308 |

**Key Observation:** The fusion model underperforms the acoustic-only branch (0.39 vs 0.60). Simple logistic stacking is insufficient; the branch predictions carry redundant rather than complementary signal.

## **10.6 Baseline (All-Features XGBoost)**

| Metric | Value |
| ----- | ----- |
| Macro F1 | 0.4400 |
| Depressed F1 | 0.2000 |
| Sensitivity | 0.1667 |
| Specificity | 0.7391 |

## **10.7 Regression Task (PHQ-8 Score)**

| Metric | Value |
| ----- | ----- |
| RMSE | 6.5588 |
| MAE | 5.4776 |
| Band Accuracy (exact) | 0.1714 |
| Within-1-Band Accuracy | **0.8000** |

---

# **11\. Parameter Complexity**

### **XGBoost (per branch)**

| Component | Estimate |
| ----- | ----- |
| Trees per branch | 500 |
| Nodes per tree (depth 3\) | \~8 nodes |
| Total nodes per branch | \~4,000 |
| Total nodes (3 branches) | \~12,000 decision nodes |

### **Meta-Learner**

| Component | Count |
| ----- | ----- |
| Input features | 3 |
| Parameters (weights \+ bias) | \~4 |

### **Foundation Model Pipeline (Pipeline 2\)**

| Model | Parameters |
| ----- | ----- |
| Whisper Encoder (openai/whisper-small) | \~244M |
| BioClinicalBERT | \~110M |
| Linear projections (2×) | \~1.4M |
| Sigmoid gate | \~1K |
| MLP Head (512→256→128→2) | \~165K |
| **Total (trainable only)** | **\~1.6M** |

---

# **12\. Training Observations**

### **Training Stability**

Early stopping triggered at approximately 40 rounds across branches — confirming that the models reached their validation optimum without extensive over-training. No severe overfitting was observed.

### **Class Imbalance Impact**

Despite using `pos_weight=2.567`, sensitivity remained below 0.5 across all branches on the validation set. This is attributable to the fundamental data scarcity (30 positive samples in training) rather than a failure of the imbalance correction strategy.

---

# **13\. Key Findings**

**1\. Acoustic features dominate.** The acoustic branch achieved the highest Macro F1 (0.60), confirming that speech patterns (pitch, glottal features, formants) are the strongest depression signals available in DAIC-WOZ.

**2\. Visual modality is weak.** Extremely low sensitivity (0.08) due to the absence of temporal modeling and noisy facial features. Static Action Unit aggregates lose most of the expressive dynamics.

**3\. Simple stacking fails.** The fused model F1 (0.38) is lower than the acoustic branch alone (0.60). Suggested improvements are attention-based fusion, cross-modal transformers, or learned gating — as implemented in Pipeline 2\.

**4\. Clinical utility of regression.** Although band accuracy is low (17%), within-1-band accuracy is 80%, meaning the regressor places patients within the correct or adjacent severity band 80% of the time — clinically actionable.

**5\. Scalability tradeoff.** Classical ML delivers fast, interpretable inference (\~50–100 ms). Foundation models offer richer representations but require GPU inference and 500 ms – 2 s latency.

---

# **14\. Conclusion**

This report documents the full methodology for the multimodal depression detection system:

The dataset is organized into clearly labeled raw source archives and preprocessed NumPy arrays, with separate files for each split and each label type. Preprocessing spans three distinct chains: session-level tabular feature extraction and branch slicing for Pipeline 1; waveform resampling, Whisper log-mel extraction, and BERT tokenization for Pipeline 2\.

The Pipeline 1 architecture (3-branch XGBoost \+ logistic stacking) is justified by the small dataset size (N=107) and the clinical need for interpretable outputs via SHAP. The Pipeline 2 architecture (frozen Whisper \+ frozen BioClinicalBERT \+ sigmoid gate \+ MLP head) is justified by the need for high-quality representations that cannot be learned from 107 samples from scratch.

Both pipelines were verified end-to-end: Pipeline 1 on the actual validation set, Pipeline 2 on a raw-input subset run confirming that wav → encoder → fusion → loss computes correctly.

The core finding is that speech carries the strongest depression signal, simple modality fusion is insufficient, and within-1-band PHQ-8 accuracy of 80% makes the system clinically useful despite modest classification F1 scores.

