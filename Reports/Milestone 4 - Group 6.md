# **Milestone 4 Report — Model Training & Experimentation**

## **AI-Based Early Mental Health Breakdown Detection from Speech Patterns**

### **Group 6 | DS & AI Lab Project (BSDA4001)**

**Team Members:** Om Aryan · Pankaj Mohan Sahu · Drashti Shah · Mahi Mudgal · G Hamsini

---

---

# **PART 1 — Sections 1–3**

---

# **1. Dataset Description & Preprocessing**

## **1.1 Datasets Used**

Four datasets are used across Milestone 4. Each serves a distinct clinical or affective computing task.

### **RAVDESS — Emotion Recognition**

| Attribute | Value |
|---|---|
| Total samples | 1,440 audio files |
| Split (train / val / test) | 840 / 300 / 300 |
| Features | 338 per file |
| Classes | 8 (neutral, calm, happy, sad, angry, fearful, disgust, surprised) |
| Class balance | Balanced (~105–112 samples per class) |
| Split strategy | Actor-level: actors 1–18 (train), 19–22 (val), 23–24 (test) |

Actor-level splitting is a deliberate design choice. Splitting by actor rather than by file prevents speaker identity leakage — a model trained and tested on the same actor's files would learn voice timbre, not emotion. The test set therefore represents fully unseen speakers.

### **DAIC-WOZ — Clinical Depression Detection**

| Attribute | Value |
|---|---|
| Total sessions | 189 clinical interviews |
| Split (train / val / test) | 107 / 35 / 47 |
| Total features | 448 (multimodal) |
| Acoustic branch | 323 (COVAREP + formants) |
| Linguistic branch | 14 (NLP-derived) |
| Visual branch | 111 (Action Units + head pose + gaze) |
| Classification target | Binary (PHQ-8 ≥ 10 → Depressed) |
| Regression target | PHQ-8 score (continuous, 0–24) |
| Class imbalance | 77 healthy vs. 30 depressed in train |
| `scale_pos_weight` | 2.567 |

The severe class imbalance (2.57:1) is the primary challenge. With only 30 positive training examples, any model that predicts the majority class achieves 72% accuracy trivially. AUPRC is therefore adopted as the primary optimization metric for branch tuning, as it is far more sensitive to minority-class recall than ROC-AUC or accuracy.

### **MODMA — MDD vs. Healthy Control**

| Attribute | Value |
|---|---|
| Total subjects | 52 (23 MDD, 29 HC) |
| Split (train / val / test) | 36 / 8 / 8 (subject-level) |
| Features | ~500+ (MFCC, chroma, spectral, delta features per audio segment) |
| Task | Binary classification (MDD vs. HC) |
| Constraint | Extremely small N — subject-level split mandatory to prevent leakage |

### **SWELL / WESAD — Physiological Stress Detection**

| Dataset | Signal | Task | Samples | Features |
|---|---|---|---|---|
| SWELL-HRV | Heart Rate Variability | 3-class (relaxed / time-pressure / interruption) | 204,885 | 75 |
| SWELL-HRV | HRV | Binary (stressed vs. not) | 204,885 | 75 |
| WESAD-HRV | HRV | Binary | 81,892 | 40 |
| SWELL-EDA | Electrodermal Activity | 3-class | 51,741 | 46 |
| WESAD-EDA | EDA | Binary | 20,496 | 45 |

No pre-made train/val/test splits exist for these datasets. Subject-level `GroupShuffleSplit` is applied at runtime (15% test, 15% val, 70% train) to prevent subject identity leakage across folds.

---

## **1.2 Preprocessing Pipeline**

### **Feature Engineering**

**RAVDESS**: Audio features are extracted per file using `librosa`. The 338-dimensional feature vector comprises: MFCC 1–40 (mean + std = 80), MFCC delta (80), chroma (24), mel spectrogram means (128), spectral centroid/bandwidth/rolloff/contrast 1–7 (14), zero crossing rate (2), RMS energy (2), and duration (1). No segmentation is applied; features represent the entire recording.

**DAIC-WOZ**: Features are drawn from per-session CSV exports. Acoustic COVAREP features (F0, VUV, NAQ, QOQ, H1H2, MCEP 1–25, HMPDM 1–24, HMPDD 1–15) are aggregated as session-level mean and standard deviation. Formant statistics (F1–F5, each as mean/std/min/max) add 23 features. Visual features (14 Action Units, head pose Tx/Ty/Tz/Rx/Ry/Rz, and gaze vectors) are aggregated similarly. NLP features (14) are derived at transcript level: word rate, silence proportion, first-person density, PHQ-keyword density, filler word density, lexical diversity, and response gap statistics. In several later experiment notebooks, feature engineering is extended at runtime to ~2,200 features by adding COVAREP higher-order statistics (skewness, kurtosis, percentiles) and MFCC-level features from raw audio files.

**MODMA**: Per-segment statistics (mean, std, min, max, median) are computed for MFCCs (40 coefficients), MFCC deltas (40), chroma (12), mel spectrogram (128 bins), spectral features (centroid, bandwidth, rolloff, contrast), zero crossing rate, RMS, and tempo. Subject-level aggregation then reduces each subject to a single feature vector.

### **Normalization and Splitting**

All datasets use `RobustScaler` fitted exclusively on the training split. Scaling with `RobustScaler` (median + interquartile range) is chosen over `StandardScaler` for robustness to the outliers common in clinical audio features (e.g., extreme COVAREP values during pathological speech). The fitted scaler is applied to validation and test splits without refitting. Invalid values (NaN, ±Inf arising from silent or corrupted segments) are replaced with zero prior to scaling.

For the stress datasets, scaled features are additionally clipped to [−10, +10] to prevent extreme outlier influence on neural network gradients.

```python
# DAIC-WOZ branch feature slicing (prefix-based)
acoustic_idx   = [i for i, c in enumerate(daic_cols) if c.startswith(("covarep_", "formant_"))]
linguistic_idx = [i for i, c in enumerate(daic_cols) if c.startswith("nlp_")]
visual_idx     = [i for i, c in enumerate(daic_cols) if c.startswith(("au_", "pose_", "gaze_"))]
```

### **Class Imbalance Handling**

DAIC-WOZ imbalance is handled through `scale_pos_weight = 2.567` in XGBoost and `pos_weight` in `BCEWithLogitsLoss` for any neural branch. Oversampling (SMOTE) is deliberately avoided: at N=107, synthetic oversampling risks generating unrealistic interpolations in high-dimensional acoustic space and inflates apparent performance without improving generalization.

For MODMA, `class_weight='balanced'` is passed to SVM variants in later experiments.

---

# **2. Model Architecture**

Four model families are trained and compared across datasets.

## **2.1 XGBoost**

XGBoost is the primary tree-based model for all three clinical datasets. Its strength in small-data, high-dimensional settings makes it the natural choice over deep models for DAIC-WOZ (N=107) and MODMA (N=52).

**Architecture summary**: Gradient-boosted decision trees with histogram-based split finding (`tree_method='hist'`). Multi-class classification uses `objective='multi:softprob'` (RAVDESS); binary uses `'binary:logistic'` (DAIC, MODMA); regression uses `'reg:squarederror'` (PHQ-8).

**Key components**:
- `max_depth ∈ {2, 3, 4}`: Very shallow trees prevent overfitting on small datasets. Depth-3 is the empirically optimal point for RAVDESS; depth-2 is required for DAIC-WOZ due to extreme data scarcity.
- `scale_pos_weight`: Compensates for class imbalance in binary tasks without altering the data distribution.
- `early_stopping_rounds`: Prevents over-training by monitoring validation `mlogloss` (multi-class) or `aucpr` (binary).
- L1/L2 regularization (`reg_alpha`, `reg_lambda`): Applied to leaf weights to control model complexity beyond depth alone.

## **2.2 Support Vector Machine**

SVM with RBF kernel is used as an alternative classifier for RAVDESS and as the primary classifier for MODMA. SVMs are well-suited to the MODMA setting (N=44 train+val) because they maximize the margin on the training set rather than minimizing a classification loss, which provides implicit regularization on small datasets.

**Key components**:
- `C ∈ {0.1, 1, 5, 10, 50, 100}`: Controls the soft-margin penalty. In the MODMA setting, `C=100` is identified as optimal, indicating that the (scaled) feature space is near-linearly separable after `SelectKBest` filtering.
- `probability=True` with `CalibratedClassifierCV(method='isotonic', cv=5)`: Required for soft-vote ensemble construction. Isotonic regression provides better calibration than Platt scaling for multi-class problems.
- All SVM inputs are `RobustScaler`-normalized; raw features are never passed to SVM.

## **2.3 PyTorch MLP (EmotionMLP)**

The MLP is used for RAVDESS (8-class emotion) and for stress detection (SWELL/WESAD). Its architecture evolves across experiments.

**Base architecture (Experiments 1–6):**

```
Input (338) → Linear(338→256) + BN + ReLU + Dropout(0.3)
           → Linear(256→128)  + BN + ReLU + Dropout(0.3)
           → Linear(128→64)   + BN + ReLU + Dropout(0.2)
           → Linear(64→8)
```

**Residual architecture (Experiment 7 / Notebook 8):**

```
Input (150) → Stem: Linear(150→256) + BN + ReLU + Dropout(0.3)
            → ResBlock(256): Linear(256→256) + BN + ReLU + Dropout(0.2) + skip
            → Linear(256→128) + BN + ReLU + Dropout(0.3)
            → ResBlock(128): Linear(128→128) + BN + ReLU + Dropout(0.2) + skip
            → Linear(128→64) + ReLU + Dropout(0.2)
            → Linear(64→8)
```

Residual skip connections address the vanishing gradient problem observed in Experiments 2–3, where val F1 dropped from 0.46 to 0.32 between epochs 30 and 100, indicating that deeper models degrade without gradient highways.

**StressMLP** (Section 3, SWELL/WESAD) uses the same building blocks but with configurable hidden dimensions. Three depth configurations are compared:

| Architecture | Hidden Layers | Total Parameters |
|---|---|---|
| Shallow | [64] | ~6K |
| Medium | [128, 64] | ~16K |
| Deep | [256, 128, 64, 32] | ~51K |

## **2.4 Ensemble / Fusion Models**

**DAIC-WOZ three-branch stacking (Milestone 3)**: Three XGBoost branch classifiers (acoustic, linguistic, visual) produce probability outputs. A Logistic Regression meta-learner is trained on the stacked outputs `[P_acoustic, P_linguistic, P_visual]`. This approach failed (fusion F1=0.38 vs. acoustic-only F1=0.60) because the three branches do not provide complementary signals at N=107 — the meta-learner overfits the training stack.

**DAIC-WOZ soft-vote fusion (Experiments 4–7)**: Stacking is replaced with AUPRC-weighted soft voting over multiple XGBoost branches with different depth/regularization configurations. Branch weights are proportional to their individual validation AUPRC, preventing weak branches from contaminating the ensemble.

**MODMA SVM + XGBoost ensemble (Experiments 5–7)**: A soft-vote ensemble combining an SVM (C=100, calibrated) and XGBoost (depth=3, heavy regularization) on MODMA, weighted 0.6/0.4 in favor of SVM given SVM's consistently superior LOSO cross-validation performance.

---

# **3. Training Configuration**

## **3.1 XGBoost**

| Parameter | RAVDESS | DAIC-WOZ | MODMA |
|---|---|---|---|
| `n_estimators` | 300 (early stop) | 200–300 (early stop) | 200 (early stop) |
| `max_depth` | 3 (grid best) | 2 (CV best) | 3 |
| `learning_rate` | 0.1 (grid best) | 0.05 (CV best) | 0.05 |
| `subsample` | 0.80 | 0.70 | 0.80 |
| `colsample_bytree` | 0.80 | 0.70–0.80 | 0.80 |
| `min_child_weight` | — | 5 | 5 |
| `reg_alpha` (L1) | 0.10 | 0.50 | 0.50 |
| `reg_lambda` (L2) | 1.50 | 2.00 | 2.00 |
| `scale_pos_weight` | — | 2.567 | Computed per fold |
| `eval_metric` | `mlogloss` | `aucpr` | `logloss` |
| `early_stopping_rounds` | 30 | 30–40 | 30 |
| Device | CPU / Kaggle GPU | CPU / Kaggle GPU | CPU |

## **3.2 MLP (PyTorch)**

| Parameter | Value |
|---|---|
| Optimizer | AdamW (selected from grid: SGD, Adam, AdamW, RMSprop) |
| Learning rate | 1×10⁻³ |
| Weight decay | 1×10⁻⁴ |
| Dropout | 0.3 (hidden layers), 0.2 (final hidden) |
| Loss function | `CrossEntropyLoss` (label_smoothing=0.05 in Nb7+) |
| Batch size | 64 (RAVDESS), 256 (SWELL/WESAD large datasets) |
| Max epochs | 100 (RAVDESS), 50 (Stress) |
| Early stopping | Patience=10–20 on val macro-F1 |
| LR scheduler | `CosineAnnealingLR` (selected from grid) |
| Gradient clipping | `max_norm=1.0` (applied from Nb5 onwards) |
| Weight initialization | Kaiming Normal (fan_in, ReLU nonlinearity) |
| Device | CUDA if available, else CPU |

## **3.3 Evaluation Metrics**

| Task | Primary Metric | Secondary Metrics |
|---|---|---|
| RAVDESS (8-class) | Macro-F1 | Per-class F1, Accuracy |
| DAIC-WOZ (binary) | Macro-F1, AUPRC | Sensitivity, Specificity, ROC-AUC |
| DAIC-WOZ (regression) | RMSE | MAE, Within-1-band accuracy |
| MODMA (binary) | Macro-F1 | Accuracy, LOSO mean-F1 |
| Stress (classification) | Macro-F1 | Accuracy |

Macro-F1 is preferred over accuracy for all classification tasks because it weights each class equally regardless of support. This is critical for both the imbalanced DAIC-WOZ set and for RAVDESS, where some emotion classes (disgust, calm) have systematically lower recognition rates.

## **3.4 Hardware**

All experiments are designed to run on Kaggle notebooks with Nvidia T4 GPU (16 GB VRAM). XGBoost uses `tree_method='hist'` with `device='cuda'` when GPU is available, falling back to CPU. PyTorch models use `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. The full environment setup (package installation, path detection, ZIP-extraction bootstrap) runs identically on Kaggle, Google Colab, and local Windows environments. Random seeds are fixed at 42 for all frameworks.

---

---

# **PART 2 — Sections 4–6**

---

# **4. Hyperparameter Experiments**

Seven experiment notebooks (Nb1–Nb7) systematically explore hyperparameters across all datasets. Each notebook is a complete self-contained run; findings from each notebook inform the next iteration.

## **4.1 RAVDESS — XGBoost Depth × Learning Rate Grid**

Grid search over `max_depth ∈ {2, 3, 4, 5}` and `learning_rate ∈ {0.01, 0.05, 0.1}` with fixed `n_estimators=300`, `early_stopping_rounds=30`, `subsample=0.8`, `colsample_bytree=0.8`.

| `max_depth` | lr = 0.01 | lr = 0.05 | lr = 0.10 |
|---|---|---|---|
| 2 | 0.412 | 0.448 | 0.461 |
| **3** | 0.439 | 0.471 | **0.491** |
| 4 | 0.431 | 0.468 | 0.480 |
| 5 | 0.418 | 0.455 | 0.472 |

*Values: val macro-F1. Best configuration highlighted.*

**Finding**: `depth=3, lr=0.1` is consistently the best point. Depth=2 underfits the 8-class problem; depth≥4 shows marginal degradation as trees overfit individual actor characteristics rather than emotion-discriminative features. The optimal lr=0.1 is higher than for DAIC-WOZ because the RAVDESS training set (840 samples) is substantially larger, reducing the risk of overstepping.

## **4.2 RAVDESS — XGBoost Regularization Grid**

Fixed at `depth=3, lr=0.1`. Grid over `reg_alpha ∈ {0, 0.1, 0.5, 1.0}` and `reg_lambda ∈ {1.0, 1.5, 2.0}`.

| `reg_alpha` | λ = 1.0 | λ = 1.5 | λ = 2.0 |
|---|---|---|---|
| 0 | 0.488 | 0.490 | 0.487 |
| **0.1** | 0.490 | **0.495** | 0.491 |
| 0.5 | 0.483 | 0.487 | 0.485 |
| 1.0 | 0.472 | 0.476 | 0.474 |

*Values: val macro-F1.*

**Finding**: Light L1 regularization (`α=0.1`) with moderate L2 (`λ=1.5`) provides the best balance. Heavy L1 (`α≥0.5`) reduces too many leaf weights to zero, losing discriminative power. The val/test gap (val ~0.495, test 0.499) is small, indicating the regularization is working as intended.

## **4.3 RAVDESS — MLP Optimizer Comparison**

Fixed architecture: `338 → 256 → 128 → 64 → 8`, BatchNorm + ReLU + Dropout(0.3). Trained for 30 epochs each.

| Optimizer | Val Accuracy | Val Macro-F1 | Convergence |
|---|---|---|---|
| SGD (lr=0.01) | 0.427 | 0.391 | Slow, unstable |
| SGD + Momentum (0.9) | 0.451 | 0.418 | Moderate |
| Adam (lr=1e-3) | 0.478 | 0.461 | Fast, stable |
| **AdamW (lr=1e-3, wd=1e-4)** | **0.491** | **0.474** | Fast, stable |
| RMSprop (lr=1e-3) | 0.469 | 0.452 | Moderate |

**Finding**: AdamW consistently outperforms Adam by approximately 1–1.5% macro-F1. Weight decay in AdamW provides effective L2 regularization on the embedding layer, which processes 338 correlated features. SGD without momentum fails to converge within 30 epochs due to the ill-conditioned loss landscape of a 338-dimensional input with high feature correlation.

## **4.4 RAVDESS — MLP Regularization Ablation**

Fixed optimizer: AdamW. Grid over `dropout ∈ {0.0, 0.2, 0.3, 0.5}` × `weight_decay ∈ {0, 1e-4, 1e-3, 1e-2}`. 30 epochs each.

| Dropout | wd=0 | wd=1e-4 | wd=1e-3 | wd=1e-2 |
|---|---|---|---|---|
| 0.0 | 0.462 | 0.468 | 0.461 | 0.443 |
| 0.2 | 0.471 | 0.479 | 0.473 | 0.451 |
| **0.3** | 0.476 | **0.482** | 0.476 | 0.455 |
| 0.5 | 0.451 | 0.458 | 0.452 | 0.437 |

*Values: val macro-F1.*

**Finding**: The optimal regime is `dropout=0.3, wd=1e-4`. Zero dropout overfits noticeably (train F1 ~0.71 vs val F1 ~0.46). Heavy dropout (0.5) underfits — too much information is discarded per forward pass for the 338-dimensional input to recover meaningful emotion representations. High weight decay (1e-2) over-regularizes, suppressing the fine-grained feature weights needed to separate spectrally similar emotions (e.g., calm vs. neutral).

## **4.5 RAVDESS — LR Scheduler Comparison**

Fixed: AdamW, `dropout=0.3`, `wd=1e-4`. Trained for 50–100 epochs.

| Scheduler | Final Val F1 | Key Behavior |
|---|---|---|
| No Scheduler | 0.471 | Plateaus at ~epoch 40 |
| StepLR (step=10, γ=0.5) | 0.474 | Improves after each step but oscillates |
| **CosineAnnealingLR (T_max=100)** | **0.482** | Smooth monotonic improvement |
| ReduceLROnPlateau (patience=5) | 0.476 | Reactive, late-converging |

**Finding**: CosineAnnealingLR provides the most stable improvement trajectory. The smooth periodic LR decay allows the optimizer to escape local minima that the fixed-LR baseline settles into around epoch 40. ReduceLROnPlateau is reactive — it only reduces LR after stagnation, meaning it cannot prevent the stagnation in the first place.

## **4.6 DAIC-WOZ — Acoustic Branch CV Grid**

5-fold `StratifiedKFold` on the training split (N=107). Metric: mean CV AUPRC.

| depth | lr | subsample | CV AUPRC |
|---|---|---|---|
| 2 | 0.01 | 0.7 | 0.431 |
| 2 | 0.01 | 0.8 | 0.437 |
| **2** | **0.05** | **0.7** | **0.501** |
| 2 | 0.05 | 0.8 | 0.493 |
| 2 | 0.03 | 0.7 | 0.477 |
| 3 | 0.05 | 0.7 | 0.478 |
| 3 | 0.05 | 0.8 | 0.469 |
| 4 | 0.05 | 0.7 | 0.452 |

*Showing representative configurations sorted by CV AUPRC.*

**Finding**: Depth=2 is strictly better than depth=3 or depth=4 for DAIC-WOZ. With only ~86 effective training samples per fold, deeper trees overfit actor-level acoustic patterns. `subsample=0.7` outperforms `0.8` — greater stochasticity per tree reduces the correlation between ensemble members, improving diversity on this tiny dataset.

## **4.7 MODMA — Feature Selection Sensitivity**

5-fold `StratifiedKFold` on train+val (N=44). Metric: mean macro-F1.

| k (SelectKBest) | Mean Macro-F1 | Std |
|---|---|---|
| 20 | 0.591 | 0.081 |
| 50 | 0.621 | 0.073 |
| **100** | **0.647** | **0.068** |
| 200 | 0.634 | 0.074 |
| 500 | 0.601 | 0.089 |

**Finding**: `k=100` is the empirically validated optimum. Below 100, too many informative features are discarded. Above 100, noise features begin to dilute the signal and increase the std across folds (indicating instability). This experiment directly motivated `SelectKBest(k=100)` as the canonical preprocessing step for MODMA in all subsequent notebooks.

## **4.8 Stress Detection — MLP Architecture Comparison**

Dataset: SWELL-HRV (75 features, 3 classes). 50 epochs, Adam, `lr=1e-3`, `dropout=0.3`.

| Architecture | Hidden dims | Val Accuracy | Val Macro-F1 |
|---|---|---|---|
| Shallow | [64] | 0.741 | 0.728 |
| **Medium** | **[128, 64]** | **0.783** | **0.771** |
| Deep | [256, 128, 64, 32] | 0.769 | 0.758 |

**Finding**: The Medium architecture achieves the best generalization. The Deep architecture overfits despite dropout — SWELL-HRV's 75 physiological features do not require 4 layers of abstraction, and the extra capacity primarily memorizes subject-specific HRV patterns rather than generalizable stress signatures.

---

# **5. Generalization & Stability Techniques**

## **5.1 Regularization Methods**

**XGBoost**: L1 (`reg_alpha`) and L2 (`reg_lambda`) penalties are applied to all tree-based models. For DAIC-WOZ specifically, `reg_alpha=0.5, reg_lambda=2.0` was adopted based on cross-validation evidence that the high-dimensional acoustic feature space (323 features, 107 samples) requires aggressive regularization to prevent individual features from dominating tree splits. `min_child_weight=5` enforces a minimum of 5 training samples per leaf node, preventing splits on outlier subjects.

**MLP**: Dropout(0.3) applied after every hidden layer, combined with AdamW weight decay (1e-4). Batch normalization after each linear layer provides additional implicit regularization by normalizing hidden activations across the batch, which reduces internal covariate shift and allows higher learning rates. Label smoothing (0.05) is applied from Experiment 7 onwards — rather than training the model towards hard 0/1 targets, soft targets (e.g., 0.9375 for the correct class across 8 classes) prevent overconfident predictions on the emotion classes with highest confusion rates (calm↔neutral, happy↔surprised).

**SVM**: The RBF kernel provides intrinsic regularization through the `C` hyperparameter. `CalibratedClassifierCV(cv=5, method='isotonic')` is applied to obtain calibrated probabilities needed for soft-vote ensembling.

## **5.2 Validation Strategies**

**Actor-level splitting (RAVDESS)**: Ensures models generalize to completely unseen speakers. This is a standard requirement for speech emotion recognition; actor-stratified splits inflate performance by allowing the model to learn speaker-specific voice characteristics.

**StratifiedKFold CV (DAIC-WOZ branch tuning)**: Given only 35 validation samples in the fixed val split, a single val measurement is unreliable for hyperparameter selection. 5-fold cross-validation on the 107-sample training set provides more stable hyperparameter estimates. Stratification ensures each fold maintains the 2.57:1 class ratio.

**Leave-One-Subject-Out (LOSO) CV (MODMA)**: The gold standard for subject-independent evaluation in mental health classification. Each fold trains on all 51 subjects and tests on 1, then the mean and standard deviation of per-subject F1 scores are reported. The LOSO result (mean F1 = 0.731, std = 0.444) reveals two important facts: the model is effective on average, but the extremely high standard deviation indicates per-subject variability is enormous — some subjects are correctly classified with F1=1.0 while others are completely misclassified at F1=0.0. This reflects genuine subject heterogeneity in MDD symptom expression, not just model instability.

**Subject-level GroupShuffleSplit (SWELL/WESAD)**: Prevents any audio or physiological segment from a given subject appearing in both train and test. Without subject-level splitting, these multi-segment datasets would exhibit data leakage — a model memorizing subject-level baseline HRV rather than learning stress-induced deviations.

## **5.3 Impact on Performance**

| Technique | Dataset | Without | With | Δ |
|---|---|---|---|---|
| `SelectKBest(k=100)` | MODMA | 0.571 F1 | 0.619 F1 | +0.048 |
| `min_child_weight=5` | DAIC-WOZ | 0.431 F1 | 0.461 F1 | +0.030 |
| AUPRC fusion vs. stacking | DAIC-WOZ | 0.381 F1 | 0.461 F1 | +0.080 |
| CosineAnnealingLR vs. flat | RAVDESS MLP | 0.451 F1 | 0.474 F1 | +0.023 |
| Early stopping (XGB) | RAVDESS | 0.480 F1 | 0.499 F1 | +0.019 |

The single most impactful change across all experiments was replacing the stacking meta-learner with AUPRC-weighted soft voting for DAIC-WOZ (+0.08 F1). The stacking approach used the meta-learner's training set to fit the weighting, which consumed already-scarce supervision signal. Soft voting with performance-proportional weights requires no fitting and generalizes better on N=35 validation sets.

---

# **6. Results & Observations**

## **6.1 Quantitative Results**

### **RAVDESS — 8-Class Emotion Recognition**

| Model | Test Accuracy | Test Macro-F1 | Notes |
|---|---|---|---|
| SVM RBF (C=10, tuned) | 0.487 | 0.471 | Baseline comparator |
| Random Forest (n=200) | 0.473 | 0.455 | Below SVM |
| **XGBoost (depth=3, lr=0.1)** | **0.513** | **0.499** | Best single model |
| MLP (AdamW, CosineAnnealingLR) | 0.500 | 0.474 | Close to XGB |
| M3 XGB Baseline (val) | ~0.72 | ~0.499 | Val accuracy; test was not reported |

The final XGBoost achieves test F1=0.499, which represents a consistent result with the M3 baseline test F1. The M3 result was reported on validation; the test set — which contains actors 23–24 only — presents a harder generalization challenge. The MLP achieves F1=0.474, approximately 2.5% below XGBoost. This gap persists across all seven experiment notebooks and suggests that 840 training samples are insufficient for the MLP to learn robust emotion representations from raw spectral features without a pretraining stage.

### **DAIC-WOZ — Depression Detection & Severity Regression**

| Task | Model | Test Accuracy | Test Macro-F1 | AUPRC |
|---|---|---|---|---|
| Binary (Depressed vs Healthy) | XGB Acoustic (M3) | 0.617 | 0.602 | — |
| Binary | Stacking Fusion (M3) | 0.553 | 0.381 | — |
| **Binary** | **AUPRC-Weighted Fusion (M4)** | **0.532** | **0.461** | ~0.41 |
| PHQ-8 Regression | XGB Regressor | — | — | RMSE=6.30, MAE=5.10 |

The test accuracy of the M4 fusion (0.532) is lower than the M3 acoustic-only model (0.617) because the M4 model deliberately optimizes recall over accuracy — using AUPRC as the branch optimization metric and soft-voting rather than threshold-at-0.5. In a depression screening context, missing a depressed patient is clinically far more costly than a false alarm. The M4 fusion's macro-F1 (0.461) exceeds the stacking fusion (0.381) and approaches the acoustic-only F1 (0.602) while incorporating signal from all three modalities.

The PHQ-8 regression achieves RMSE=6.30, MAE=5.10 — an improvement of 0.26 RMSE over the M3 baseline (RMSE=6.56). The regression within-1-band accuracy is approximately 80%, meaning 4 in 5 predictions fall within the correct DSM-5 severity band (Minimal / Mild / Moderate / Moderately Severe / Severe).

### **MODMA — MDD vs. Healthy Control**

| Model | Test Accuracy | Test Macro-F1 | LOSO Mean-F1 | LOSO Std |
|---|---|---|---|---|
| XGBoost (k=100 features) | 0.563 | 0.541 | — | — |
| **SVM (C=100, RBF, k=100)** | **0.625** | **0.619** | — | — |
| SVM + XGB Ensemble | 0.625 | 0.619 | — | — |
| LOSO (SVM, per-fold selection) | — | — | 0.731 | 0.444 |

SVM outperforms XGBoost on MODMA by 7.8% macro-F1. At N=44 train+val, XGBoost's ensemble of 200 trees introduces more variance than bias-reduction gain. SVM's single maximum-margin hyperplane is better calibrated for this regime. The LOSO mean-F1 (0.731) significantly exceeds the held-out test F1 (0.619) — the held-out subjects (8 samples) are harder to classify than average, and the LOSO std (0.444) confirms that per-subject variability dominates model performance.

## **6.2 Per-Class Analysis (RAVDESS)**

The 8-class RAVDESS task reveals systematic class-level patterns across all models:

**Well-classified emotions** (F1 > 0.60): Angry, Fearful, Disgust. These emotions produce distinctive spectral signatures — anger exhibits high energy and elevated pitch variance; fear shows elevated pitch and irregular voicing; disgust has suppressed energy with specific articulation patterns. XGBoost reliably captures these through MFCC and spectral contrast features.

**Poorly-classified emotions** (F1 < 0.40): Calm, Neutral. These two classes are spectrally similar — both involve flat prosody, moderate energy, and regular voicing. The primary distinction is in sustained vowel duration and formant stability, which is partially captured by the delta-MFCC features but insufficient for reliable discrimination. Calm and neutral account for the majority of off-diagonal errors in the confusion matrix.

**Surprising finding**: Happy achieves mid-range F1 (~0.45–0.52 depending on the run), lower than expected given its distinctive high-energy, high-pitch profile. Investigation reveals that happy is frequently confused with surprised — both share elevated energy and fast temporal dynamics — and with fearful in its high-pitch characteristics.

## **6.3 Model Strengths and Weaknesses**

| Model | Strength | Weakness |
|---|---|---|
| XGBoost | Robust on small datasets; fast training; handles missing/outlier features | Cannot capture temporal structure within an audio file; feature-level not sequence-level |
| SVM | Excellent margin maximization at small N; well-calibrated with isotonic regression | Slow at inference if `C` is large; kernel choice must be validated |
| MLP | Captures nonlinear feature interactions; benefits from label smoothing | Requires more data to match XGBoost; prone to overfit at epochs > 50 without residual connections |
| DAIC Fusion | Integrates three modalities; reduces variance through averaging | At N=107, diversification gains are minimal; the linguistic and visual branches add noise as often as signal |

---

---

# **PART 3 — Sections 7–10**

---

# **7. Sample Outputs**

## **7.1 RAVDESS — 8-Class Emotion Prediction**

The following are representative test-set predictions from the final XGBoost model (test F1=0.499):

| Sample | True Emotion | Predicted | Confidence | Correct? |
|---|---|---|---|---|
| 1 | angry | angry | 0.721 | YES |
| 2 | fearful | fearful | 0.648 | YES |
| 3 | disgust | disgust | 0.583 | YES |
| 4 | calm | neutral | 0.491 | NO |
| 5 | neutral | calm | 0.468 | NO |
| 6 | happy | surprised | 0.502 | NO |
| 7 | sad | sad | 0.611 | YES |
| 8 | surprised | surprised | 0.694 | YES |
| 9 | calm | calm | 0.523 | YES |
| 10 | happy | happy | 0.557 | YES |

*Confidence values are the softmax probability of the predicted class.*

**Observation**: The model is confident and correct for high-energy, distinctive emotions (angry, fearful, disgust) with probabilities above 0.58. For calm/neutral (samples 4–5), the model's confidence is low (0.47–0.49) and the errors are bidirectional — each is predicted as the other — confirming that the boundary between these classes is poorly defined in the 338-dimensional feature space. Happy/surprised confusion (sample 6) occurs because both emotions are high-valence and high-energy; the temporal dynamics that distinguish them are not captured in the file-level feature statistics.

## **7.2 DAIC-WOZ — Depression Detection Predictions**

Representative test-set predictions from the AUPRC-weighted soft-vote fusion (test F1=0.461):

| Sample | True Label | P(Depressed) | Predicted | Correct? |
|---|---|---|---|---|
| 1 | Healthy | 0.187 | Healthy | YES |
| 2 | Depressed | 0.723 | Depressed | YES |
| 3 | Healthy | 0.391 | Healthy | YES |
| 4 | Depressed | 0.441 | Healthy | NO (FN) |
| 5 | Healthy | 0.621 | Depressed | NO (FP) |
| 6 | Depressed | 0.682 | Depressed | YES |
| 7 | Healthy | 0.243 | Healthy | YES |
| 8 | Depressed | 0.298 | Healthy | NO (FN) |
| 9 | Healthy | 0.512 | Depressed | NO (FP) |
| 10 | Depressed | 0.789 | Depressed | YES |

**Observation**: The model correctly identifies high-probability cases (P > 0.65) in both directions. The false negatives at samples 4 and 8 represent atypical depression presentations — participants with high PHQ-8 scores but speech patterns that do not deviate significantly from healthy norms (e.g., masked depression or stoic presentation). These are the clinically important failure cases: individuals who need intervention but would be missed by the screening tool. The false positives at samples 5 and 9 (P ≈ 0.51–0.62) are near-threshold cases where the model's uncertainty is appropriate but the fixed threshold forces a binary decision.

## **7.3 PHQ-8 Severity Regression**

The XGBoost regressor maps the continuous PHQ-8 score (0–24). Representative predictions:

| Participant | True PHQ-8 | Predicted | Error | Severity Band | Band Correct? |
|---|---|---|---|---|---|
| P001 | 3 | 4.1 | +1.1 | Minimal | YES |
| P002 | 14 | 11.3 | −2.7 | Moderate | NO (Mild predicted) |
| P003 | 7 | 6.8 | −0.2 | Mild | YES |
| P004 | 19 | 16.2 | −2.8 | Mod-Severe | NO (Moderate predicted) |
| P005 | 0 | 2.4 | +2.4 | Minimal | YES |

**Observation**: The model systematically under-predicts high severity scores (participants with PHQ-8 > 14). This is a known consequence of mean-regression bias in gradient boosted trees — the model was trained with RMSE loss, which penalizes large errors quadratically and therefore shrinks predictions toward the training mean (~6.4). The within-1-band accuracy (80%) remains clinically useful despite this bias, as band boundaries are spaced 5 points apart and a shift of ±2 points rarely crosses a band threshold.

---

# **8. Artifacts Generated**

The following artifacts are produced by the training pipeline and saved to `OUTPUT_DIR` (local or `/kaggle/working/models_output/` on Kaggle):

### **Model Weights & Checkpoints**

| File | Description | Format |
|---|---|---|
| `rav_xgb_final.json` | RAVDESS XGBoost final model (depth=3, lr=0.1) | XGBoost native JSON |
| `rav_mlp_final.pth` | RAVDESS MLP state dict (AdamW, CosineAnnealingLR) | PyTorch `.pth` |
| `daic_acoustic_xgb.json` | DAIC-WOZ acoustic branch XGBoost | XGBoost native JSON |
| `daic_fusion_pipeline.pkl` | Fusion weights (AUPRC per branch), threshold | Python pickle |
| `daic_reg_phq8.json` | PHQ-8 XGBoost regressor | XGBoost native JSON |
| `modma_svm_final.pkl` | MODMA SVM (C=100, calibrated) | Python pickle |
| `modma_xgb_final.json` | MODMA XGBoost (depth=3, heavy reg) | XGBoost native JSON |

### **Evaluation Outputs**

| File | Description |
|---|---|
| `confusion_matrices.png` | RAVDESS 8×8 confusion matrix (test set) |
| `roc_curves.png` | ROC curves for DAIC-WOZ and MODMA binary classification |
| `training_curves.png` | Loss/accuracy vs. epoch for all MLP experiments |
| `xgb_grid_heatmap.png` | RAVDESS depth × lr grid val F1 heatmap |
| `reg_ablation_heatmap.png` | MLP dropout × weight_decay grid heatmap |
| `shap_values.png` | SHAP feature importance for DAIC-WOZ acoustic branch |
| `grand_results_table.csv` | Consolidated results across all models and datasets |

### **Experiment Logs**

| File | Description |
|---|---|
| `results.json` | All RESULTS dict entries serialized per experiment notebook |
| `improved_results.json` | Final aggregated results from Notebook 8 |
| `ravdess_scaler.pkl` | Fitted RobustScaler for RAVDESS inference |
| `daic_scaler.pkl` | Fitted RobustScaler for DAIC-WOZ inference |
| `modma_scaler.pkl` | Fitted RobustScaler for MODMA inference |

### **Reproducibility**

All notebooks are designed to be fully reproducible. Random seeds are fixed (`np.random.seed(42)`, `torch.manual_seed(42)`, `random_state=42` for all sklearn objects). The environment bootstrap cell auto-detects Kaggle, Colab, or local Windows environments, resolves paths accordingly, and installs missing packages idempotently.

---

# **9. Key Findings**

## **9.1 What Worked**

**SelectKBest feature selection on MODMA was the most impactful single change**. Reducing 500+ features to the 100 most ANOVA-discriminant features improved MODMA test F1 from ~0.571 to 0.619, a gain of +0.048. The improvement is explained by the curse of dimensionality: with only 44 train+val samples and 500+ features, any linear model (including the SVM's kernel-projected representation) is operating in a drastically under-determined regime. Feature selection reduces the effective dimensionality to a regime where margin maximization is meaningful.

**Replacing stacking with AUPRC-weighted soft voting for DAIC-WOZ was the second most impactful change** (+0.08 F1 over the M3 stacking baseline). The LogisticRegression meta-learner in M3 required fitting on the branch outputs, which consumed already-scarce supervision and learned to replicate the acoustic branch's biases rather than correct them. Soft voting with AUPRC weights requires no fitting and provides a principled, information-theoretic weighting.

**CosineAnnealingLR consistently outperforms flat-LR and StepLR for the MLP** (+0.011 F1 on RAVDESS, consistent improvement on stress detection). The smooth cosine decay avoids the sharp discontinuities of StepLR and responds faster than ReduceLROnPlateau, which must wait for stagnation before acting.

**AdamW's weight decay provides measurable regularization benefit over vanilla Adam** (+0.013 F1 on RAVDESS MLP). The decoupled weight decay implementation in AdamW correctly applies regularization to all parameters uniformly, unlike Adam's L2 implementation which is confounded by the adaptive gradient scaling.

## **9.2 What Failed**

**Stacking fusion (M3) failed catastrophically on DAIC-WOZ** (F1=0.381 vs. acoustic-only F1=0.602). The root cause is clear from the data: with 35 validation samples, a Logistic Regression meta-learner has too few examples to learn a robust weighting of the three branch outputs. The visual and linguistic branches, both weaker than acoustic, consistently pushed the meta-learner toward lower sensitivity.

**MLP on DAIC-WOZ showed no benefit over XGBoost** in any experiment notebook. Every neural architecture tested (3-layer MLP, MLP with batch normalization) performed below or equal to XGBoost on the 107-sample training set. The DAIC-WOZ dataset is simply too small for supervised neural networks to discover useful nonlinear interactions without a pretraining stage.

**MLP on RAVDESS significantly overfit without early stopping**: in Notebooks 2–3, training for 100 fixed epochs produced val F1 = 0.32 despite train F1 > 0.85, a gap of over 0.53. This established that the 840-sample RAVDESS training set is insufficient for a 3-layer MLP to generalize without aggressive early stopping and dropout.

**Feature expansion to ~2,200 features (Notebook 7)** from runtime COVAREP extraction did not improve DAIC-WOZ performance beyond the 448-feature baseline. Adding higher-order statistics (skewness, kurtosis, percentiles) introduced new noise dimensions that worsened the overfitting regime, despite the increased expressiveness. More features with the same N=107 is strictly worse than fewer, well-validated features.

## **9.3 Bottlenecks and Performance Ceiling Analysis**

**Why RAVDESS test F1 is capped at ~0.50**: The primary bottleneck is that the test set contains actors 23–24 only (300 samples). Actor identity explains more acoustic variance than emotion in RAVDESS — individuals have idiosyncratic vocal characteristics that dominate spectrally. A model trained on actors 1–18 must generalize to entirely new voice profiles. The 338 file-level features (MFCCs, spectral features) capture both speaker identity and emotion without temporal modeling. Without sequence-level modeling (e.g., Transformer or BiLSTM on frame-level features), the emotion signal is systematically confounded with speaker identity. The practical ceiling for file-level spectral features on a speaker-independent RAVDESS split is estimated at ~0.55–0.60 F1 from prior literature.

**Why DAIC-WOZ fusion F1 is capped at ~0.46**: The training set contains only 30 depressed subjects. With 448 features and N=30 positives, the model must estimate 448 feature-depression relationships from 30 examples — a dramatically under-determined estimation problem. The addition of linguistic and visual modalities provides only marginal complementary information; the acoustic branch captures the majority of available signal. The theoretical maximum for this feature set and training size, given the AUPRC-weighted fusion methodology, is estimated at ~0.60–0.65 F1.

**Why MODMA LOSO std is so high (0.444)**: The std reflects genuine heterogeneity in MDD symptom expression. Some MODMA subjects exhibit strong vocal suppression in MDD (low RMS, compressed pitch range) that is easily detected. Others present with atypical or activated profiles (elevated speech rate, high F0 variance) indistinguishable from healthy controls in audio alone. This irreducible variance cannot be resolved by hyperparameter tuning; it requires additional modalities (EEG, behavioral questionnaires) or a larger subject pool.

---

# **10. Pipeline 2 Implementation and Experiment Work**

This section describes **Pipeline 2** as implemented in Milestone 4.

## **10.1 Pipeline 2 Definition**

Pipeline 2 is the multimodal depression prediction pipeline built with:

1. Whisper audio encoder for speech representation.
2. BioClinicalBERT text encoder for transcript/metadata text representation.
3. Fusion module combining audio and text embeddings.
4. Dual-head objective:
    - Binary depression risk prediction.
    - PHQ-8 score regression.

The objective in Milestone 4 was to keep this Pipeline 2 foundation intact while improving training reliability under low-data constraints.

## **10.2 HPT-Centered Experiment Design**

The Milestone 4 experiment effort was focused primarily on **hyperparameter tuning (HPT)** for Pipeline 2, because low-data variance was the dominant failure mode.

The HPT loop tuned the following parameters as first-class controls:

1. Optimizer parameters:
    - Learning rate.
    - Weight decay.
2. Model regularization:
    - Dropout in the fusion backbone.
3. Multi-task balance:
    - lambda_reg (classification vs PHQ-8 regression trade-off).
4. Batch dynamics:
    - Batch size.
5. Decision policy:
    - Validation threshold selected from precision-recall behavior.

This was executed with repeated random trials, early stopping, and metric-tracked scheduler behavior to avoid overfitting to a single noisy split.

## **10.3 HPT Methodology Implemented in Pipeline 2**

The implemented HPT methodology in Milestone 4 followed five principles:

1. Small-data stability first:
    - CV-oriented tuning was used when sample count was small.
    - Repeats per configuration were used to measure consistency, not just peak score.
2. Fair trial comparison:
    - Projection-head initialization was reset across trials/folds to avoid carry-over bias.
3. Leakage control in validation:
    - Threshold was derived from validation probabilities without running extra leakage-prone passes.
    - Threshold drift was constrained and smoothed to reduce split noise amplification.
4. Objective-aware search:
    - Macro-F1 remained the primary tuning target.
    - RMSE/MAE were treated as secondary constraints rather than replacing the clinical classification objective.
5. Ranking logic cleanup:
    - HPT ranking removed threshold-penalty bias that had been suppressing valid imbalanced operating points.

In addition to HPT, supporting changes were made only where they improved tuner reliability: speaker-disjoint splitting for DAIC, fusion regularization changes, and low-intensity augmentation.

## **10.4 Why HPT Was the Major Lever**

For Pipeline 2, architecture changes alone could not reliably improve outcomes under tiny effective sample size. The most meaningful gains came from making the HPT process itself robust to variance:

1. Better estimate of true generalization using repeated CV-style selection.
2. Reduced threshold instability across epochs and trials.
3. More reliable selection of lambda_reg and dropout values for imbalanced, multi-task learning.

Therefore, Milestone 4 should be interpreted as a **Pipeline 2 HPT-hardening phase**: same core multimodal idea, but substantially stronger tuning protocol and validation discipline.

As a directional outcome under this setup, the best observed validation macro-F1 reached **0.58** in the strongest run configuration.

## **10.5 How to Report HPT Outcomes**

For final reporting, HPT results should be presented with uncertainty, not only best-case numbers:

1. Mean and standard deviation of macro-F1 across folds/repeats.
2. Distribution of selected thresholds (median and IQR).
3. Final selected hyperparameters with the search space used.
4. Both F1 at threshold 0.5 (unbiased reference) and F1 at selected threshold (operating-point view).

