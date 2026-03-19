#  AI-Based Early Mental Health Breakdown Detection from Speech Patterns

> **Course:** Data Science and AI Lab Project (BSDA4001)
> **Milestone:** 1
> **Group:** 6 | Indian Institute of Technology Madras

---

## Team Members

| Name | Roll Number |
|---|---|
| G Hamsini | 22f3000767 |
| Om Aryan | 21f3002286 |
| Drashti Shah | 22f2001483 |
| Pankaj Mohan Sahu | 21f2001203 |
| Mahi Mudgal | 21f3002602 |

---

##  Milestone 1 — Team Contributions

### G Hamsini — `22f3000767`
- **Abstract** — Drafted the project abstract summarizing the multimodal AI screening architecture, fusion approach, and ethical design goals.
- **Objectives** — Defined the four primary project objectives covering early screening, severity estimation, biomarker extraction, and clinical interpretability.
- **Model Architecture** *(co-contributed)* — Co-designed the overall architecture including the Input & Branch Processing pipeline (Whisper and BioClinical-ModernBERT branches) and the Signal Extraction & RAG-Grounded Reasoning layer.

---

### Om Aryan — `21f3002286`
- **Model Architecture** *(co-contributed)* — Co-designed the Gated Dual-Modal Fusion mechanism (Sigmoid-based gating, 512D shared space, Compact Classifier Head, and Weighted Cross-Entropy Loss strategy).
- **Literature Review** — Surveyed recent SOTA work on foundation models (Whisper, BioClinicalBERT), multimodal fusion strategies, and RAG-based clinical decision support; identified key gaps and justified the proposed approach.

---

### Drashti Shah — `22f2001483`
- **Dataset Research and Resources** — Identified and documented all datasets used in the pipeline: DAIC-WOZ (primary benchmark), MODMA (embedding alignment), RAVDESS (acoustic robustness), and the Contextual Stress Dataset (real-world generalization).
- **Project Roles** — Defined the four specialized team roles: Multimodal Audio Engineer, Clinical NLP & LLM Architect, Fusion & Optimization Lead, and MLOps & Clinical Validation Lead.

---

### Pankaj Mohan Sahu — `21f2001203`
- **Feasibility and Compute** — Assessed hardware constraints (Kaggle + Google Colab, T4 GPUs), model-level feasibility with frozen encoders, memory optimization strategies, transcript handling, and ethical safeguards.
- **Evaluation Metrics** — Designed the full evaluation framework across four dimensions: Classification metrics (Macro-F1, Sensitivity, AUPRC), Regression metrics (RMSE, Severity Band Accuracy), RAG evaluation (RAGAS + SHAP), and Long-Context & Robustness benchmarks (LOSO, Noise Robustness).

---

### Mahi Mudgal — `21f3002602`
- **Project Timeline** — Planned and documented the 6-phase project timeline spanning February 20 to April 16, covering Foundation & Alignment, Data Engineering, Architectural Engineering, Optimization & Training, Forensic Evaluation & RAG Integration, and Deployment & Clinical Reasoning.

---

## Contribution Summary

| Member | Roll Number | Sections |
|---|---|---|
| G Hamsini | 22f3000767 | Abstract, Objectives, Model Architecture (co) |
| Om Aryan | 21f3002286 | Model Architecture (co), Literature Review |
| Drashti Shah | 22f2001483 | Dataset & Resources, Project Roles |
| Pankaj Mohan Sahu | 21f2001203 | Feasibility & Compute, Evaluation Metrics |
| Mahi Mudgal | 21f3002602 | Project Timeline |



---

## Milestone 2 — Team Contributions

| Name | Roll Number | Contribution |
|---|---|---|
| Drashti Shah | 22f2001483 | Dataset Preparation — Organized and preprocessed datasets, including structuring audio–transcript pairs and preparing train/test splits for model training and evaluation. |
| G Hamsini | 22f3000767| Dataset Preparation — Assisted in cleaning and structuring the DAIC-WOZ dataset and ensuring compatibility with the multimodal pipeline. |
| Pankaj Mohan Sahu | 21f2001203 | Exploratory Data Analysis (EDA) — Conducted statistical analysis of the datasets, examining audio duration, transcript characteristics, and PHQ-9 label distributions. |
| Om Aryan | 21f3002286 | Exploratory Data Analysis (EDA) — Performed visualizations and feature-level exploration to understand acoustic and linguistic patterns relevant to mental health detection. |
| Mahi Mudgal | 21f3002602 | Documentation — Compiled and organized Milestone 2 documentation, detailing dataset preparation procedures and EDA findings. |



---

## Milestone 3 — Team Contributions

| Name | Roll Number | Contribution |
|---|---|---|
| Drashti Shah | 22f2001483 | Dataset Structuring & Preprocessing — Designed and implemented structured data pipelines, including organization of raw and processed datasets, creation of train/validation/test splits, and preprocessing of audio and transcript data for model ingestion. |
| G Hamsini | 22f3000767 | Dataset Structuring & Preprocessing — Assisted in refining dataset organization, ensuring consistency across multimodal inputs, and implementing preprocessing steps for both speech and textual data. |
| Om Aryan | 21f3002286 | Model Architecture & Pipeline — Developed the end-to-end model pipeline, integrating speech processing, transcript-based analysis, and system flow from input to prediction. |
| Pankaj Mohan Sahu | 21f2001203 | Model Architecture & Pipeline — Contributed to architecture design decisions, pipeline integration, and implementation of data flow across different components of the system. |
| Mahi Mudgal | 21f3002602 | Evaluation & Justification — Defined evaluation metrics, validated the end-to-end pipeline on sample data, and provided justification for model architecture including strengths, limitations, and performance considerations. |

*Submitted to: IIT Madras — BSDA4001 Data Science and AI Lab*
