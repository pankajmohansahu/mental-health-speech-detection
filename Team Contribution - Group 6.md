# 🧠 AI-Based Early Mental Health Breakdown Detection from Speech Patterns

> **Course:** Data Science and AI Lab Project (BSDA4001)
> **Milestone:** 1
> **Group:** 6 | Indian Institute of Technology Madras

---

## 👥 Team Members

| Name | Roll Number |
|---|---|
| G Hamsini | 22f3000767 |
| Om Aryan | 21f3002286 |
| Drashti Shah | 22f2001483 |
| Pankaj Mohan Sahu | 21f2001203 |
| Mahi Mudgal | 21f3002602 |

---

## 📋 Milestone 1 — Team Contributions

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

## 📊 Contribution Summary

| Member | Roll Number | Sections |
|---|---|---|
| G Hamsini | 22f3000767 | Abstract, Objectives, Model Architecture (co) |
| Om Aryan | 21f3002286 | Model Architecture (co), Literature Review |
| Drashti Shah | 22f2001483 | Dataset & Resources, Project Roles |
| Pankaj Mohan Sahu | 21f2001203 | Feasibility & Compute, Evaluation Metrics |
| Mahi Mudgal | 21f3002602 | Project Timeline |

---

## 📁 Report Structure

```
Milestone 1 Report
├── Abstract
├── Objectives
├── Literature Review
├── Model Architecture
├── Dataset & Resources
├── Feasibility and Compute
├── Evaluation Metrics
├── Project Roles
├── Project Timeline
└── Conclusion
```

---

*Submitted to: IIT Madras — BSDA4001 Data Science and AI Lab*
