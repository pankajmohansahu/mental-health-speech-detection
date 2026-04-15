# Licenses and Attributions

## AI-Based Early Mental Health Breakdown Detection from Speech Patterns

**Group 6 | BSDA4001 | IIT Madras**

---

## 1. Code License

### MIT License

Copyright (c) 2025 Group 6 — IIT Madras BSDA4001

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**

---

## 2. Dataset Licenses and Attributions

### 2.1 RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song

| Attribute | Detail |
|---|---|
| **Full name** | The Ryerson Audio-Visual Database of Emotional Speech and Song |
| **License** | Creative Commons Attribution (CC BY) |
| **Version** | 1.0.0 |
| **DOI** | 10.5281/zenodo.1188976 |
| **URL** | https://zenodo.org/record/1188976 |
| **Usage in project** | 8-class emotion recognition (neutral, calm, happy, sad, angry, fearful, disgust, surprised) |

**Citation:**
> Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLOS ONE*, 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391

**Note:** RAVDESS is made available under CC BY 4.0. Redistribution requires attribution to the original authors.

---

### 2.2 DAIC-WOZ — Distress Analysis Interview Corpus (Wizard of Oz)

| Attribute | Detail |
|---|---|
| **Full name** | Distress Analysis Interview Corpus — Wizard of Oz |
| **License** | Restricted research use — requires Data Use Agreement |
| **Curated by** | University of Southern California Institute for Creative Technologies (USC ICT) |
| **Access** | Researchers must apply for access via USC ICT |
| **URL** | https://dcapswoz.ict.usc.edu/ |
| **Usage in project** | Binary depression detection (PHQ-8 ≥ 10) and PHQ-8 severity regression |

**Citation:**
> Gratch, J., Artstein, R., Lucas, G., Stratou, G., Scherer, S., Nazarian, A., ... & Morency, L. P. (2014). The Distress Analysis Interview Corpus of human and computer interviews. In *Proceedings of LREC 2014*, Reykjavik, Iceland.

**Important:** DAIC-WOZ data is NOT included in this repository. Users must independently obtain access through the USC ICT portal and agree to their data use terms. The dataset may only be used for non-commercial research purposes.

---

### 2.3 MODMA — Multi-modal Open Dataset for Mental-disorder Analysis

| Attribute | Detail |
|---|---|
| **Full name** | Multi-modal Open Dataset for Mental-disorder Analysis |
| **License** | Research use — subject to original data release terms |
| **Published by** | Lanzhou University |
| **URL** | http://modma.lzu.edu.cn/ |
| **Paper** | https://arxiv.org/abs/2002.09283 |
| **Usage in project** | Binary classification — Major Depressive Disorder (MDD) vs. Healthy Control (HC) |

**Citation:**
> Cai, H., Gao, Y., Sun, S., Li, N., Tian, F., Xiao, H., ... & Li, J. (2020). MODMA dataset: a Multi-modal Open Dataset for Mental-disorder Analysis. *arXiv:2002.09283*.

**Note:** MODMA data is NOT included in this repository. Access is subject to the terms set by Lanzhou University. The dataset is intended for academic research only.

---

### 2.4 SWELL-KW — SWELL Knowledge Work Dataset for Stress and User Modeling

| Attribute | Detail |
|---|---|
| **Full name** | SWELL Knowledge Work Dataset |
| **License** | Creative Commons Attribution Non-Commercial (CC BY-NC) |
| **Published by** | Radboud University Nijmegen |
| **DOI** | 10.1145/2663204.2663257 |
| **URL** | https://cs.ru.nl/~skoldijk/SWELL-KW/Dataset.html |
| **Usage in project** | Physiological stress detection (HRV + EDA, 3-class and binary) |

**Citation:**
> Koldijk, S., Sappelli, M., Verberne, S., Neerincx, M. A., & Kraaij, W. (2014). The SWELL knowledge work dataset for stress and user modeling research. In *Proceedings of the 16th International Conference on Multimodal Interaction (ICMI 2014)*, pp. 291–298. ACM.

**Note:** SWELL is available for non-commercial research use with attribution.

---

## 3. Model and Library Attributions

### 3.1 OpenAI Whisper

| Attribute | Detail |
|---|---|
| **Model** | `openai/whisper-base` |
| **License** | MIT License |
| **Published by** | OpenAI |
| **Paper** | Radford et al. (2022), *Robust Speech Recognition via Large-Scale Weak Supervision* |
| **URL** | https://github.com/openai/whisper |

**Usage in project:** Frozen speech encoder for 512-dimensional speech embedding extraction (Pipeline 2).

**Citation:**
> Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust speech recognition via large-scale weak supervision. *arXiv:2212.04356*.

---

### 3.2 XGBoost

| License | Apache License 2.0 |
|---|---|
| **URL** | https://github.com/dmlc/xgboost |

**Citation:**
> Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of KDD 2016*.

---

### 3.3 PyTorch

| License | BSD-style (PyTorch License) |
|---|---|
| **URL** | https://github.com/pytorch/pytorch |

---

### 3.4 Librosa

| License | ISC License |
|---|---|
| **URL** | https://github.com/librosa/librosa |

---

### 3.5 Scikit-learn

| License | BSD 3-Clause License |
|---|---|
| **URL** | https://github.com/scikit-learn/scikit-learn |

---

### 3.6 Gradio

| License | Apache License 2.0 |
|---|---|
| **URL** | https://github.com/gradio-app/gradio |

---

## 4. Ethical Use Disclaimer

This project and its associated data, models, and application are intended exclusively for **academic research purposes** within the IIT Madras BSDA4001 course. The system:

- Is **not** a licensed medical device.
- Should **not** be used for clinical diagnosis, treatment, or screening of any mental health condition.
- Must be used in compliance with all applicable data protection regulations (e.g., PDPB in India, GDPR in Europe).
- Must not be used to discriminate against individuals based on inferred mental health states.

Any use of this system outside the academic research context requires independent ethical review and appropriate regulatory approval.
