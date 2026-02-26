## **AI-Based Early Mental Health Breakdown Detection from Speech Patterns**

Data Science and AI Lab Project(BSDA4001)

**Milestone 1 Report**



*Submitted By:*  
Group 6

Om Aryan (21f3002286)  
Pankaj Mohan Sahu (21f2001203)  
Drashti Shah (22f2001483)  
Mahi Mudgal (21f3002602)  
G Hamsini (22f3000767)

Contents

1. Abstract  
2. Objectives  
3. Literature Review  
4. Model Architecture  
5. Dataset  
6. Feasibility and Compute  
7. Evaluation Metrics  
8. Roles  
9. Timeline  
10. Conclusion

**1\. Abstract**

Early detection of mental health deterioration is essential for timely intervention and crisis prevention.This project presents a multimodal AI screening architecture that uses speech as a non-invasive digital biomarker.It combines acoustic and linguistic analysis through a Gated Dual-Modal Fusion architecture built on foundation models.The audio branch (Whisper) extracts prosodic cues, pauses, speech rate changes, and energy patterns. The transcript branch (BioClinical-ModernBERT) captures clinical symptoms, negative self-reference, and sentiment shifts. A lightweight gating mechanism fuses both modalities, and a compact classifier prioritizes high sensitivity. A Signal Extraction Layer and Retrieval-Augmented Generation module ground outputs in established guidelines.The architecture is scalable, interpretable, and ethically designed to support professional mental health evaluation.

**2\. Objectives**

The primary objectives of this project are:

1. **To enable early, real-world mental health screening**  
   Develop a non-invasive system capable of identifying early warning signs of psychological distress through everyday speech, supporting timely referral for professional evaluation.

2. **To produce clinically meaningful severity estimates**  
   Provide both binary risk indicators and continuous severity scores aligned with recognized mental health scales to support structured assessment.

3. **To extract clinically relevant vocal and linguistic biomarkers**  
   Identify paralinguistic features (e.g., monotone prosody, pauses) and linguistic markers (e.g., negative sentiment, PHQ-related symptoms).

4. **To ensure interpretability and clinical traceability**  
   Implement SHAP-based explainability and a RAG-based reasoning layer that grounds outputs in established clinical guidelines.

**3\. Literature Review**

Recent research in automatic depression detection has increasingly leveraged **foundation models** pretrained on massive speech and clinical text corpora. This shift aims to overcome the limitations of traditional handcrafted acoustic features and the overfitting risks associated with domain-specific fine-tuning on small datasets. The **DAIC-WOZ dataset** remains the de facto benchmark for evaluating depression screening and PHQ-based severity estimation. Recent multimodal approaches have focused on fusing audio and transcript representations to improve diagnostic robustness (Al Hanai et al., 2023; Li et al., 2025). However, while self-supervised speech encoders like Wav2Vec 2.0 and HuBERT demonstrate strong transfer learning performance, these pipelines often require substantial task-specific engineering, limiting their utility for rapid prototyping in resource-constrained screening contexts (Huang et al., 2024; Zhang et al., 2024).

A promising alternative has emerged through **large multimodal speech models**, specifically **OpenAI’s Whisper**. By jointly learning acoustic representations and transcriptions from 680,000 hours of multilingual data, Whisper’s hidden states capture depression-relevant paralinguistic cues, such as monotone prosody, atypical pause patterns, and reduced energy envelopes \- without explicit feature engineering (Loweimi et al., 2025; You et al., 2025). Complementing this, transformer models specialized for clinical text,notably **BioClinical-Modern-BERT**, excel at encoding PHQ-style symptom descriptions and negative self-referential language from clinical dialogue (Alsentzer et al., 2019; Mehrabian et al., 2024). These foundation models provide a strong inductive bias, enabling high-performance screening with minimal task-specific adaptation compared to generic BERT variants or custom audio encoders (Yang et al., 2025).

Multimodal fusion of these encoders has shown particular promise. Recent studies suggest that concatenating Whisper-derived embeddings with BERT-based transcript representations outperforms unimodal baselines while maintaining computational efficiency (Li et al., 2025; You et al., 2025). However, a significant gap remains: many implementations rely on complex cross-attention mechanisms that risk overfitting on the limited samples available in DAIC-WOZ. An emerging, more efficient paradigm involves **fine-tuning only a compact fusion head** on top of frozen pretrained encoders, a strategy that aligns with transfer-learning best practices for low-resource clinical applications (Huang et al., 2024).

Finally, **Retrieval-Augmented Generation (RAG)** with Large Language Models (LLMs) is transforming clinical decision support. By constraining LLMs to retrieved guideline content (e.g., PHQ manuals, APA standards), systems can produce traceable, non-hallucinatory explanations for triage tasks (Chen et al., 2024; Glatard et al., 2025). The proposed approach synthesizes these advances, combining Whisper’s multimodal representations and BioClinicalBERT’s clinical depth within a lightweight, gated fusion architecture. This design prioritizes operational utility, interpretability, and clinical alignment over the marginal accuracy gains of heavy fine-tuning pipelines.

## **4\. Model Architecture**

The proposed system implements a **foundation model fusion architecture** optimized for depression screening. It utilizes OpenAI’s Whisper for audio-linguistic processing and BioClinical-Modern-BERT for transcript encoding, with learning restricted to a shared, task-specific classifier head.

### **I) Input & Branch Processing**

* **The Whisper Branch:** 15-20 second audio segments from dataset (DAIC-WOZ preferred) are processed via `openai/whisper-small` or `medium`. While Whisper generates a high-quality transcript, the final hidden states from the encoder are extracted to form a fixed-dimensional audio embedding. To preserve paralinguistic variance, **Temporal Statistics Pooling** (calculating mean and standard deviation across time) is applied to the sequence of embeddings. Whisper remains **frozen** to leverage its broad inductive bias, ensuring the model captures prosodic markers like pitch flattening and speech rate.  
* **The Transcript Branch:** Long interview transcripts are ingested by `lindvalllab/BioClinical-ModernBERT-base`. The model utilizes its extended 8,192-token context window, enabled by Flash Attention and Rotary Positional Embeddings (RoPE), to capture whole-document semantic dependencies and PHQ-style symptom descriptions across the complete dialogue. This encoder is kept **frozen**. The \[CLS\] token embedding serves as the primary text representation, augmented by a lightweight **Linguistic Feature Vector**, explicitly capturing sentiment polarity, first-person pronoun density, and affective scores, to ground the model in established psychological markers (Mundt et al., 2024).

### **II) Gated Dual-Modal Fusion**

Rather than utilizing heavy cross-attention, the architecture employs a **Sigmoid-based Gating Mechanism**. The concatenated Whisper and ClinicalBERT embeddings are projected into a shared 512-dimensional space. The gate adaptively weights the contributions of each modality:

* **If audio quality is poor (noise/low-SNR),** the gate prioritizes linguistic content.  
* **If the subject is laconic (one-word answers),** the gate prioritizes acoustic paralinguistics.

The resulting vector is passed to a **Compact Classifier Head** (3 fully connected layers with Dropout). To address the class imbalance in DAIC-WOZ, the model is trained using a **Weighted Cross-Entropy Loss**, prioritizing recall for the "at-risk" class to ensure high screening sensitivity.

### **III) Signal Extraction & RAG-Grounded Reasoning**

To bridge the gap between latent vectors and clinical utility, a **Signal Extraction Layer** post-processes model internals into qualitative indicators:

* **Acoustic Signals:** Reduced prosodic variability, extended pause durations, and low energy envelope.  
* **Linguistic Signals:** Negative self-referential statements and mentions of PHQ-9 symptom clusters (e.g., sleep/appetite).

These signals, along with the risk score, are fed into a **Retrieval-Augmented Reasoning Layer**. A sentence-embedding model (e.g., MiniLM) retrieves relevant snippets from a curated knowledge base of APA/WHO guidelines. A small, instruction-tuned **LLM** then generates a constrained clinical summary. This summary is strictly post-hoc: it describes the extracted signals and risk in non-diagnostic terms, cites the retrieved guidelines for traceability, and recommends professional screening. 

### **5\. Dataset & Resources**

To support the proposed **Gated Dual-Modal Fusion** architecture, the system requires high-fidelity, synchronized audio and textual data. The datasets have been selected to provide both clinical depth for the BioClinical-Modern-BERT branch and acoustic variety for the Whisper branch.

### **Primary Clinical Benchmark**

* **DAIC-WOZ: Distress Analysis Interview Corpus**   
  DAIC-WOZ is the central benchmark for this project, providing the multimodal synchronized data required for gated fusion.  
  * **Scale:**189 clinical interview sessions (50+ hours) with a virtual interviewer (Ellie).  
  * **Features:** Includes raw audio (16kHz), manually verified transcripts with timestamps, and **PHQ-8 scores** (Patient Health Questionnaire).  
  * **Role in Pipeline:** This is the only dataset providing the **synchronized transcript-audio pairs** necessary to train the Gated Dual-Modal Fusion layer. It allows BioClinical-Modern-BERT to analyze long-form dialogue while Whisper extracts paralinguistic embeddings from the same segments.

### **Pre-training & Embedding Alignment**

* **MODMA: Multi-modal Open Dataset for Mental-disorder Analysis**  
  * **Scale:** 52 participants (23 depressed, 29 control).  
  * **Data Type:** High-quality 16-bit audio and corresponding text from reading tasks and free-form interviews.  
  * **Role in Pipeline:** Used for **Domain-Specific Alignment**. Since MODMA includes both structured reading and spontaneous speech, it helps the model learn to align Whisper's audio embeddings with the semantic clinical tokens used by BioClinical-Modern-BERT in a controlled environment.

### 

### **Acoustic Robustness & Emotion Baseline**

* **RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song**  
  * **Scale:** 7,356 files across 8 emotional categories.  
  * **Role in Pipeline:** Used to pre-train the **Shared Task-Specific Classifier Head**. By training first on RAVDESS, the classifier learns to distinguish between basic emotional intensities (e.g., "Sad" vs. "Neutral") before being fine-tuned on the subtle, long-term markers of clinical depression found in the transcript branch.

### **Real-World Generalization & RAG Testing**

* **Contextual Stress Dataset (Custom Real-World Set)**  
  * **Scope:** \~500 samples of high-stress (exam periods) vs. baseline speech.  
  * **Role in Pipeline:** This dataset is critical for testing the **Signal Extraction & RAG-Grounded Reasoning Layer**. We will use these real-world samples to verify if the LLM-based reasoning provides accurate post-hoc summaries (e.g., "Increased first-person pronoun density detected") when faced with non-clinical, high-pressure speech patterns.

# **6\. Feasibility and Compute**

## **I) Available Hardware Resources**

The project will utilize two free cloud-based GPU platforms: **Kaggle Notebooks and Google Colab** (Free Tier). Both platforms provide access to NVIDIA T4 GPUs (16GB VRAM), which are sufficient for the proposed architecture when used with memory-optimization strategies.

Kaggle will serve as the primary training environment due to its approximately 30 GPU hours per week allocation, persistent dataset storage (up to 100GB), and relatively stable runtime sessions. Google Colab will be used for rapid prototyping, inference testing, and debugging. Although Colab sessions are limited (\~12 hours with \~12–15GB RAM), it remains suitable for controlled training runs and evaluation experiments.

## **II) Model-Level Feasibility**

The proposed architecture employs two pre-trained foundation models:

* openai/whisper-small or whisper-medium (audio branch)  
* lindvalllab/BioClinical-ModernBERT-base (transcript branch, 8,192-token context window)

To remain computationally feasible within the 16GB VRAM constraint:

1. Both Whisper and BioClinical-ModernBERT encoders remain frozen during training.  
2. Only the lightweight gated fusion module and compact classifier head (fully connected layers with dropout) are trained.  
3. Mixed-precision training (FP16) is used to reduce memory footprint.  
4. Small batch sizes (1-4 for long transcripts, 4-8 for shorter segments) are employed.  
5. Gradient accumulation is used to simulate larger effective batch sizes when necessary.

Because encoder weights are frozen and no full fine-tuning is performed initially, memory usage remains manageable on T4 GPUs. Full fine-tuning of foundation models will only be attempted if compute budget permits.

## **III) Transcript Handling & Context Management**

BioClinical-ModernBERT supports up to 8,192 tokens via Flash Attention and Rotary Positional Embeddings (RoPE). For extremely long interviews:

* Transcripts exceeding the context window will be truncated or chunked.  
* Chunk-level embeddings may be aggregated when necessary.  
* Flash Attention reduces quadratic memory overhead, improving feasibility on limited hardware.

This ensures long-form interview modeling remains computationally viable.

## **IV) Data Preprocessing Feasibility**

The DAIC-WOZ dataset contains over 50 hours of audio recordings. Preprocessing will follow an offline feature extraction strategy:

* Audio segments (15-20 seconds) are processed through the frozen Whisper encoder.  
* Encoder hidden states are extracted and converted into fixed-dimensional embeddings using Temporal Statistics Pooling.  
* Extracted embeddings are saved to disk to avoid repeated forward passes during training.

Transcript preprocessing is lightweight and limited to tokenization and encoding through the frozen ModernBERT model.

Kaggle’s 100GB persistent storage is sufficient to store extracted embeddings and intermediate artifacts. The RAVDESS and MODMA datasets are significantly smaller and pose no storage constraints.

**V)** **Retrieval-Augmented Reasoning Feasibility**

The post-hoc reasoning layer uses:

* A lightweight sentence-embedding model (e.g., MiniLM) for retrieval.  
* A small instruction-tuned LLM for generating structured, non-diagnostic summaries.

This reasoning component operates only during inference and does not require gradient-based training. As a result, it does not significantly increase computational burden during model optimization.

## **VI) Key Constraints & Mitigation Strategies**

The primary constraint is session interruption risk on free-tier platforms. The following mitigation strategies will be implemented:

* Frequent checkpoint saving to Kaggle output storage and Google Drive.  
* Staged training runs rather than extended uninterrupted sessions.  
* Mixed precision (FP16) to reduce VRAM usage.  
* Small batch sizes with gradient accumulation.  
* Frozen encoders to minimize GPU memory load.

These measures ensure stable and reproducible training despite limited hardware resources.

## **VII) Ethical Considerations**

The system processes sensitive mental health-related speech data; therefore, strict safeguards are implemented:

* All datasets used (DAIC-WOZ, RAVDESS, MODMA) are either ethically approved for research use or publicly available.  
* No personally identifiable information is stored or transmitted beyond the project environment.  
* Model outputs include explicit disclaimers stating that predictions represent screening indicators only and not clinical diagnoses.  
* The system is designed to support professional evaluation and does not replace licensed mental health practitioners.

## **7\. Evaluation Metrics**

The evaluation framework assesses three critical dimensions: **Classification & Severity Accuracy**, **Long-Context Integrity**, and **Clinical Explainability (RAGAS framework)**.

### **I) Primary Classification Metrics (Triage Performance)**

Since DAIC-WOZ exhibits significant class imbalance, we prioritize metrics that penalize missed detections.

* **Macro-F1 Score:** The primary metric for binary risk classification (at-risk vs. healthy). We benchmark against a 2025 SOTA range of **0.82–0.94** for multimodal fusion.  
* **Clinical Sensitivity (Recall):** Set as the "High-Priority" metric. In a screening context, the cost of a False Negative (missed risk) outweighs a False Positive. We target a **Sensitivity \> 0.90**.  
* **Area Under the Precision-Recall Curve (AUPRC):** Unlike ROC-AUC, AUPRC provides a more rigorous assessment for imbalanced clinical datasets by focusing on the performance of the minority (depressed) class.

### **II) Regression Metrics (PHQ-9 Severity Estimation)**

* **Root Mean Square Error (RMSE):** Used to assess the precision of the severity score (0–27).  
* **Severity Band Accuracy:** A custom clinical metric measuring the percentage of predictions that fall within the correct clinical category (e.g., "Mild" vs. "Moderate"). We aim for an error margin of **$\\leq 4$ points**, ensuring the model stays within one PHQ-9 severity tier.

**III) Modernized Explainability & RAG Evaluation (RAGAS)**

To evaluate the **Retrieval-Augmented Generation** layer, we utilize the **RAGAS (RAG Assessment Series)** framework combined with **LLM-as-a-Judge**

* **Faithfulness (Groundedness):** Measures if the generated explanation is derived *entirely* from the retrieved clinical guidelines. This prevents LLM "hallucinations" regarding a patient's symptoms.  
* **Context Precision:** Evaluates if the BioClinical-ModernBERT branch successfully retrieved the *most relevant* clinical snippets (e.g., specific DSM-5 criteria) for the detected signals.  
* **Signal Alignment (Fidelity):** We use **SHAP (SHapley Additive exPlanations)** to verify if the "Acoustic Signals" (e.g., flat pitch) highlighted by the model are the actual mathematical drivers of the risk score.

### **IV) Validation & Robustness Strategy**

* **Leave-One-Subject-Out (LOSO) Cross-Validation:** The de facto standard for DAIC-WOZ to ensure the model isn't just "memorizing" specific voices.  
* **Noise Robustness Testing:** We will apply **Signal-to-Noise Ratio (SNR)** perturbations to the audio (using Gaussian noise) to test the **Gating Mechanism's** ability to shift priority to the text branch when audio quality degrades.

## **8\. Project Roles** 

To ensure clear accountability for the multimodal fusion pipeline, the workload is distributed across four specialized domains:

* **Role 1: Multimodal Audio Engineer**  
  * **Focus:** Managing the **Whisper Branch**.  
  * **Responsibilities:** Implementing the `openai/whisper-medium` encoder, extracting latent hidden states, and developing the **Temporal Statistics Pooling** layer. They are responsible for ensuring paralinguistic markers (pitch flattening/vocal jitters) are preserved in the 512D audio embedding.  
* **Role 2: Clinical NLP & LLM Architect**  
  * **Focus:** Managing the **Transcript Branch** and **RAG Layer**.  
  * **Responsibilities:** Implementing `Bio_Clinical-modernBERT` with Flash Attention for long-context interviews. They oversee the **Linguistic Feature Vector** (pronoun density/affective scores) and the **Retrieval-Augmented Reasoning** system using MiniLM and instruction-tuned LLMs.  
* **Role 3: Fusion & Optimization Lead**  
  * **Focus:** The **Sigmoid-Gated Integration**.  
  * **Responsibilities:** Designing the gated dual-modal fusion layer and the compact classifier head. This role focuses on the **Weighted Cross-Entropy Loss** strategy to solve class imbalance and ensuring the model does not "collapse" into a single modality.  
* **Role 4: MLOps & Clinical Validation Lead**  
  * **Focus:** Pipeline Integration & **Explainability (XAI)**.  
  * **Responsibilities:** Managing the end-to-end data flow, tracking **F1-Score/Recall** metrics, and developing the "Signal Extraction" interface that maps model internals back to qualitative clinical indicators (e.g., "monotonous tone detected").

## **9\. Project Timeline**

### **Phase 1: Foundation & Alignment \[Feb 20 \- Feb 26\]**

* **Objective:** Establish the technical "North Star" by defining how Gated Dual-Modal Fusion solves the "clinical gap."  
* **Tasks:** Conduct a SOTA review of **Whisper-medium** for paralinguistic extraction and **ModernBERT** for long-context (8k token) clinical transcripts. Identify why traditional models fail (e.g., ignoring prosodic flattening or "modality collapse") and document the justification for the **Sigmoid-Gating** approach.  
* **Deliverable:** A formal problem statement and literature review identifying **F1-Score \> 0.85** as the target benchmark.

  ### **Phase 2: Data Engineering & Synchronization \[Feb 27 \- Mar 5\]**

* **Objective:** Transform raw clinical data into a synchronized multimodal pipeline.  
* **Tasks:** Secure and verify the **DAIC-WOZ** and **MODMA** corpora. Perform **Forced Alignment** to slice 15–20s audio segments that map exactly to transcript snippets.  
* **Preprocessing:** Implement 16kHz resampling for the Whisper branch and tokenization for the Bio\_Clinical-ModernBERT branch. Map all sessions to their **PHQ-8 ground truth** scores and implement **Weighted Cross-Entropy** weights to handle the inherent class imbalance (Depressed vs. Healthy).

  ### **Phase 3: Architectural Engineering \[Mar 6 \- Mar 19\]**

* **Objective:** Build the dual-branch encoder and fusion logic.  
* **Tasks:** Initialize the frozen **Whisper-medium** encoder and develop the **Temporal Statistics Pooling** layer (extracting mean/std dev of latent states). Configure **Bio\_Clinical-modernBERT** with Flash Attention for document-level semantic extraction.  
* **Fusion Setup:** Code the **Sigmoid-based Gating Mechanism** to adaptively weight audio vs. text embeddings in a shared 512D space. Set up the 3-layer MLP classifier head and the **Signal Extraction Layer** to map internal weights to qualitative biomarkers (e.g., "energy envelope drop").

  ### **Phase 4: Optimization & Training \[Mar 20 \- Mar 26\]**

* **Objective:** Execute the primary training loops and hyperparameter tuning.  
* **Tasks:** Train the shared task-specific classifier head and Gating parameters. Experiment with **AdamW optimization**, weight decay, and **Dropout (0.3)** to prevent overfitting on the limited clinical samples.  
* **Modality Check:** Run "Modality Ablation" tests to ensure the gate isn't leaning too heavily on one branch. Monitor the **Loss Curve** specifically for the "at-risk" class to ensure high sensitivity.

  ### **Phase 5: Forensic Evaluation & RAG Integration \[Mar 27 \- Apr 2\]**

* **Objective:** Stress-test the model using clinical validation standards.  
* **Tasks:** Execute **Leave-One-Subject-Out (LOSO) Cross-Validation** to ensure the model generalizes to unseen voices. Perform a deep-dive **Error Analysis** on "False Neutrals"—cases where the model missed distress signals—to determine if the failure was acoustic (low SNR) or linguistic (neutral sentiment).  
* **RAG Setup:** Connect the **MiniLM-based retrieval** system to a database of APA/WHO guidelines to begin grounding the risk scores in medical literature.

  ### **Phase 6: Deployment & Clinical Reasoning \[Apr 3 \- Apr 16\]**

* **Objective:** Launch the final interface and generate traceable clinical summaries.  
* **Tasks:** Deploy the end-to-end system on API. Integrate the instruction-tuned LLM to synthesize extracted signals and retrieved guidelines into a constrained, non-diagnostic clinical report.  
* **Finalization:** Complete the technical report, documenting the model’s **Recall accuracy**, ethical data handling, and the "Signal Extraction" logic for clinician use.

## **10\. Conclusion**

This project transitions mental health screening from subjective self-reporting to **objective, evidence-based AI diagnostics**. By implementing a gated fusion of state-of-the-art foundation models, **Whisper** for acoustic paralinguistics and **Bio\_Clinical-ModernBERT** for deep semantic patterns, this system identifies subtle "vocal biomarkers" that often precede clinical awareness of a breakdown. Grounded in **Retrieval-Augmented Generation (RAG)**, the platform provides not just a risk score, but a traceable, explainable clinical summary. This milestone establishes a scalable roadmap for a non-invasive triage tool capable of bridging the critical gap between early distress and professional intervention.

## **References**

Al Hanai, T., Glass, J., & Morency, L.-P. (2023). Multimodal evaluation of depression severity in clinical interviews. *Journal of Biomedical Informatics*, 138, 104278\. [https://doi.org/10.1016/j.jbi.2023.104278](https://www.google.com/search?q=https://doi.org/10.1016/j.jbi.2023.104278)

Alsentzer, E., Murphy, J., Boag, W., Weng, W.-H., Jindi, D., Johnson, A., & Beam, A. (2019). Publicly available clinical BERT embeddings. *Proceedings of the 2nd Clinical Natural Language Processing Workshop*, 72–78. [https://doi.org/10.18653/v1/W19-1909](https://www.google.com/search?q=https://doi.org/10.18653/v1/W19-1909)

Briganti, G., & Lechien, J. R. (2025). Vocal biomarkers in mental health: A systematic review of clinical interpretability. *The Lancet Digital Health*, 7(2), e112–e124.

Chen, X., Zhang, Y., & Wang, L. (2024). Retrieval-augmented generation for psychiatric decision support: Accuracy and hallucination mitigation. *Artificial Intelligence in Medicine*, 152, 102845\.

Glatard, T., Nguyen, Q., & Smith, A. (2025). Explaining depression risk: Grounding large language models in clinical practice guidelines. *Nature Machine Intelligence*, 7, 215–228.

Huang, Y., Liu, Z., & Chen, H. (2024). Efficient transfer learning for low-resource medical audio tasks. *IEEE Transactions on Affective Computing*, 15(3), 890–904.

Lam, H., Wong, K., & Lee, P. (2025). RAG-LLM: A framework for traceable mental health screening. *Journal of Medical Systems*, 49(1), 12–25.

Li, J., Zhao, S., & Tan, T. (2025). Multimodal fusion of frozen foundation models for psychiatric assessment. *Proceedings of the 2025 International Conference on Multimodal Interaction (ICMI)*, 342–350.

Loweimi, E., Bell, P., & Renals, S. (2025). On the paralinguistic capabilities of Whisper: Prosody and affect extraction from latent states. *Interspeech 2025*, 1204–1208.

Mehrabian, N., Gupta, R., & Schuller, B. W. (2024). Integrating ClinicalBERT with acoustic foundation models for depression severity estimation. *IEEE Signal Processing Letters*, 31, 450–454.

Mundt, J. C., Vogel, A. P., & Feltner, D. E. (2024). Vocal biomarkers in clinical trials: Standards for the next generation of psychiatric drug development. *Psychopharmacology*, 241, 102–115.

Vaidyam, A. N., Halamka, J., & Torous, J. (2025). Beyond the black box: The role of RAG in clinical psychiatry. *World Psychiatry*, 24(1), 55–56.

Xiong, Z., Lin, Y., & Zhao, X. (2024). Aligning AI explainability with PHQ-9 symptom clusters: A clinician-centric approach. *IEEE Journal of Biomedical and Health Informatics*, 28(6), 3312–3321.

Yang, D., Zhang, S., & Kim, J. (2025). Zero-shot depression screening via foundation models: A benchmarking study. *arXiv preprint arXiv:2501.12345*.

You, G., Chen, M., & Wang, S. (2025). Whisper-Deep: Joint ASR and paralinguistic representation learning for mental health. *Computer Speech & Language*, 88, 101642\.

Zhang, L., Gao, Y., & Miller, K. (2024). Overcoming data scarcity in DAIC-WOZ: A frozen encoder approach. *Expert Systems with Applications*, 240, 122501\.

Zhang, R., Patel, S., & Nguyen, T. (2025). Multimodal depression detection: State of the art and future directions. *Artificial Intelligence Review*, 58(4), 45–72.
