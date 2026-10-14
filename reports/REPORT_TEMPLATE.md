# RAGGuard: Evaluating Classical Probes on LLM Hidden States for RAG Hallucination Detection

**Course:** 19MAM56 — Advanced Machine Learning Lab  
**Student/team:** [Fill in]  
**Dataset version and access date:** [Fill in]  
**Code repository:** [Public GitHub URL]  
**Deployment URL:** [Only after successful deployment]

## Abstract (≤200 words)
[Problem, approach, dataset, actual results, main finding, limitations. Do not insert planned or synthetic results.]

## 1. Introduction
Define RAG, response-level hallucinations, why text-only checks can fail, objective, research questions and contributions.

## 2. Literature Review
Discuss factuality probes, hidden-state hallucination detection, RAGTruth, output-based baselines, and the closest existing methods. Cite primary papers in IEEE style. Explain precisely what is different in this evaluation.

## 3. Methodology
### 3.1 Dataset and labels
Give source, license/terms, schema, class counts, annotation aggregation rule, duplicates and missingness. State whether the probe uses observer-model or generator-native activations.
### 3.2 Pipeline / block diagram
`RAGTruth → schema validation → label normalization → leakage-aware split → text baseline / hidden-state extraction → classical probes → calibration → evaluation → API/UI`
### 3.3 Formulation
Let response example be `(q_i, c_i, a_i, y_i)`, where `q` is question, `c` context, `a` answer, and `y∈{0,1}`. For selected layer `l`, mean-pooled representation `h_i^(l)` is mapped by a classifier `f_l(h_i^(l)) = p(y_i=1 | h_i^(l))`. Text-only baselines use the same response records and split protocol.
### 3.4 Models and tuning
Document Logistic Regression, Linear SVM (calibrated), Random Forest and any optional XGBoost/Optuna run. Include search space, seed, validation protocol and chosen parameters.
### 3.5 Evaluation
Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, confusion matrix, source-grouped generalization, inference latency and resource use. Explain threshold selection on validation only.

## 4. Implementation and Experiments
Exact OS/runtime, package versions, CPU/GPU, model checkpoint/revision, maximum sequence length, batch size, sample counts and experiment tracking. Include at least five plots: class distribution, text/embedding projection, confusion matrix, ROC, PR, calibration and layer/model comparison where available.

## 5. Results and Comparative Analysis
Insert real tables from `artifacts/*/metrics.json` and `artifacts/probes/comparison.csv`. Discuss differences, confidence/calibration, source holdout performance, false positives/negatives and latency. Do not claim statistical significance without a suitable test or repeated seeds.

## 6. Conclusion and Future Work
Summarize supported findings and limitations. Future work may include claim-level localization, generator-native activation capture, additional RAG datasets, stronger embedding baselines and online drift monitoring.

## References (IEEE style)
[1] Y. Zhang et al., “RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models,” 2024. Verify final bibliographic metadata from the official paper/repository.
[2] J. Han et al., “Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation,” Findings of EMNLP, 2025. https://aclanthology.org/2025.findings-emnlp.880/.
[3] Z. Zhang et al., “ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs,” ACL, 2025. https://aclanthology.org/2025.acl-long.880/.
[4] M. Preiß, “Hallucination Detection with the Internal Layers of LLMs,” arXiv:2509.14254, 2025. https://arxiv.org/abs/2509.14254.
[5] ParticleMedia, “RAGTruth dataset,” GitHub repository. https://github.com/ParticleMedia/RAGTruth.
