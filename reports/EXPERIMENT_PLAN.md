# Experiment plan and rubric evidence

## Research questions and planned evidence

| RQ | Experiment | Output/evidence |
|---|---|---|
| RQ1 | TF-IDF baseline vs hidden-state probes on same records/split | metrics table and PR curve |
| RQ2 | Every hidden-state layer, Logistic Regression; selected layers with Linear SVM/Random Forest | `comparison.csv`, layer-vs-F1 plot |
| RQ3 | Random stratified split vs source-grouped split; optional held-out generator | separate metrics and documented split counts |
| RQ4 | Validation-threshold selection and probability calibration analysis | Brier score, reliability diagram, threshold/coverage curve |
| RQ5 | Feature extraction seconds, train seconds, per-sample inference latency, model size and peak RAM/VRAM | efficiency table |
| Optional | Span-level analysis from RAGTruth annotations | token/span precision/recall/F1 only after valid alignment |

## Acceptance checklist

- [ ] Dataset source, version, license/terms, label mapping and split strategy documented.
- [ ] Duplicate or near-duplicate leakage checked.
- [ ] Baseline and probe use consistent splits and metrics.
- [ ] Vectorizer/PCA/scaler fitted on training only.
- [ ] Threshold tuned on validation, test untouched until final evaluation.
- [ ] At least five meaningful visualizations generated.
- [ ] Calibration and source-grouped generalization reported.
- [ ] API validation and tests pass.
- [ ] Docker image builds and health endpoint works.
- [ ] Cloud UI is genuinely deployed and URL tested before being included in submission.
- [ ] Report contains actual measured results, not placeholders or demo-data scores.
