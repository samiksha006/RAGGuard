# RAGGuard: Model Tuning and Benchmarking Report

## 1. Objective

Compare the original word-and-character TF-IDF Logistic Regression baseline against the tuned model for hallucination detection in RAG answers.

## 2. Experimental setup

- Dataset: RAGTruth processed training dataset.
- Split: stratified 80% training and 20% validation.
- Random seed: 42.
- Evaluation threshold: 0.5 for both models.
- Model selection: three-fold cross-validation using F1-score.
- Tuned search configurations requested: 6.
- Cross-validation fits completed: 18.

## 3. Best tuned hyperparameters

```json
{
  "clf__C": 6.79657809075816,
  "features__char__max_features": 15000,
  "features__char__ngram_range": [
    3,
    5
  ],
  "features__char__sublinear_tf": true,
  "features__word__max_features": 30000,
  "features__word__ngram_range": [
    1,
    2
  ],
  "features__word__sublinear_tf": true
}
```

## 4. Validation results

| Metric      |   Original baseline |   Tuned model |   Absolute change | Better model   |
|:------------|--------------------:|--------------:|------------------:|:---------------|
| Accuracy    |              0.7372 |        0.7412 |            0.0040 | Tuned          |
| Precision   |              0.7209 |        0.7167 |           -0.0042 | Baseline       |
| Recall      |              0.6689 |        0.6927 |            0.0238 | Tuned          |
| F1-score    |              0.6939 |        0.7045 |            0.0106 | Tuned          |
| PR-AUC      |              0.7932 |        0.8010 |            0.0078 | Tuned          |
| ROC-AUC     |              0.7899 |        0.7997 |            0.0098 | Tuned          |
| Brier score |              0.1798 |        0.1816 |            0.0018 | Baseline       |

## 5. Interpretation

- Accuracy increased from 0.7372 to 0.7412.
- F1-score increased from 0.6939 to 0.7045.
- PR-AUC increased from 0.7932 to 0.8010.
- Recall increased from 0.6689 to 0.6927, while precision changed from 0.7209 to 0.7167.
- The Brier score should be minimized; it was slightly lower for the original baseline.

## 6. Limitations

- These are internal validation results, not a final independent test-set estimate.
- The tuned model was selected through cross-validation, so its validation results should be interpreted cautiously.
- No claim of statistical significance is made.
- Results depend on the dataset, split, threshold, and chosen metrics.

## 7. Conclusion

The tuned model improved F1-score, recall, PR-AUC, ROC-AUC, and accuracy on the shared validation split. Precision decreased slightly, and the Brier score was marginally worse. The tuned model is promising, but an independent evaluation is needed before claiming generalization improvement.


