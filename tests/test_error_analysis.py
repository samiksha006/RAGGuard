"""Unit and integration tests for src.analysis.error_analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import error_analysis


def test_confusion_matrix_and_metrics_known_synthetic():
    # Synthetic ground truth and predictions:
    # 4 cases:
    # 0 -> 0 : TN (2 cases)
    # 0 -> 1 : FP (1 case)
    # 1 -> 0 : FN (1 case)
    # 1 -> 1 : TP (2 cases)
    # Total = 6, TN=2, FP=1, FN=1, TP=2
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.7, 0.3, 0.8, 0.9]

    metrics = error_analysis.compute_confusion_matrix_and_metrics(y_true, y_pred, y_prob)

    cm = metrics["confusion_matrix"]
    assert cm["tn"] == 2
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tp"] == 2
    assert cm["matrix_2x2"] == [[2, 1], [1, 2]]

    clf = metrics["classification_metrics"]
    assert pytest.approx(clf["accuracy"], rel=1e-4) == 4 / 6  # 0.6667
    assert pytest.approx(clf["precision"], rel=1e-4) == 2 / 3  # 0.6667
    assert pytest.approx(clf["recall"], rel=1e-4) == 2 / 3  # 0.6667
    assert pytest.approx(clf["specificity"], rel=1e-4) == 2 / 3  # 0.6667
    assert pytest.approx(clf["f1"], rel=1e-4) == 2 / 3  # 0.6667
    assert pytest.approx(clf["balanced_accuracy"], rel=1e-4) == 2 / 3
    assert pytest.approx(clf["false_positive_rate"], rel=1e-4) == 1 / 3
    assert pytest.approx(clf["false_negative_rate"], rel=1e-4) == 1 / 3

    assert "ranking_metrics" in metrics
    assert metrics["ranking_metrics"]["roc_auc"] is not None
    assert metrics["ranking_metrics"]["pr_auc"] is not None
    assert metrics["ranking_metrics"]["brier_score"] is not None


def test_confusion_matrix_zero_divisions_handled():
    # All true class 0, all predicted 0 (no positive predictions, no positive true)
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    metrics = error_analysis.compute_confusion_matrix_and_metrics(y_true, y_pred)
    clf = metrics["classification_metrics"]
    assert clf["accuracy"] == 1.0
    assert clf["precision"] == 0.0
    assert clf["recall"] == 0.0
    assert clf["f1"] == 0.0
    assert clf["specificity"] == 1.0


def test_empty_or_invalid_inputs():
    with pytest.raises(ValueError, match="cannot be empty"):
        error_analysis.compute_confusion_matrix_and_metrics([], [])

    with pytest.raises(ValueError, match="identical lengths"):
        error_analysis.compute_confusion_matrix_and_metrics([0, 1], [0])

    with pytest.raises(ValueError, match="invalid classes"):
        error_analysis.compute_confusion_matrix_and_metrics([0, 2], [0, 1])

    with pytest.raises(ValueError, match="NaN or Inf"):
        error_analysis.compute_confusion_matrix_and_metrics([0, 1], [0, 1], [0.2, np.nan])


def test_validate_prediction_alignment_matching():
    test_df = pd.DataFrame({
        "row_id": [10, 20, 30],
        "label": [0, 1, 0],
        "question": ["q1", "q2", "q3"],
        "context": ["c1", "c2", "c3"],
        "answer": ["a1", "a2", "a3"],
        "model": ["m1", "m2", "m3"],
        "task_type": ["QA", "Summary", "QA"],
    })
    pred_df = pd.DataFrame({
        "row_id": [10, 20, 30],
        "y_true": [0, 1, 0],
        "probability": [0.2, 0.8, 0.4],
        "prediction": [0, 1, 0],
    })

    aligned = error_analysis.validate_prediction_alignment(test_df, pred_df)
    assert len(aligned) == 3
    assert list(aligned["row_id"]) == [10, 20, 30]
    assert list(aligned["y_true"]) == [0, 1, 0]
    assert list(aligned["prediction"]) == [0, 1, 0]
    assert list(aligned["probability"]) == [0.2, 0.8, 0.4]
    assert "question" in aligned.columns
    assert "context" in aligned.columns
    assert "answer" in aligned.columns


def test_validate_prediction_alignment_mismatched_ids():
    test_df = pd.DataFrame({"row_id": [1, 2, 3], "label": [0, 1, 0]})
    pred_df = pd.DataFrame({"row_id": [1, 2, 4], "y_true": [0, 1, 0], "prediction": [0, 1, 0]})

    with pytest.raises(ValueError, match="Row ID mismatch"):
        error_analysis.validate_prediction_alignment(test_df, pred_df)


def test_validate_prediction_alignment_mismatched_row_counts():
    test_df = pd.DataFrame({"row_id": [1, 2], "label": [0, 1]})
    pred_df = pd.DataFrame({"row_id": [1], "y_true": [0], "prediction": [0]})

    with pytest.raises(ValueError, match="Row count mismatch"):
        error_analysis.validate_prediction_alignment(test_df, pred_df)


def test_validate_prediction_alignment_mismatched_labels():
    test_df = pd.DataFrame({"row_id": [1, 2], "label": [0, 1]})
    pred_df = pd.DataFrame({"row_id": [1, 2], "y_true": [0, 0], "prediction": [0, 0]})

    with pytest.raises(ValueError, match="label mismatch"):
        error_analysis.validate_prediction_alignment(test_df, pred_df)


def test_validate_prediction_alignment_missing_optional_text_columns():
    test_df = pd.DataFrame({"row_id": [100], "label": [1]})
    pred_df = pd.DataFrame({"row_id": [100], "y_true": [1], "prediction": [1], "probability": [0.9]})

    aligned = error_analysis.validate_prediction_alignment(test_df, pred_df)
    assert len(aligned) == 1
    assert aligned["y_true"].iloc[0] == 1
    assert "question" not in aligned.columns or pd.isna(aligned["question"].iloc[0])


def test_extract_misclassifications_categorization():
    aligned = pd.DataFrame({
        "row_id": [1, 2, 3, 4],
        "y_true": [0, 0, 1, 1],
        "prediction": [0, 1, 0, 1],
        "probability": [0.1, 0.7, 0.3, 0.9],
        "question": ["q1", "q2", "q3", "q4"],
        "answer": ["a1", "a2", "a3", "a4"],
    })

    errors = error_analysis.extract_misclassifications(aligned, threshold=0.5)
    assert len(errors) == 2

    # Row 2 is FP (0 -> 1)
    fp_row = errors[errors["row_id"] == 2].iloc[0]
    assert fp_row["error_type"] == "False Positive"
    assert pytest.approx(fp_row["confidence_margin"], rel=1e-4) == abs(0.7 - 0.5)

    # Row 3 is FN (1 -> 0)
    fn_row = errors[errors["row_id"] == 3].iloc[0]
    assert fn_row["error_type"] == "False Negative"
    assert pytest.approx(fn_row["confidence_margin"], rel=1e-4) == abs(0.3 - 0.5)


def test_compute_subgroup_analysis():
    aligned = pd.DataFrame({
        "row_id": [1, 2, 3, 4],
        "y_true": [0, 0, 1, 1],
        "prediction": [0, 1, 0, 1],
        "task_type": ["QA", "QA", "Summary", "Summary"],
        "model": ["gpt-4", "gpt-4", "llama", "llama"],
        "question": ["short", "longer question", "q3", "q4"],
        "context": ["ctx1", "ctx2", "ctx3", "ctx4"],
        "answer": ["ans1", "ans2", "ans3", "ans4"],
    })

    subgroup = error_analysis.compute_subgroup_analysis(aligned)
    assert "by_task_type" in subgroup
    assert "QA" in subgroup["by_task_type"]
    assert subgroup["by_task_type"]["QA"]["false_positives"] == 1
    assert subgroup["by_task_type"]["QA"]["false_negatives"] == 0

    assert "by_generator_model" in subgroup
    assert "gpt-4" in subgroup["by_generator_model"]
    assert subgroup["by_generator_model"]["gpt-4"]["false_positives"] == 1


def test_pipeline_generates_all_output_files_in_tmp_path(tmp_path):
    # Create minimal synthetic files for end-to-end test
    test_csv = tmp_path / "test.csv"
    pd.DataFrame({
        "row_id": [1, 2, 3, 4],
        "label": [0, 0, 1, 1],
        "question": ["q1", "q2", "q3", "q4"],
        "context": ["c1", "c2", "c3", "c4"],
        "answer": ["a1", "a2", "a3", "a4"],
        "model": ["m1", "m2", "m1", "m2"],
        "task_type": ["QA", "Summary", "QA", "Summary"],
    }).to_csv(test_csv, index=False)

    pred_csv = tmp_path / "probe_preds.csv"
    pd.DataFrame({
        "row_id": [1, 2, 3, 4],
        "y_true": [0, 0, 1, 1],
        "prediction": [0, 1, 0, 1],
        "probability": [0.2, 0.7, 0.4, 0.8],
    }).to_csv(pred_csv, index=False)

    comp_csv = tmp_path / "comparison.csv"
    pd.DataFrame({
        "feature_segment": ["answer_only", "question_answer", "text_baseline"],
        "layer": [6.0, 5.0, np.nan],
        "classifier_name": ["logistic_regression", "logistic_regression", "tfidf_logistic_regression"],
        "threshold": [0.45, 0.36, 0.455],
        "f1": [0.62, 0.59, 0.65],
        "selected_by_validation": [False, True, False],
    }).to_csv(comp_csv, index=False)

    base_json = tmp_path / "baseline_metrics.json"
    base_json.write_text(
        json.dumps({
            "accuracy": 0.74,
            "precision": 0.61,
            "recall": 0.69,
            "f1": 0.65,
            "pr_auc": 0.72,
            "roc_auc": 0.79,
            "brier_score": 0.17,
            "selected_threshold": 0.455,
        }),
        encoding="utf-8",
    )

    base_preds_csv = tmp_path / "baseline_preds.csv"
    pd.DataFrame({
        "row_id": [1, 2, 3, 4],
        "y_true": [0, 0, 1, 1],
        "prediction": [0, 0, 0, 1],
        "probability": [0.1, 0.3, 0.4, 0.7],
    }).to_csv(base_preds_csv, index=False)

    out_dir = tmp_path / "output_analysis"

    results = error_analysis.run_error_analysis(
        predictions_path=pred_csv,
        test_data_path=test_csv,
        comparison_csv_path=comp_csv,
        baseline_metrics_path=base_json,
        baseline_predictions_path=base_preds_csv,
        train_data_path=None,
        output_dir=out_dir,
    )

    assert results is not None
    expected_files = [
        "class_distribution.png",
        "classification_metrics_comparison.png",
        "ranking_metrics_comparison.png",
        "layer_performance.png",
        "probe_confusion_matrix.png",
        "probe_score_distributions.png",
        "error_breakdown.png",
        "precision_recall_curve.png",
        "misclassified_examples.csv",
        "error_analysis_metrics.json",
        "error_analysis_report.md",
    ]

    for fname in expected_files:
        fpath = out_dir / fname
        assert fpath.is_file(), f"Expected file not found: {fpath}"
        assert fpath.stat().st_size > 0, f"File is empty: {fpath}"


def test_reproducibility_on_real_artifacts():
    # Verify execution against the project's actual Phase 1 artifacts
    pred_path = Path("artifacts/probes/combined_features/selected_probe_predictions_official_test.csv")
    test_path = Path("data/processed/ragtruth/test.csv")
    if not (pred_path.is_file() and test_path.is_file()):
        pytest.skip("Real artifact files not found in workspace.")

    test_df = pd.read_csv(test_path)
    pred_df = pd.read_csv(pred_path)
    aligned = error_analysis.validate_prediction_alignment(test_df, pred_df)

    assert len(aligned) == 2700
    metrics = error_analysis.compute_confusion_matrix_and_metrics(
        aligned["y_true"], aligned["prediction"], aligned["probability"]
    )

    cm = metrics["confusion_matrix"]
    assert cm["tn"] == 947
    assert cm["fp"] == 810
    assert cm["fn"] == 205
    assert cm["tp"] == 738

    clf = metrics["classification_metrics"]
    assert pytest.approx(clf["accuracy"], rel=1e-5) == 0.624074
    assert pytest.approx(clf["precision"], rel=1e-5) == 0.476744
    assert pytest.approx(clf["recall"], rel=1e-5) == 0.782609
    assert pytest.approx(clf["f1"], rel=1e-5) == 0.592533

    rank = metrics["ranking_metrics"]
    assert pytest.approx(rank["roc_auc"], rel=1e-5) == 0.735894
    assert pytest.approx(rank["pr_auc"], rel=1e-5) == 0.541192
