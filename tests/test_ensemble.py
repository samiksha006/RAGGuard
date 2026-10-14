"""Automated tests for Phase 3: Hybrid Ensemble and Benchmarking."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import ensemble


def test_compute_comprehensive_metrics_synthetic():
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    thresh = 0.5

    m = ensemble.compute_comprehensive_metrics(y_true, probs, thresh, "TestModel")
    assert m["model_name"] == "TestModel"
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["tn"] == 2
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["pr_auc"] == 1.0
    assert m["roc_auc"] == 1.0


def test_grid_search_ensemble_bounded():
    y_val = np.array([0, 0, 1, 1, 0, 1])
    p_tfidf = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    p_probe = np.array([0.2, 0.1, 0.7, 0.8, 0.4, 0.6])

    alphas = np.array([0.0, 0.5, 1.0])
    thresholds = np.array([0.3, 0.5, 0.7])

    search_df, best_alpha, best_thresh, best_f1 = ensemble.grid_search_ensemble(
        p_tfidf, p_probe, y_val, alphas=alphas, thresholds=thresholds
    )

    assert len(search_df) == 3
    assert 0.0 <= best_alpha <= 1.0
    assert 0.0 <= best_thresh <= 1.0
    assert 0.0 <= best_f1 <= 1.0
    assert "validation_f1" in search_df.columns
    assert "alpha" in search_df.columns


def test_paired_bootstrap_synthetic():
    y_test = np.array([0, 0, 0, 1, 1, 1] * 20)
    p_tfidf = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 20)
    p_probe = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8] * 20)
    p_hybrid = 0.5 * p_tfidf + 0.5 * p_probe

    proba_dict = {
        "TF-IDF": p_tfidf,
        "Probe": p_probe,
        "Hybrid Ensemble (alpha=0.4)": p_hybrid,
    }
    thresh_dict = {"TF-IDF": 0.5, "Probe": 0.5, "Hybrid Ensemble (alpha=0.4)": 0.5}

    boot_res = ensemble.compute_paired_bootstrap(
        y_test, proba_dict, thresh_dict, n_bootstraps=50, seed=42
    )

    assert "models" in boot_res
    assert "TF-IDF" in boot_res["models"]
    assert "f1" in boot_res["models"]["TF-IDF"]
    assert "ci_95_lower" in boot_res["models"]["TF-IDF"]["f1"]
    assert "ci_95_upper" in boot_res["models"]["TF-IDF"]["f1"]
    assert "paired_differences" in boot_res


def test_partition_isolation_and_no_leakage():
    membership_path = Path("artifacts/probes/combined_features/split_membership.csv")
    if not membership_path.is_file():
        pytest.skip("split_membership.csv not found")

    membership = pd.read_csv(membership_path)
    train_ids = set(membership[membership["partition"] == "train"]["row_id"])
    val_ids = set(membership[membership["partition"] == "validation"]["row_id"])
    test_ids = set(membership[membership["partition"] == "official_test"]["row_id"])

    # Disjoint partitions
    assert len(train_ids) == 12072
    assert len(val_ids) == 3018
    assert len(test_ids) == 2700

    assert len(train_ids & val_ids) == 0, "Leakage between train and validation partitions!"
    assert len(train_ids & test_ids) == 0, "Leakage between train and official test partitions!"
    assert len(val_ids & test_ids) == 0, "Leakage between validation and official test partitions!"


def test_phase3_artifacts_generated_and_valid():
    base_dir = Path("artifacts/phase3")
    if not base_dir.is_dir():
        pytest.skip("artifacts/phase3 directory not found")

    expected_files = [
        "metrics/validation_comparison.csv",
        "metrics/official_test_comparison.csv",
        "metrics/ensemble_search.csv",
        "metrics/experiment_metadata.json",
        "predictions/validation_predictions.csv",
        "predictions/official_test_predictions.csv",
        "reports/phase3_research_report.md",
        "reports/phase3_error_analysis.md",
        "plots/model_comparison.png",
        "plots/precision_recall_curves.png",
        "plots/roc_curves.png",
        "plots/ensemble_weight_sensitivity.png",
        "plots/confusion_matrices.png",
    ]

    for rel_path in expected_files:
        p = base_dir / rel_path
        assert p.is_file(), f"Missing Phase 3 artifact: {p}"
        assert p.stat().st_size > 0, f"Artifact is empty: {p}"

    # Verify CSVs parse cleanly
    val_comp = pd.read_csv(base_dir / "metrics/validation_comparison.csv")
    assert len(val_comp) >= 3
    test_comp = pd.read_csv(base_dir / "metrics/official_test_comparison.csv")
    assert len(test_comp) >= 3

    # Verify predictions have correct row count
    test_preds = pd.read_csv(base_dir / "predictions/official_test_predictions.csv")
    assert len(test_preds) == 2700
    val_preds = pd.read_csv(base_dir / "predictions/validation_predictions.csv")
    assert len(val_preds) == 3018

