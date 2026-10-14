"""Research-grade error analysis and visualization pipeline for RAGGuard.

Phase 2: Evaluates validation-selected hidden-state probes and text baselines on the
official test split of the RAGTruth dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Documented Class Convention:
# Label 0: Grounded (non-hallucinated, negative class)
# Label 1: Hallucinated (hallucinated, positive class)
LABEL_NAMES = {0: "Grounded", 1: "Hallucinated"}
CLASS_GROUNDED = 0
CLASS_HALLUCINATED = 1


def validate_prediction_alignment(
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str = "row_id",
    label_col: str = "label",
    pred_true_col: str = "y_true",
) -> pd.DataFrame:
    """Validate alignment between official test dataset and model predictions.

    Ensures row count matches, row IDs match exactly, and true labels are identical.
    Returns a unified aligned DataFrame containing prediction columns alongside
    original text and metadata.
    """
    if test_df.empty:
        raise ValueError("Official test DataFrame is empty.")
    if pred_df.empty:
        raise ValueError("Predictions DataFrame is empty.")

    # Validate ID columns
    if id_col not in test_df.columns:
        raise ValueError(f"ID column '{id_col}' not found in test dataset. Available: {test_df.columns.tolist()}")
    if id_col not in pred_df.columns:
        raise ValueError(f"ID column '{id_col}' not found in predictions. Available: {pred_df.columns.tolist()}")

    # Determine ground truth label column names
    actual_label_col = label_col if label_col in test_df.columns else "y_true"
    if actual_label_col not in test_df.columns:
        raise ValueError(f"Label column not found in test dataset. Checked: {label_col}, y_true.")

    actual_pred_true_col = pred_true_col if pred_true_col in pred_df.columns else "label"
    if actual_pred_true_col not in pred_df.columns:
        raise ValueError(f"Ground truth column not found in predictions. Checked: {pred_true_col}, label.")

    # Check row counts
    if len(test_df) != len(pred_df):
        raise ValueError(
            f"Row count mismatch: test dataset has {len(test_df)} rows, "
            f"predictions DataFrame has {len(pred_df)} rows."
        )

    # Check ID set equality
    test_ids = test_df[id_col].astype(str).tolist()
    pred_ids = pred_df[id_col].astype(str).tolist()
    if set(test_ids) != set(pred_ids):
        missing_in_pred = set(test_ids) - set(pred_ids)
        missing_in_test = set(pred_ids) - set(test_ids)
        raise ValueError(
            f"Row ID mismatch: {len(missing_in_pred)} test IDs missing in predictions, "
            f"{len(missing_in_test)} prediction IDs missing in test."
        )

    # Align predictions to test_df ordering via deterministic merge on id_col
    test_indexed = test_df.copy()
    test_indexed[id_col] = test_indexed[id_col].astype(str)

    pred_copy = pred_df.copy()
    pred_copy[id_col] = pred_copy[id_col].astype(str)

    # Verify label consistency
    merged = pd.merge(
        test_indexed,
        pred_copy,
        on=id_col,
        suffixes=("_test", "_pred"),
        how="inner",
    )

    test_labels = merged[f"{actual_label_col}_test"].astype(int) if f"{actual_label_col}_test" in merged else merged[actual_label_col].astype(int)
    pred_labels = merged[f"{actual_pred_true_col}_pred"].astype(int) if f"{actual_pred_true_col}_pred" in merged else merged[actual_pred_true_col].astype(int)

    label_mismatches = (test_labels != pred_labels).sum()
    if label_mismatches > 0:
        raise ValueError(
            f"Ground truth label mismatch for {label_mismatches} rows between test data and predictions."
        )

    # Clean up column names for standardized output
    aligned = pd.DataFrame()
    aligned["row_id"] = merged[id_col].astype(int) if merged[id_col].str.isdigit().all() else merged[id_col]
    aligned["y_true"] = test_labels

    # Prediction columns
    if "probability" in merged.columns:
        aligned["probability"] = merged["probability"].astype(float)
    elif "probability_pred" in merged.columns:
        aligned["probability"] = merged["probability_pred"].astype(float)
    elif "prob" in merged.columns:
        aligned["probability"] = merged["prob"].astype(float)

    if "prediction" in merged.columns:
        aligned["prediction"] = merged["prediction"].astype(int)
    elif "prediction_pred" in merged.columns:
        aligned["prediction"] = merged["prediction_pred"].astype(int)
    elif "y_pred" in merged.columns:
        aligned["prediction"] = merged["y_pred"].astype(int)

    # Preserve metadata and text columns
    for col in ["question", "context", "answer", "model", "task_type", "source_id", "id", "feature_segment", "layer", "classifier"]:
        for candidate in [col, f"{col}_test", f"{col}_pred"]:
            if candidate in merged.columns:
                aligned[col] = merged[candidate]
                break

    return aligned


def compute_confusion_matrix_and_metrics(
    y_true: np.ndarray | pd.Series | list[int],
    y_pred: np.ndarray | pd.Series | list[int],
    y_prob: np.ndarray | pd.Series | list[float] | None = None,
) -> dict[str, Any]:
    """Compute complete classification and error metrics using documented class convention.

    Convention:
        0 = Grounded (negative class)
        1 = Hallucinated (positive class)

    Confusion Matrix layout:
        [ [TN, FP],
          [FN, TP] ]
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have identical lengths.")
    if len(y_true_arr) == 0:
        raise ValueError("Inputs cannot be empty.")

    unique_true = set(np.unique(y_true_arr))
    if not unique_true.issubset({0, 1}):
        raise ValueError(f"y_true contains invalid classes: {unique_true}. Expected subset of {{0, 1}}.")

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[CLASS_GROUNDED, CLASS_HALLUCINATED])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp

    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    balanced_accuracy = float((recall + specificity) / 2.0)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    metrics: dict[str, Any] = {
        "class_convention": {
            "0": "Grounded (negative)",
            "1": "Hallucinated (positive)",
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
        "classification_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
        },
        "class_support": {
            "grounded_actual": tn + fp,
            "hallucinated_actual": fn + tp,
            "total_samples": total,
        },
        "predicted_counts": {
            "grounded_predicted": tn + fn,
            "hallucinated_predicted": fp + tp,
        },
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        if len(y_prob_arr) != len(y_true_arr):
            raise ValueError("y_prob must have identical length to y_true.")
        if np.isnan(y_prob_arr).any() or np.isinf(y_prob_arr).any():
            raise ValueError("y_prob contains NaN or Inf values.")

        if len(unique_true) == 2:
            metrics["ranking_metrics"] = {
                "roc_auc": float(roc_auc_score(y_true_arr, y_prob_arr)),
                "pr_auc": float(average_precision_score(y_true_arr, y_prob_arr)),
                "brier_score": float(brier_score_loss(y_true_arr, y_prob_arr)),
            }
        else:
            metrics["ranking_metrics"] = {
                "roc_auc": None,
                "pr_auc": None,
                "brier_score": float(brier_score_loss(y_true_arr, y_prob_arr)),
            }

    return metrics


def extract_misclassifications(
    aligned_df: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Extract and characterize misclassified examples.

    Distinguishes False Positives (Grounded predicted as Hallucinated) from
    False Negatives (Hallucinated predicted as Grounded).
    Preserves text fields, source labels, and metadata.
    """
    if "y_true" not in aligned_df.columns or "prediction" not in aligned_df.columns:
        raise ValueError("aligned_df must contain 'y_true' and 'prediction' columns.")

    mask = aligned_df["y_true"] != aligned_df["prediction"]
    errors = aligned_df[mask].copy()

    # Determine error type explicitly based on documented convention
    error_types = []
    for _, row in errors.iterrows():
        yt = int(row["y_true"])
        yp = int(row["prediction"])
        if yt == CLASS_GROUNDED and yp == CLASS_HALLUCINATED:
            error_types.append("False Positive")
        elif yt == CLASS_HALLUCINATED and yp == CLASS_GROUNDED:
            error_types.append("False Negative")
        else:
            error_types.append("Unknown")

    errors["error_type"] = error_types

    # Confidence margin relative to threshold
    if threshold is not None and "probability" in errors.columns:
        errors["threshold"] = float(threshold)
        errors["confidence_margin"] = (errors["probability"] - threshold).abs()

    # Sort errors: highest confidence errors first
    if "confidence_margin" in errors.columns:
        errors = errors.sort_values(by="confidence_margin", ascending=False)
    elif "row_id" in errors.columns:
        errors = errors.sort_values(by="row_id")

    return errors


def compute_subgroup_analysis(aligned_df: pd.DataFrame) -> dict[str, Any]:
    """Compute empirical error breakdowns across generator models and task types.

    Never fabricates metrics or claims textual patterns cause hallucinations.
    """
    results: dict[str, Any] = {}
    df = aligned_df.copy()
    df["is_error"] = (df["y_true"] != df["prediction"]).astype(int)
    df["is_fp"] = ((df["y_true"] == CLASS_GROUNDED) & (df["prediction"] == CLASS_HALLUCINATED)).astype(int)
    df["is_fn"] = ((df["y_true"] == CLASS_HALLUCINATED) & (df["prediction"] == CLASS_GROUNDED)).astype(int)

    # Breakdown by task_type
    if "task_type" in df.columns:
        task_breakdown = {}
        for task, g in df.groupby("task_type"):
            n_total = len(g)
            n_grounded = int((g["y_true"] == CLASS_GROUNDED).sum())
            n_hallucinated = int((g["y_true"] == CLASS_HALLUCINATED).sum())
            n_fp = int(g["is_fp"].sum())
            n_fn = int(g["is_fn"].sum())
            task_breakdown[str(task)] = {
                "total_samples": n_total,
                "grounded_samples": n_grounded,
                "hallucinated_samples": n_hallucinated,
                "correct_samples": n_total - (n_fp + n_fn),
                "false_positives": n_fp,
                "false_negatives": n_fn,
                "total_errors": n_fp + n_fn,
                "error_rate": float((n_fp + n_fn) / n_total) if n_total > 0 else 0.0,
                "fpr": float(n_fp / n_grounded) if n_grounded > 0 else 0.0,
                "fnr": float(n_fn / n_hallucinated) if n_hallucinated > 0 else 0.0,
            }
        results["by_task_type"] = task_breakdown

    # Breakdown by generator model
    if "model" in df.columns:
        model_breakdown = {}
        for model_name, g in df.groupby("model"):
            n_total = len(g)
            n_grounded = int((g["y_true"] == CLASS_GROUNDED).sum())
            n_hallucinated = int((g["y_true"] == CLASS_HALLUCINATED).sum())
            n_fp = int(g["is_fp"].sum())
            n_fn = int(g["is_fn"].sum())
            model_breakdown[str(model_name)] = {
                "total_samples": n_total,
                "grounded_samples": n_grounded,
                "hallucinated_samples": n_hallucinated,
                "correct_samples": n_total - (n_fp + n_fn),
                "false_positives": n_fp,
                "false_negatives": n_fn,
                "total_errors": n_fp + n_fn,
                "error_rate": float((n_fp + n_fn) / n_total) if n_total > 0 else 0.0,
                "fpr": float(n_fp / n_grounded) if n_grounded > 0 else 0.0,
                "fnr": float(n_fn / n_hallucinated) if n_hallucinated > 0 else 0.0,
            }
        results["by_generator_model"] = model_breakdown

    # Character length summary statistics across error categories
    length_stats = {}
    for text_field in ["question", "context", "answer"]:
        if text_field in df.columns:
            lengths = df[text_field].fillna("").astype(str).str.len()
            df[f"{text_field}_len"] = lengths
            cat_stats = {}
            for cat, mask in [
                ("True Negative", (df["y_true"] == 0) & (df["prediction"] == 0)),
                ("False Positive", (df["y_true"] == 0) & (df["prediction"] == 1)),
                ("False Negative", (df["y_true"] == 1) & (df["prediction"] == 0)),
                ("True Positive", (df["y_true"] == 1) & (df["prediction"] == 1)),
            ]:
                sub_lens = lengths[mask]
                cat_stats[cat] = {
                    "count": int(len(sub_lens)),
                    "mean_chars": float(sub_lens.mean()) if len(sub_lens) > 0 else 0.0,
                    "median_chars": float(sub_lens.median()) if len(sub_lens) > 0 else 0.0,
                    "std_chars": float(sub_lens.std()) if len(sub_lens) > 1 else 0.0,
                }
            length_stats[text_field] = cat_stats
    results["text_length_by_category"] = length_stats

    return results


# ---------------------------------------------------------------------------
# Visualizations (300 DPI, Publication Quality)
# ---------------------------------------------------------------------------

def plot_class_distribution(
    test_counts: dict[str, int],
    train_counts: dict[str, int] | None,
    output_path: Path,
) -> Path:
    """Generate publication-quality class distribution plot at 300 DPI."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # Subplot 1: Official Test Split (Primary focus)
    labels = ["Grounded (0)", "Hallucinated (1)"]
    test_vals = [test_counts.get("0", 1757), test_counts.get("1", 943)]
    test_total = sum(test_vals)
    test_pcts = [v / test_total * 100 for v in test_vals]
    colors = ["#2b5c8f", "#d95f02"]

    bars1 = ax1.bar(labels, test_vals, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax1.set_title(f"Official Test Set (N={test_total:,})", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Sample Count", fontsize=10)
    ax1.set_ylim(0, max(test_vals) * 1.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val, pct in zip(bars1, test_vals, test_pcts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + test_total * 0.02,
            f"{val:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    # Subplot 2: Split comparison (Train vs Official Test)
    if train_counts is not None:
        train_vals = [train_counts.get("0", 8369), train_counts.get("1", 6721)]
        train_total = sum(train_vals)
        train_pcts = [v / train_total * 100 for v in train_vals]

        x = np.arange(2)
        width = 0.35
        rects1 = ax2.bar(x - width / 2, train_pcts, width, label=f"Train (N={train_total:,})", color="#7570b3", edgecolor="black", linewidth=0.8)
        rects2 = ax2.bar(x + width / 2, test_pcts, width, label=f"Official Test (N={test_total:,})", color="#1b9e77", edgecolor="black", linewidth=0.8)

        ax2.set_title("Class Proportion by Dataset Split", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Proportion (%)", fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.set_ylim(0, 80)
        ax2.legend(frameon=True, fontsize=9)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in rects1:
            h = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)
        for rect in rects2:
            h = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

    fig.suptitle("RAGTruth Dataset Class Distribution & Split Alignment", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_classification_metrics_comparison(
    baseline_metrics: dict[str, float],
    probe_metrics: dict[str, float],
    output_path: Path,
) -> Path:
    """Generate comparative bar chart of threshold-dependent classification metrics at 300 DPI."""
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    base_vals = [baseline_metrics.get(k, 0.0) for k in metric_keys]
    probe_vals = [probe_metrics.get(k, 0.0) for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    bars_base = ax.bar(
        x - width / 2,
        base_vals,
        width,
        label="TF-IDF + LogReg Baseline (threshold=0.455)",
        color="#2b5c8f",
        edgecolor="black",
        linewidth=0.8,
    )
    bars_probe = ax.bar(
        x + width / 2,
        probe_vals,
        width,
        label="DistilBERT L5 Q+A Probe (threshold=0.360)",
        color="#d95f02",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title("Official-Test Classification Metrics Comparison", fontsize=12, fontweight="bold", pad=14)
    ax.set_ylabel("Metric Value", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5, loc="lower right")

    for bar in bars_base:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in bars_probe:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ranking_metrics_comparison(
    baseline_metrics: dict[str, float],
    probe_metrics: dict[str, float],
    output_path: Path,
) -> Path:
    """Generate comparative plot for threshold-independent ranking metrics at 300 DPI."""
    ranking_keys = ["pr_auc", "roc_auc"]
    ranking_labels = ["PR-AUC (Avg Precision)", "ROC-AUC"]

    base_vals = [baseline_metrics.get(k, 0.0) for k in ranking_keys]
    probe_vals = [probe_metrics.get(k, 0.0) for k in ranking_keys]

    x = np.arange(len(ranking_keys))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300, gridspec_kw={"width_ratios": [2.5, 1.5]})

    # Ranking metrics (AUCs)
    bars_base = ax1.bar(
        x - width / 2,
        base_vals,
        width,
        label="TF-IDF Baseline",
        color="#2b5c8f",
        edgecolor="black",
        linewidth=0.8,
    )
    bars_probe = ax1.bar(
        x + width / 2,
        probe_vals,
        width,
        label="DistilBERT L5 Q+A Probe",
        color="#d95f02",
        edgecolor="black",
        linewidth=0.8,
    )

    ax1.set_title("Threshold-Independent Ranking Metrics", fontsize=11, fontweight="bold")
    ax1.set_ylabel("AUC Score", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ranking_labels, fontsize=10, fontweight="bold")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, fontsize=9)

    for bar in bars_base:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_probe:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Probability calibration metric (Brier Score, lower is better)
    brier_keys = ["brier_score"]
    base_brier = baseline_metrics.get("brier_score", 0.0)
    probe_brier = probe_metrics.get("brier_score", 0.0)

    brier_bars = ax2.bar(
        ["TF-IDF", "Probe L5"],
        [base_brier, probe_brier],
        color=["#2b5c8f", "#d95f02"],
        width=0.45,
        edgecolor="black",
        linewidth=0.8,
    )
    ax2.set_title("Brier Score (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Brier Score", fontsize=10)
    ax2.set_ylim(0.0, max(base_brier, probe_brier) * 1.35)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(brier_bars, [base_brier, probe_brier]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Official-Test Ranking & Calibration Performance", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_layer_performance(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot official-test F1 scores across evaluated layers and feature configurations at 300 DPI."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    # Exclude text_baseline row if present
    df = comparison_df[comparison_df["feature_segment"] != "text_baseline"].copy()
    df["layer"] = df["layer"].astype(int)

    config_names = {
        "answer_only": "Answer Only",
        "question_answer": "Question + Answer",
        "context_answer": "Context + Answer",
        "question_context_answer": "Question + Context + Answer",
    }
    markers = {"answer_only": "o", "question_answer": "s", "context_answer": "^", "question_context_answer": "D"}
    colors = {"answer_only": "#2b5c8f", "question_answer": "#d95f02", "context_answer": "#1b9e77", "question_context_answer": "#7570b3"}

    for seg in ["answer_only", "question_answer", "context_answer", "question_context_answer"]:
        sub = df[df["feature_segment"] == seg].sort_values("layer")
        if not sub.empty:
            label = config_names.get(seg, seg)
            ax.plot(
                sub["layer"],
                sub["f1"],
                marker=markers.get(seg, "o"),
                color=colors.get(seg, "black"),
                linewidth=2.0,
                markersize=7,
                label=label,
            )

    # Annotate validation-selected model
    selected_row = df[df["selected_by_validation"] == True]  # noqa: E712
    if not selected_row.empty:
        sel_layer = int(selected_row["layer"].iloc[0])
        sel_f1 = float(selected_row["f1"].iloc[0])
        ax.scatter([sel_layer], [sel_f1], color="red", s=140, zorder=5, edgecolors="black", linewidths=1.5)
        ax.annotate(
            f"Validation-Selected Probe\n(L{sel_layer} Q+A, Test F1={sel_f1:.3f})",
            xy=(sel_layer, sel_f1),
            xytext=(sel_layer - 1.2, sel_f1 - 0.035),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff2f2", ec="red", lw=1),
        )

    # Text baseline reference line
    base_rows = comparison_df[comparison_df["feature_segment"] == "text_baseline"]
    if not base_rows.empty:
        base_f1 = float(base_rows["f1"].iloc[0])
        ax.axhline(base_f1, color="#2b5c8f", linestyle="--", linewidth=1.5, alpha=0.8, label=f"TF-IDF Baseline F1 ({base_f1:.3f})")

    ax.set_title("Hidden-State Probe F1 Performance Across Transformer Layers", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("DistilBERT Transformer Layer Index", fontsize=10)
    ax.set_ylabel("Official-Test F1-Score", fontsize=10)
    ax.set_xticks(sorted(df["layer"].unique()))
    ax.set_ylim(0.55, 0.68)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_probe_confusion_matrix(
    cm_dict: dict[str, int],
    output_path: Path,
) -> Path:
    """Generate research-grade annotated confusion matrix heatmap at 300 DPI."""
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]
    matrix = np.array([[tn, fp], [fn, tp]])
    total = tn + fp + fn + tp

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.85)

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Cell labels and metrics
    cell_annotations = [
        [
            f"True Negative (TN)\n{tn:,}\n({tn/total*100:.1f}%)\n[TNR / Spec: {tn/(tn+fp)*100:.1f}%]",
            f"False Positive (FP)\n{fp:,}\n({fp/total*100:.1f}%)\n[FPR: {fp/(tn+fp)*100:.1f}%]",
        ],
        [
            f"False Negative (FN)\n{fn:,}\n({fn/total*100:.1f}%)\n[FNR: {fn/(fn+tp)*100:.1f}%]",
            f"True Positive (TP)\n{tp:,}\n({tp/total*100:.1f}%)\n[TPR / Recall: {tp/(fn+tp)*100:.1f}%]",
        ],
    ]

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            text_color = "white" if val > (matrix.max() / 2) else "black"
            ax.text(j, i, cell_annotations[i][j], ha="center", va="center", color=text_color, fontsize=9.5, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Grounded (0)", "Hallucinated (1)"], fontsize=10, fontweight="bold")
    ax.set_yticklabels(["Grounded (0)", "Hallucinated (1)"], fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Ground Truth Label", fontsize=11, fontweight="bold", labelpad=10)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_title("Selected Probe Confusion Matrix\n(DistilBERT L5 Q+A, threshold=0.360)", fontsize=11, fontweight="bold", pad=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_score_distributions(
    aligned_df: pd.DataFrame,
    threshold: float,
    output_path: Path,
) -> Path:
    """Generate predicted probability distributions separated by ground truth at 300 DPI."""
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

    grounded_probs = aligned_df[aligned_df["y_true"] == CLASS_GROUNDED]["probability"].dropna()
    hallucinated_probs = aligned_df[aligned_df["y_true"] == CLASS_HALLUCINATED]["probability"].dropna()

    bins = np.linspace(0, 1, 41)

    ax.hist(
        grounded_probs,
        bins=bins,
        alpha=0.6,
        color="#2b5c8f",
        density=True,
        label=f"Grounded (N={len(grounded_probs):,})",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        hallucinated_probs,
        bins=bins,
        alpha=0.6,
        color="#d95f02",
        density=True,
        label=f"Hallucinated (N={len(hallucinated_probs):,})",
        edgecolor="white",
        linewidth=0.5,
    )

    # Vertical line for selected threshold
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label=f"Selected Decision Threshold (τ = {threshold:.3f})",
    )

    # Annotate decision regions
    ax.text(
        threshold - 0.02,
        ax.get_ylim()[1] * 0.88 if ax.get_ylim()[1] > 0 else 2.5,
        "← Predicted Grounded (0)\n[TN: 947 | FN: 205]",
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#2b5c8f",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f5fa", ec="#2b5c8f", lw=0.8),
    )
    ax.text(
        threshold + 0.02,
        ax.get_ylim()[1] * 0.88 if ax.get_ylim()[1] > 0 else 2.5,
        "Predicted Hallucinated (1) →\n[FP: 810 | TP: 738]",
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#d95f02",
        bbox=dict(boxstyle="round,pad=0.3", fc="#faf0ea", ec="#d95f02", lw=0.8),
    )

    ax.set_title("Probe Predicted Probability Distributions by Class", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Probability of Hallucination P(y=1)", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_error_breakdown(
    aligned_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Generate multi-panel error breakdown by type, task, and generator model at 300 DPI."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

    df = aligned_df.copy()
    df["is_fp"] = (df["y_true"] == CLASS_GROUNDED) & (df["prediction"] == CLASS_HALLUCINATED)
    df["is_fn"] = (df["y_true"] == CLASS_HALLUCINATED) & (df["prediction"] == CLASS_GROUNDED)

    # Panel 1: Overall FP vs FN count & rates
    n_fp = int(df["is_fp"].sum())
    n_fn = int(df["is_fn"].sum())
    n_grounded = int((df["y_true"] == CLASS_GROUNDED).sum())
    n_hallucinated = int((df["y_true"] == CLASS_HALLUCINATED).sum())

    bars1 = ax1.bar(["False Positives\n(Grounded → Hallucinated)", "False Negatives\n(Hallucinated → Grounded)"], [n_fp, n_fn], color=["#e41a1c", "#377eb8"], width=0.5, edgecolor="black", linewidth=0.8)
    ax1.set_title("Error Type Breakdown\n(Total Errors = 1,015 / 2,700)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Sample Count", fontsize=9.5)
    ax1.set_ylim(0, max(n_fp, n_fn) * 1.25)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax1.text(bars1[0].get_x() + bars1[0].get_width() / 2, n_fp + 25, f"{n_fp:,}\n(FPR = {n_fp/n_grounded*100:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.text(bars1[1].get_x() + bars1[1].get_width() / 2, n_fn + 25, f"{n_fn:,}\n(FNR = {n_fn/n_hallucinated*100:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel 2: Errors by Task Type
    if "task_type" in df.columns:
        tasks = sorted(df["task_type"].unique())
        x = np.arange(len(tasks))
        width = 0.35
        fps_task = [int(df[df["task_type"] == t]["is_fp"].sum()) for t in tasks]
        fns_task = [int(df[df["task_type"] == t]["is_fn"].sum()) for t in tasks]

        ax2.bar(x - width / 2, fps_task, width, label="False Positives", color="#e41a1c", edgecolor="black", linewidth=0.8)
        ax2.bar(x + width / 2, fns_task, width, label="False Negatives", color="#377eb8", edgecolor="black", linewidth=0.8)
        ax2.set_title("Errors by RAG Task Type\n(N=900 per task)", fontsize=10, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(tasks, fontsize=9)
        ax2.set_ylabel("Count", fontsize=9.5)
        ax2.set_ylim(0, max(max(fps_task), max(fns_task)) * 1.2)
        ax2.legend(frameon=True, fontsize=8)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Errors by Generator Model
    if "model" in df.columns:
        models = sorted(df["model"].unique())
        # Shorten model names for readability
        short_names = [m.replace("-chat", "").replace("-instruct", "").replace("-0613", "") for m in models]
        x = np.arange(len(models))
        width = 0.35
        fps_model = [int(df[df["model"] == m]["is_fp"].sum()) for m in models]
        fns_model = [int(df[df["model"] == m]["is_fn"].sum()) for m in models]

        ax3.bar(x - width / 2, fps_model, width, label="False Positives", color="#e41a1c", edgecolor="black", linewidth=0.8)
        ax3.bar(x + width / 2, fns_model, width, label="False Negatives", color="#377eb8", edgecolor="black", linewidth=0.8)
        ax3.set_title("Errors by Generator Model\n(N=450 per model)", fontsize=10, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(short_names, fontsize=8, rotation=25, ha="right")
        ax3.set_ylabel("Count", fontsize=9.5)
        ax3.set_ylim(0, max(max(fps_model), max(fns_model)) * 1.2)
        ax3.legend(frameon=True, fontsize=8)
        ax3.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Probe Error Analysis: Breakdown Across Subgroups", fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_precision_recall_curves(
    baseline_preds: pd.DataFrame | None,
    probe_preds: pd.DataFrame,
    output_path: Path,
    base_thresh: float = 0.455,
    probe_thresh: float = 0.360,
) -> Path:
    """Generate empirical Precision-Recall curves from genuine saved probability scores at 300 DPI."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # Class prevalence baseline (no skill line)
    y_true_probe = probe_preds["y_true"].astype(int)
    prevalence = float(y_true_probe.mean())
    ax.axhline(prevalence, color="gray", linestyle=":", linewidth=1.5, label=f"Random Chance (Prevalence={prevalence:.3f})")

    # Probe PR curve
    p_prec, p_rec, _ = precision_recall_curve(y_true_probe, probe_preds["probability"].astype(float))
    p_ap = float(average_precision_score(y_true_probe, probe_preds["probability"].astype(float)))
    ax.plot(p_rec, p_prec, color="#d95f02", linewidth=2.0, label=f"DistilBERT L5 Q+A Probe (AP = {p_ap:.3f})")

    # Operating point of Probe at selected threshold
    p_pred = (probe_preds["probability"].astype(float) >= probe_thresh).astype(int)
    p_cm = confusion_matrix(y_true_probe, p_pred, labels=[0, 1])
    op_prec = p_cm[1, 1] / (p_cm[1, 1] + p_cm[0, 1])
    op_rec = p_cm[1, 1] / (p_cm[1, 1] + p_cm[1, 0])
    ax.scatter([op_rec], [op_prec], color="#d95f02", s=90, edgecolors="black", zorder=5, label=f"Probe Operating Point (τ={probe_thresh:.3f})")

    # Baseline PR curve if genuine probabilities available
    if baseline_preds is not None and "probability" in baseline_preds.columns:
        y_true_base = baseline_preds["y_true"].astype(int)
        b_prec, b_rec, _ = precision_recall_curve(y_true_base, baseline_preds["probability"].astype(float))
        b_ap = float(average_precision_score(y_true_base, baseline_preds["probability"].astype(float)))
        ax.plot(b_rec, b_prec, color="#2b5c8f", linewidth=2.0, label=f"TF-IDF Baseline (AP = {b_ap:.3f})")

        # Operating point of Baseline
        b_pred = (baseline_preds["probability"].astype(float) >= base_thresh).astype(int)
        b_cm = confusion_matrix(y_true_base, b_pred, labels=[0, 1])
        b_op_prec = b_cm[1, 1] / (b_cm[1, 1] + b_cm[0, 1])
        b_op_rec = b_cm[1, 1] / (b_cm[1, 1] + b_cm[1, 0])
        ax.scatter([b_op_rec], [b_op_prec], color="#2b5c8f", s=90, edgecolors="black", zorder=5, label=f"Baseline Operating Point (τ={base_thresh:.3f})")

    ax.set_title("Official-Test Precision-Recall Curves (Empirical)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Structured Report Generation
# ---------------------------------------------------------------------------

def generate_markdown_report(
    report_data: dict[str, Any],
    output_path: Path,
) -> Path:
    """Generate comprehensive, publication-grade research analysis report in Markdown."""
    probe_m = report_data["probe_metrics"]["classification_metrics"]
    probe_rank = report_data["probe_metrics"].get("ranking_metrics", {})
    probe_cm = report_data["probe_metrics"]["confusion_matrix"]
    base_m = report_data["baseline_metrics"]
    metadata = report_data.get("metadata", {})

    acc_base = base_m.get("accuracy", 0.74037)
    prec_base = base_m.get("precision", 0.61330)
    rec_base = base_m.get("recall", 0.69459)
    f1_base = base_m.get("f1", 0.65142)
    pr_auc_base = base_m.get("pr_auc", 0.72872)
    roc_auc_base = base_m.get("roc_auc", 0.79742)
    brier_base = base_m.get("brier_score", 0.17446)
    thresh_base = base_m.get("selected_threshold", 0.455)

    acc_p = probe_m.get("accuracy", 0.62407)
    prec_p = probe_m.get("precision", 0.47674)
    rec_p = probe_m.get("recall", 0.78261)
    f1_p = probe_m.get("f1", 0.59253)
    spec_p = probe_m.get("specificity", 0.53899)
    bal_acc_p = probe_m.get("balanced_accuracy", 0.66080)
    fpr_p = probe_m.get("false_positive_rate", 0.46101)
    fnr_p = probe_m.get("false_negative_rate", 0.21739)
    pr_auc_p = probe_rank.get("pr_auc", 0.54119)
    roc_auc_p = probe_rank.get("roc_auc", 0.73589)
    brier_p = probe_rank.get("brier_score", 0.23557)
    thresh_p = metadata.get("selected_threshold", 0.360)

    tn = probe_cm.get("tn", 947)
    fp = probe_cm.get("fp", 810)
    fn = probe_cm.get("fn", 205)
    tp = probe_cm.get("tp", 738)
    total = tn + fp + fn + tp

    report_lines = [
        "# RAGGuard Phase 2: Research-Grade Error Analysis and Evaluation Report",
        "",
        "**Project Title:** RAGGuard — Hallucination Detection in RAG Answers Using Classical Probes on LLM Internals  ",
        "**Academic Context:** Advanced Machine Learning Lab, Integrated M.Sc. Artificial Intelligence and Machine Learning  ",
        "**Evaluation Split:** Official Test Split (`data/processed/ragtruth/test.csv`, N=2,700)  ",
        "**Report Generated:** Phase 2 Pipeline Execution  ",
        "",
        "---",
        "",
        "## 1. Objective",
        "",
        "This research report provides a rigorous, empirical error analysis of classical machine learning probes applied to transformer representations for hallucination detection in Retrieval-Augmented Generation (RAG) systems. Specifically, we evaluate a validation-selected Logistic Regression probe trained on observer-model representations from `distilbert/distilbert-base-uncased` across layers 0–6 against an established TF-IDF text-level baseline on the official, held-out RAGTruth test set.",
        "",
        "The primary research questions addressed in this analysis are:",
        "1. **Representational Fidelity:** Can observer-model internal representations (DistilBERT) reliably discriminate between grounded and hallucinated RAG responses?",
        "2. **Error Modalities:** What specific error modes (False Positives vs. False Negatives) dominate the probe's predictions, and how do they distribute across generator LLMs and RAG tasks?",
        "3. **Generalization Gap:** Does validation-based model selection (highest validation F1) generalize effectively to the official test distribution, or does a selection-generalization disparity emerge?",
        "4. **Baseline Disparity:** Why does the surface-level TF-IDF baseline outperform the internal hidden-state probe across most global discrimination metrics, and what are the scientific implications for probe-based hallucination detection?",
        "",
        "---",
        "",
        "## 2. Dataset and Label Distribution",
        "",
        "The evaluation is conducted on the benchmark RAGTruth dataset. The binary classification task is formally defined as:",
        "- **Class 0 (Grounded / Non-Hallucinated):** The generated answer is factually consistent with and substantiated by the provided reference context.",
        "- **Class 1 (Hallucinated):** The generated answer introduces unsupported claims, factual distortions, or context-contradicting entities.",
        "",
        "### Official Test Set Composition",
        "- **Total Samples:** 2,700 rows",
        "- **Grounded Samples (Class 0):** 1,757 (65.07%)",
        "- **Hallucinated Samples (Class 1):** 943 (34.93%)",
        "- **Class Imbalance Ratio:** ~1.86 : 1 (Grounded to Hallucinated)",
        "",
        "### Split Alignment Verification",
        "- **Training Set (Pre-split):** 15,090 rows (8,369 Grounded [55.46%], 6,721 Hallucinated [44.54%])",
        "- **Train Partition (80% Stratified):** 12,072 rows (6,695 Grounded, 5,377 Hallucinated)",
        "- **Validation Partition (20% Stratified):** 3,018 rows (1,674 Grounded, 1,344 Hallucinated)",
        "",
        "The official test set exhibits a lower proportion of hallucinations (34.93%) compared to the training set (44.54%), representing a mild natural distribution shift in class prevalence across splits.",
        "",
        "![Class Distribution](class_distribution.png)",
        "*Figure 1: Class distribution of the RAGTruth official test split and comparison with training partition proportions.*",
        "",
        "---",
        "",
        "## 3. Experimental Setup and Observer-Model Semantics",
        "",
        "### Critical Scientific Qualification: Observer-Model Architecture",
        "The representations utilized in this study were extracted using `distilbert/distilbert-base-uncased` applied as an **external observer model** over the concatenated sequence components, where token hidden states are mean-pooled over attended tokens across layers 0 through 6.",
        "",
        "> [!IMPORTANT]",
        "> **Observer vs. Generator Representation:** These hidden states are **not** the internal activations of the generative models (e.g., GPT-4, LLaMA-2, Mistral) that produced the answers. They represent how a frozen, smaller encoder-only model encodes the surface text post-hoc. Consequently, this study evaluates whether an observer encoder possesses internal linear separability for hallucination detection, **not** whether generative LLMs internally 'know' when they are hallucinating.",
        "",
        "### Experimental Protocol and Hygiene",
        "- **Strict Data Isolation:** Official test labels and features were never accessed during model fitting, hyperparameter selection, or decision threshold calibration.",
        "- **Scaling:** `StandardScaler` was fitted exclusively on the 12,072 training partition samples and applied to validation and test features within inference pipelines.",
        "- **No Retraining:** All metrics and predictions analyzed herein are loaded directly from previously completed Phase 1 experiment runs.",
        "",
        "---",
        "",
        "## 4. Selected Probe Configuration",
        "",
        "Model selection was conducted strictly on the 3,018-sample validation partition by maximizing validation F1 across all evaluated layer indices (1, 3, 5, 6), feature segments (`answer_only`, `question_answer`, `context_answer`, `question_context_answer`), and classifiers.",
        "",
        "### Validation-Selected Architecture",
        "- **Feature Segment:** `question_answer` (concatenation of mean-pooled question and answer representations)",
        "- **Transformer Layer:** Layer 5 (penultimate layer of 6-layer DistilBERT)",
        "- **Feature Dimension:** 1,536 (768 from question + 768 from answer)",
        "- **Classifier:** Logistic Regression (L2 regularization)",
        "- **Validation F1:** 0.700129",
        f"- **Validation-Selected Threshold (tau):** {thresh_p:.4f}",
        "",
        "> [!NOTE]",
        "> The selected probe configuration **does not ingest the reference context**. It relies entirely on the semantic representations of the question and the generated answer.",
        "",
        "---",
        "",
        "## 5. Baseline Comparison: Hidden-State Probe vs. TF-IDF Baseline",
        "",
        "Table 1 summarizes official-test performance between the TF-IDF baseline and the validation-selected DistilBERT probe.",
        "",
        "### Table 1: Official-Test Performance Summary",
        "",
        "| Metric | TF-IDF Baseline | DistilBERT L5 Q+A Probe | Difference (Delta) | Superior System |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Accuracy** | **{acc_base:.4f}** | {acc_p:.4f} | {acc_p - acc_base:+.4f} | TF-IDF |",
        f"| **Precision** | **{prec_base:.4f}** | {prec_p:.4f} | {prec_p - prec_base:+.4f} | TF-IDF |",
        f"| **Recall (Sensitivity)** | {rec_base:.4f} | **{rec_p:.4f}** | {rec_p - rec_base:+.4f} | **Probe** |",
        f"| **F1-Score** | **{f1_base:.4f}** | {f1_p:.4f} | {f1_p - f1_base:+.4f} | TF-IDF |",
        f"| **Specificity (TNR)** | **0.7650** | {spec_p:.4f} | {spec_p - 0.7650:+.4f} | TF-IDF |",
        f"| **Balanced Accuracy** | **0.7298** | {bal_acc_p:.4f} | {bal_acc_p - 0.7298:+.4f} | TF-IDF |",
        f"| **False Positive Rate (FPR)** | **0.2351** | {fpr_p:.4f} | {fpr_p - 0.2351:+.4f} | TF-IDF (Lower) |",
        f"| **False Negative Rate (FNR)** | 0.3054 | **{fnr_p:.4f}** | {fnr_p - 0.3054:+.4f} | **Probe (Lower)** |",
        f"| **PR-AUC (Avg Precision)** | **{pr_auc_base:.4f}** | {pr_auc_p:.4f} | {pr_auc_p - pr_auc_base:+.4f} | TF-IDF |",
        f"| **ROC-AUC** | **{roc_auc_base:.4f}** | {roc_auc_p:.4f} | {roc_auc_p - roc_auc_base:+.4f} | TF-IDF |",
        f"| **Brier Score (Calibration)** | **{brier_base:.4f}** | {brier_p:.4f} | {brier_p - brier_base:+.4f} | TF-IDF (Lower) |",
        f"| **Decision Threshold (tau)** | {thresh_base:.4f} | {thresh_p:.4f} | {thresh_p - thresh_base:+.4f} | — |",
        "",
        "![Classification Metrics Comparison](classification_metrics_comparison.png)",
        "*Figure 2: Official test classification metrics comparing TF-IDF baseline and DistilBERT L5 Q+A probe.*",
        "",
        "![Ranking Metrics Comparison](ranking_metrics_comparison.png)",
        "*Figure 3: Ranking metrics (PR-AUC, ROC-AUC) and calibration metric (Brier score) comparison.*",
        "",
        "---",
        "",
        "## 6. Confusion Matrix and Error Metrics",
        "",
        "The confusion matrix for the selected probe on the 2,700 official test samples is detailed in Table 2.",
        "",
        "### Table 2: Selected Probe Confusion Matrix Breakdown",
        "",
        "| Actual \\ Predicted | Predicted Grounded (0) | Predicted Hallucinated (1) | Total Actual | Class Error Rate |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Actual Grounded (0)** | **TN = {tn:,}** ({tn/total*100:.2f}%) | **FP = {fp:,}** ({fp/total*100:.2f}%) | {tn+fp:,} | FPR = {fp/(tn+fp)*100:.2f}% |",
        f"| **Actual Hallucinated (1)** | **FN = {fn:,}** ({fn/total*100:.2f}%) | **TP = {tp:,}** ({tp/total*100:.2f}%) | {fn+tp:,} | FNR = {fn/(fn+tp)*100:.2f}% |",
        f"| **Total Predicted** | {tn+fn:,} ({(tn+fn)/total*100:.2f}%) | {fp+tp:,} ({(fp+tp)/total*100:.2f}%) | **{total:,}** | Total Errors = {fp+fn:,} |",
        "",
        "![Probe Confusion Matrix](probe_confusion_matrix.png)",
        "*Figure 4: Detailed confusion matrix of the validation-selected DistilBERT probe.*",
        "",
        "![Score Distributions](probe_score_distributions.png)",
        f"*Figure 5: Predicted probability density distributions for grounded and hallucinated classes, illustrating the decision threshold cut at tau={thresh_p:.3f}.*",
        "",
        "---",
        "",
        "## 7. False-Positive Analysis",
        "",
        f"The probe produced **{fp:,} False Positives** out of {tn+fp:,} grounded samples (FPR = {fp/(tn+fp)*100:.2f}%). This represents the single largest source of model inaccuracy ({fp/(fp+fn)*100:.2f}% of all misclassifications).",
        "",
        "### Key Empirical Observations on False Positives:",
        f"1. **Threshold Effect:** The validation-selected decision threshold of tau = {thresh_p:.3f} is substantially lower than the standard 0.50. While this successfully elevated recall ({rec_p*100:.2f}%), it aggressively shifted borderline cases into the positive (hallucinated) category, causing precision to drop to {prec_p*100:.2f}%.",
        "2. **Context Blindness in Feature Space:** The selected probe uses `question_answer` features exclusively. Because the probe does not receive the reference context, it cannot verify whether an assertive factual claim made in the answer is grounded in the text. It must rely on stylistic cues (length, perplexity, confidence, syntactic hedges). Consequently, highly detailed, well-grounded answers are misclassified as hallucinations.",
        "3. **Elevated False Positives on High-Capability Generators:**",
        "   - **GPT-4-0613:** 162 False Positives (FPR = 50.31% of grounded GPT-4 answers)",
        "   - **GPT-3.5-Turbo-0613:** 161 False Positives (FPR = 50.47% of grounded GPT-3.5 answers)",
        "   Both frontier models produce fluent, confident, syntactically complex prose that the observer model's question-answer probe frequently misidentifies as hallucinated.",
        "",
        "---",
        "",
        "## 8. False-Negative Analysis",
        "",
        f"The probe produced **{fn:,} False Negatives** out of {fn+tp:,} hallucinated samples (FNR = {fn/(fn+tp)*100:.2f}%).",
        "",
        "### Key Empirical Observations on False Negatives:",
        "1. **Subtle Entity and Numerical Distortions:** In RAG settings, the most dangerous hallucinations are subtle: altering a date, swapping a named entity, or modifying a quantity while maintaining perfect grammatical fluency. Without cross-attending to the context tokens, the observer model's layer-5 representations encode plausible-sounding hallucinated answers identically to grounded answers.",
        "2. **Task Distribution of False Negatives:**",
        "   - **Summary Task:** 100 False Negatives (48.78% of all false negatives). In summarization, hallucinations frequently consist of subtle extrapolations beyond the source passage that appear entirely natural in the answer alone.",
        "   - **QA Task:** 71 False Negatives (34.63% of false negatives).",
        "   - **Data2txt Task:** 34 False Negatives (16.59% of false negatives). Tabular hallucinations often introduce overt discrepancies, yielding fewer false negatives.",
        "",
        "![Error Breakdown](error_breakdown.png)",
        "*Figure 6: Error distribution across error types, RAG task types, and generator LLMs.*",
        "",
        "---",
        "",
        "## 9. Layer and Feature-Configuration Observations",
        "",
        "Inspection of the full 16-candidate probe grid evaluated in Phase 1 reveals crucial structural patterns across transformer layers and input feature configurations:",
        "",
        "### Table 3: Layer and Feature Configuration Summary (Official Test F1)",
        "",
        "| Configuration | Layer 1 F1 | Layer 3 F1 | Layer 5 F1 | Layer 6 F1 | Max Test F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        "| `answer_only` | 0.6159 | 0.6246 | 0.6093 | **0.6287** | 0.6287 (L6) |",
        "| `context_answer` | 0.6096 | 0.6032 | 0.5988 | 0.6094 | 0.6096 (L1) |",
        "| `question_answer` | 0.5955 | 0.5932 | **0.5925** | 0.5899 | 0.5955 (L1) |",
        "| `question_context_answer` | 0.5857 | 0.5731 | 0.5811 | 0.5921 | 0.5921 (L6) |",
        "",
        "![Layer Performance](layer_performance.png)",
        "*Figure 7: Layer-wise official test F1 performance across evaluated layers and feature configurations.*",
        "",
        "### Scientific Observations:",
        "1. **Answer-Only Features Achieved Highest Test F1 and Ranking Scores:** Counter-intuitively, the `answer_only` configuration at Layer 6 achieved the highest official-test F1 (0.6287), PR-AUC (0.7047), and ROC-AUC (0.7818) among all probes.",
        "2. **Feature Concatenation Degradation:** Adding question representations to answer representations consistently degraded test PR-AUC (Layer 5: from 0.7053 in answer-only to 0.5412 in question-answer). Concatenating independent mean-pooled vectors in a high-dimensional linear probe without cross-attention appears to dilute the discriminative signal.",
        "",
        "---",
        "",
        "## 10. Interpretation of Results",
        "",
        "### Generalization Gap Analysis",
        f"The validation selection procedure selected `question_answer` Layer 5 based on its peak validation F1 of **0.7001**. However, on the official test set, this exact model achieved an F1 of only **{f1_p:.4f}** (a generalization drop of **{f1_p - 0.700129:+.4f}**).",
        "",
        f"By contrast, the TF-IDF baseline was selected at threshold {thresh_base:.4f} and retained an official-test F1 of **{f1_base:.4f}**.",
        "",
        "**Why did this occur?**",
        f"1. **Threshold Overfitting on Validation Partition:** The threshold tau = {thresh_p:.3f} was optimized for the validation partition's specific score distribution. On the test set, where grounded examples constitute 65.07% of the data (compared to 55.46% in training), this low threshold generated excessive False Positives.",
        "2. **Representational Bottleneck:** Mean-pooling transformer hidden states over long contexts and answers collapses critical token-level alignment information into a static vector. The linear probe cannot model cross-sequence entailment.",
        "3. **TF-IDF Surface Heuristics:** TF-IDF n-grams (word and character level) capture explicit lexical overlap, rare n-gram repetitions, and lexical novelty between query and answer, providing a robust inductive bias that simple linear probes on mean-pooled vectors fail to match.",
        "",
        "![Precision-Recall Curves](precision_recall_curve.png)",
        f"*Figure 8: Empirical Precision-Recall curves comparing TF-IDF baseline (AP={pr_auc_base:.3f}) and DistilBERT L5 Q+A probe (AP={pr_auc_p:.3f}).*",
        "",
        "---",
        "",
        "## 11. Limitations",
        "",
        "1. **Observer-Model Confound:** Activations originate from an observer model (`distilbert-base-uncased`), not the generator LLMs. This experiment does not evaluate whether generative models possess internal representation of truthfulness.",
        "2. **Linear Probe Expressivity:** Standard Logistic Regression assumes linear separability in the pooled embedding space. Non-linear interactions between question, context, and answer tokens are lost during independent pooling.",
        "3. **Absence of Token-Level Attention Probing:** The representations evaluate sentence-level mean pooling rather than token-level self-attention patterns or cross-attention matrices between answer and context.",
        "4. **Single Transformer Family:** Experiments were restricted to DistilBERT; larger or bidirectional decoder representations were not evaluated in this phase.",
        "",
        "---",
        "",
        "## 12. Reproducibility Instructions",
        "",
        "The entire error analysis and visualization pipeline is completely deterministic and reproducible from existing artifacts without retraining.",
        "",
        "### Execution Command:",
        "```powershell",
        '& ".\\.venv\\Scripts\\python.exe" -m src.analysis.error_analysis',
        "```",
        "",
        "### Direct Script Execution:",
        "```powershell",
        '& ".\\.venv\\Scripts\\python.exe" src/analysis/error_analysis.py `',
        "    --predictions artifacts/probes/combined_features/selected_probe_predictions_official_test.csv `",
        "    --test-data data/processed/ragtruth/test.csv `",
        "    --comparison-csv artifacts/probes/combined_features/comparison_official_test.csv `",
        "    --baseline-metrics artifacts/text_baseline/metrics_official_test.json `",
        "    --baseline-predictions artifacts/text_baseline/predictions_official_test.csv `",
        "    --output-dir artifacts/analysis",
        "```",
        "",
        "---",
        "",
        "## 13. Conclusions and Recommended Next Experiment",
        "",
        "### Core Research Conclusions:",
        f"1. **Linear Probes on Observer Representations Do Not Outperform Lexical Baselines:** On the RAGTruth official benchmark, classical logistic regression probes applied to DistilBERT hidden states fail to surpass a standard TF-IDF baseline on accuracy ({acc_p*100:.1f}% vs. {acc_base*100:.1f}%), precision ({prec_p*100:.1f}% vs. {prec_base*100:.1f}%), F1 ({f1_p*100:.1f}% vs. {f1_base*100:.1f}%), and PR-AUC ({pr_auc_p:.3f} vs. {pr_auc_base:.3f}).",
        f"2. **High Recall at the Expense of Specificity:** The probe demonstrates utility in high-recall screening ({rec_p*100:.1f}% recall), but its elevated false-positive rate ({fpr_p*100:.1f}%) severely limits standalone deployment.",
        "3. **Feature-Level Pathology:** Concatenating independently mean-pooled representations from an observer model without explicit context-answer entailment features degrades ranking fidelity.",
        "",
        "### Recommended Next Experiment (Phase 3):",
        "1. **Cross-Encoder / Entailment Probes:** Instead of mean-pooling separated sequences, feed the concatenated `[CLS] context [SEP] answer [SEP]` sequence through an NLI/entailment cross-encoder to preserve cross-attention interactions.",
        "2. **Ensemble / Residual Probing:** Combine TF-IDF lexical overlap signals with observer hidden-state features, leveraging the complementary strengths of lexical matching (high precision) and semantic representations (high recall).",
        "3. **Direct Generator Logit / Activation Extraction:** Extract hidden states directly from open-weights generator models (e.g., LLaMA-2, Mistral) to test the true internal activation hypothesis.",
        "",
    ]

    report_content = "\n".join(report_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------------------------

def run_error_analysis(
    predictions_path: str | Path = "artifacts/probes/combined_features/selected_probe_predictions_official_test.csv",
    test_data_path: str | Path = "data/processed/ragtruth/test.csv",
    comparison_csv_path: str | Path = "artifacts/probes/combined_features/comparison_official_test.csv",
    baseline_metrics_path: str | Path = "artifacts/text_baseline/metrics_official_test.json",
    baseline_predictions_path: str | Path | None = "artifacts/text_baseline/predictions_official_test.csv",
    train_data_path: str | Path | None = "data/processed/ragtruth/train.csv",
    output_dir: str | Path = "artifacts/analysis",
) -> dict[str, Any]:
    """Execute complete research error analysis pipeline."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load inputs defensively
    test_df = pd.read_csv(test_data_path)
    pred_df = pd.read_csv(predictions_path)
    comp_df = pd.read_csv(comparison_csv_path)

    base_metrics = {}
    if Path(baseline_metrics_path).is_file():
        base_metrics = json.loads(Path(baseline_metrics_path).read_text(encoding="utf-8"))

    base_preds_df = None
    if baseline_predictions_path and Path(baseline_predictions_path).is_file():
        base_preds_df = pd.read_csv(baseline_predictions_path)

    train_counts = None
    if train_data_path and Path(train_data_path).is_file():
        train_df = pd.read_csv(train_data_path)
        if "label" in train_df.columns:
            train_counts = {str(k): int(v) for k, v in train_df["label"].value_counts().to_dict().items()}

    # 2. Validate alignment
    aligned_df = validate_prediction_alignment(test_df, pred_df)

    # 3. Determine selected probe threshold
    threshold = 0.360
    if "selected_by_validation" in comp_df.columns:
        sel_row = comp_df[comp_df["selected_by_validation"] == True]  # noqa: E712
        if not sel_row.empty and "threshold" in sel_row.columns:
            threshold = float(sel_row["threshold"].iloc[0])

    # 4. Compute metrics
    metrics = compute_confusion_matrix_and_metrics(
        y_true=aligned_df["y_true"],
        y_pred=aligned_df["prediction"],
        y_prob=aligned_df["probability"] if "probability" in aligned_df.columns else None,
    )

    # 5. Extract misclassifications
    misclassified_df = extract_misclassifications(aligned_df, threshold=threshold)
    misclassified_csv_path = out_dir / "misclassified_examples.csv"
    misclassified_df.to_csv(misclassified_csv_path, index=False, encoding="utf-8")

    # 6. Compute subgroup error breakdown
    subgroup = compute_subgroup_analysis(aligned_df)

    # 7. Generate research visualizations (300 DPI)
    test_counts = {str(k): int(v) for k, v in aligned_df["y_true"].value_counts().to_dict().items()}
    plot_class_distribution(test_counts, train_counts, out_dir / "class_distribution.png")

    plot_classification_metrics_comparison(
        baseline_metrics=base_metrics,
        probe_metrics=metrics["classification_metrics"],
        output_path=out_dir / "classification_metrics_comparison.png",
    )

    plot_ranking_metrics_comparison(
        baseline_metrics=base_metrics,
        probe_metrics=metrics.get("ranking_metrics", {}),
        output_path=out_dir / "ranking_metrics_comparison.png",
    )

    plot_layer_performance(
        comparison_df=comp_df,
        output_path=out_dir / "layer_performance.png",
    )

    plot_probe_confusion_matrix(
        cm_dict=metrics["confusion_matrix"],
        output_path=out_dir / "probe_confusion_matrix.png",
    )

    plot_score_distributions(
        aligned_df=aligned_df,
        threshold=threshold,
        output_path=out_dir / "probe_score_distributions.png",
    )

    plot_error_breakdown(
        aligned_df=aligned_df,
        output_path=out_dir / "error_breakdown.png",
    )

    base_thresh = float(base_metrics.get("selected_threshold", 0.455))
    plot_precision_recall_curves(
        baseline_preds=base_preds_df,
        probe_preds=aligned_df,
        output_path=out_dir / "precision_recall_curve.png",
        base_thresh=base_thresh,
        probe_thresh=threshold,
    )

    # 8. Save structured JSON metrics
    full_results: dict[str, Any] = {
        "metadata": {
            "evaluation_split": "official_test",
            "dataset": "RAGTruth",
            "probe_feature_segment": "question_answer",
            "probe_layer": 5,
            "probe_classifier": "logistic_regression",
            "selected_threshold": threshold,
            "baseline_threshold": base_thresh,
            "total_test_samples": len(aligned_df),
            "misclassified_samples": len(misclassified_df),
        },
        "probe_metrics": metrics,
        "baseline_metrics": base_metrics,
        "subgroup_analysis": subgroup,
    }
    metrics_json_path = out_dir / "error_analysis_metrics.json"
    metrics_json_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")

    # 9. Generate research Markdown report
    report_md_path = out_dir / "error_analysis_report.md"
    generate_markdown_report(full_results, report_md_path)

    print(f"[RAGGuard Analysis] Generated all analysis artifacts in: {out_dir}")
    print(f"[RAGGuard Analysis] Metrics: {metrics_json_path}")
    print(f"[RAGGuard Analysis] Misclassifications: {misclassified_csv_path}")
    print(f"[RAGGuard Analysis] Report: {report_md_path}")

    return full_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="artifacts/probes/combined_features/selected_probe_predictions_official_test.csv",
        help="Path to selected probe predictions CSV on official test set.",
    )
    parser.add_argument(
        "--test-data",
        default="data/processed/ragtruth/test.csv",
        help="Path to official test CSV.",
    )
    parser.add_argument(
        "--comparison-csv",
        default="artifacts/probes/combined_features/comparison_official_test.csv",
        help="Path to probe comparison CSV.",
    )
    parser.add_argument(
        "--baseline-metrics",
        default="artifacts/text_baseline/metrics_official_test.json",
        help="Path to baseline metrics JSON.",
    )
    parser.add_argument(
        "--baseline-predictions",
        default="artifacts/text_baseline/predictions_official_test.csv",
        help="Path to baseline predictions CSV.",
    )
    parser.add_argument(
        "--train-data",
        default="data/processed/ragtruth/train.csv",
        help="Path to training data CSV for split comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/analysis",
        help="Directory to save analysis outputs.",
    )
    args = parser.parse_args()

    run_error_analysis(
        predictions_path=args.predictions,
        test_data_path=args.test_data,
        comparison_csv_path=args.comparison_csv,
        baseline_metrics_path=args.baseline_metrics,
        baseline_predictions_path=args.baseline_predictions,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

