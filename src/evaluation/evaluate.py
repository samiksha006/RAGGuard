"""Generate standard metrics and plots from a predictions CSV."""
import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def evaluate_predictions(
    predictions_path: str | Path,
    output_dir: str | Path,
    split_name: str | None = None,
) -> dict[str, Any]:
    df = pd.read_csv(predictions_path)
    required = {"y_true", "probability", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{predictions_path}: missing required columns: {', '.join(sorted(missing))}"
        )
    if df.empty:
        raise ValueError(f"{predictions_path}: no predictions to evaluate")

    y = df["y_true"].astype(int)
    probabilities = df["probability"].astype(float)
    predictions = df["prediction"].astype(int)
    suffix = f"_{split_name}" if split_name else ""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "classification_report": classification_report(
            y, predictions, output_dict=True, zero_division=0
        ),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "evaluation_split": split_name or "unspecified",
        "predictions_file": str(predictions_path),
        "n_test": len(df),
    }
    if y.nunique() == 2:
        report["roc_auc"] = float(roc_auc_score(y, probabilities))
        report["pr_auc"] = float(average_precision_score(y, probabilities))

    metrics_path = output / f"evaluation_metrics{suffix}.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    figure, axis = plt.subplots()
    axis.imshow(matrix)
    axis.set(
        title=f"Confusion matrix{f' - {split_name}' if split_name else ''}",
        xlabel="Predicted",
        ylabel="Actual",
    )
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output / f"confusion_matrix{suffix}.png", dpi=160)
    plt.close(figure)

    if y.nunique() == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(y, probabilities)
        figure, axis = plt.subplots()
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"AUC={auc(false_positive_rate, true_positive_rate):.3f}",
        )
        axis.plot([0, 1], [0, 1], "--")
        axis.set(
            xlabel="False positive rate",
            ylabel="True positive rate",
            title=f"ROC curve{f' - {split_name}' if split_name else ''}",
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / f"roc_curve{suffix}.png", dpi=160)
        plt.close(figure)

        precision, recall, _ = precision_recall_curve(y, probabilities)
        figure, axis = plt.subplots()
        axis.plot(
            recall,
            precision,
            label=f"AP={average_precision_score(y, probabilities):.3f}",
        )
        axis.set(
            xlabel="Recall",
            ylabel="Precision",
            title=f"Precision-recall curve{f' - {split_name}' if split_name else ''}",
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / f"pr_curve{suffix}.png", dpi=160)
        plt.close(figure)

        observed, mean_probability = calibration_curve(
            y, probabilities, n_bins=10, strategy="uniform"
        )
        figure, axis = plt.subplots()
        axis.plot(mean_probability, observed, "o-", label="Model")
        axis.plot([0, 1], [0, 1], "--", label="Ideal")
        axis.set(
            xlabel="Mean predicted probability",
            ylabel="Observed positive fraction",
            title=f"Reliability diagram{f' - {split_name}' if split_name else ''}",
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / f"reliability{suffix}.png", dpi=160)
        plt.close(figure)

    if "answer" in df:
        df.loc[predictions != y].to_csv(
            output / f"error_analysis{suffix}.csv", index=False
        )
    print(json.dumps(report, indent=2))
    print(f"Evaluation outputs saved to {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="reports/results")
    parser.add_argument(
        "--split-name",
        help="Include the split name in all output filenames and report metadata.",
    )
    args = parser.parse_args()
    evaluate_predictions(args.predictions, args.output_dir, args.split_name)


if __name__ == "__main__":
    main()
