
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "artifacts/comparison/validation_comparison.json"
OUTPUT_DIR = ROOT / "artifacts/comparison/plots"


def main():
    results = json.loads(INPUT.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = results["original_baseline"]
    tuned = results["tuned_model"]

    metrics = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC"]

    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, [baseline[m] for m in metrics], width, label="Original baseline")
    ax.bar(x + width / 2, [tuned[m] for m in metrics], width, label="Tuned model")

    ax.set_title("RAGGuard: Original vs Tuned Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_metrics_comparison.png", dpi=200)
    plt.close(fig)

    # Brier score: lower is better, so show separately.
    fig, ax = plt.subplots(figsize=(7, 5))
    values = [baseline["brier_score"], tuned["brier_score"]]
    bars = ax.bar(["Original baseline", "Tuned model"], values)
    ax.set_title("Brier Score Comparison (Lower Is Better)")
    ax.set_ylabel("Brier score")
    ax.bar_label(bars, fmt="%.4f")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "brier_score_comparison.png", dpi=200)
    plt.close(fig)

    print("Saved visualization files:")
    print(OUTPUT_DIR / "model_metrics_comparison.png")
    print(OUTPUT_DIR / "brier_score_comparison.png")


if __name__ == "__main__":
    main()
