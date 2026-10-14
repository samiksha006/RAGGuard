
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COMPARISON_FILE = ROOT / "artifacts/comparison/validation_comparison.json"
TUNING_FILE = ROOT / "artifacts/tuning/experiment_metadata.json"
OUTPUT_DIR = ROOT / "artifacts/comparison"


def main():
    comparison = json.loads(
        COMPARISON_FILE.read_text(encoding="utf-8")
    )
    tuning = json.loads(
        TUNING_FILE.read_text(encoding="utf-8")
    )

    baseline = comparison["original_baseline"]
    tuned = comparison["tuned_model"]

    metrics = [
        ("Accuracy", "accuracy", True),
        ("Precision", "precision", True),
        ("Recall", "recall", True),
        ("F1-score", "f1", True),
        ("PR-AUC", "pr_auc", True),
        ("ROC-AUC", "roc_auc", True),
        ("Brier score", "brier_score", False),
    ]

    rows = []
    for name, key, higher_is_better in metrics:
        old = baseline[key]
        new = tuned[key]
        improvement = new - old
        rows.append({
            "Metric": name,
            "Original baseline": old,
            "Tuned model": new,
            "Absolute change": improvement,
            "Better model": (
                "Tuned" if (improvement > 0) == higher_is_better
                else "Baseline" if improvement != 0
                else "Tie"
            ),
        })

    table = pd.DataFrame(rows)

    report = [
        "# RAGGuard: Model Tuning and Benchmarking Report",
        "",
        "## 1. Objective",
        "",
        "Compare the original word-and-character TF-IDF Logistic "
        "Regression baseline against the tuned model for hallucination "
        "detection in RAG answers.",
        "",
        "## 2. Experimental setup",
        "",
        "- Dataset: RAGTruth processed training dataset.",
        "- Split: stratified 80% training and 20% validation.",
        "- Random seed: 42.",
        "- Evaluation threshold: 0.5 for both models.",
        "- Model selection: three-fold cross-validation using F1-score.",
        "- Tuned search configurations requested: "
        f"{tuning['iterations_requested']}.",
        "- Cross-validation fits completed: "
        f"{tuning['iterations_completed']}.",
        "",
        "## 3. Best tuned hyperparameters",
        "",
        "```json",
        json.dumps(tuning["best_parameters"], indent=2),
        "```",
        "",
        "## 4. Validation results",
        "",
        table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 5. Interpretation",
        "",
        f"- Accuracy increased from {baseline['accuracy']:.4f} "
        f"to {tuned['accuracy']:.4f}.",
        f"- F1-score increased from {baseline['f1']:.4f} "
        f"to {tuned['f1']:.4f}.",
        f"- PR-AUC increased from {baseline['pr_auc']:.4f} "
        f"to {tuned['pr_auc']:.4f}.",
        f"- Recall increased from {baseline['recall']:.4f} "
        f"to {tuned['recall']:.4f}, while precision changed from "
        f"{baseline['precision']:.4f} to {tuned['precision']:.4f}.",
        "- The Brier score should be minimized; it was slightly lower "
        "for the original baseline.",
        "",
        "## 6. Limitations",
        "",
        "- These are internal validation results, not a final independent "
        "test-set estimate.",
        "- The tuned model was selected through cross-validation, so its "
        "validation results should be interpreted cautiously.",
        "- No claim of statistical significance is made.",
        "- Results depend on the dataset, split, threshold, and chosen metrics.",
        "",
        "## 7. Conclusion",
        "",
        "The tuned model improved F1-score, recall, PR-AUC, ROC-AUC, and "
        "accuracy on the shared validation split. Precision decreased "
        "slightly, and the Brier score was marginally worse. The tuned "
        "model is promising, but an independent evaluation is needed "
        "before claiming generalization improvement.",
        "",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "model_comparison_report.md"
    output_path.write_text("\n".join(report), encoding="utf-8")

    print(f"Report saved to: {output_path}")
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
