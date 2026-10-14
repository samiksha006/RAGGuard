"""Create dataset overview plots and metadata for a supplied predictions file."""
import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_report(
    data_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    split_name: str | None = None,
) -> dict[str, Any]:
    data = pd.read_csv(data_path)
    required = {"label", "question", "context", "answer"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"{data_path}: missing required columns: {', '.join(sorted(missing))}"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{split_name}" if split_name else ""

    counts = data["label"].value_counts().sort_index().to_dict()
    figure, axis = plt.subplots()
    data["label"].value_counts().sort_index().plot(kind="bar", ax=axis)
    axis.set(
        title=f"Response-level class distribution{f' - {split_name}' if split_name else ''}",
        xlabel="Label (0=non-hallucinated, 1=hallucinated)",
        ylabel="Rows",
    )
    figure.tight_layout()
    figure.savefig(output / f"class_distribution{suffix}.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots()
    data[["question", "context", "answer"]].fillna("").astype(str).apply(
        lambda column: column.str.len()
    ).plot(kind="box", ax=axis)
    axis.set(
        title=f"Text length distributions{f' - {split_name}' if split_name else ''}",
        ylabel="Characters",
    )
    figure.tight_layout()
    figure.savefig(output / f"text_length_boxplot{suffix}.png", dpi=160)
    plt.close(figure)

    pca_status = "omitted for official test to avoid fitting a projection on test data"
    if split_name != "official_test":
        try:
            from sklearn.decomposition import PCA
            from sklearn.feature_extraction.text import TfidfVectorizer

            text = (
                data["question"].fillna("").astype(str)
                + " "
                + data["context"].fillna("").astype(str)
                + " "
                + data["answer"].fillna("").astype(str)
            ).tolist()
            features = TfidfVectorizer(max_features=2000).fit_transform(text).toarray()
            embedding = PCA(n_components=2, random_state=42).fit_transform(features)
            figure, axis = plt.subplots()
            scatter = axis.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=data["label"],
                cmap="coolwarm",
                s=10,
                alpha=0.6,
            )
            axis.set(
                title=f"PCA projection of TF-IDF features{f' - {split_name}' if split_name else ''}",
                xlabel="PC1",
                ylabel="PC2",
            )
            figure.colorbar(scatter, ax=axis, label="Label")
            figure.tight_layout()
            figure.savefig(output / f"pca_projection{suffix}.png", dpi=160)
            plt.close(figure)
            pca_status = "generated"
        except (ValueError, ImportError) as exc:
            pca_status = f"not generated: {type(exc).__name__}: {exc}"

    report: dict[str, Any] = {
        "evaluation_split": split_name or "unspecified",
        "data_file": str(data_path),
        "predictions_file": str(predictions_path),
        "n_rows": len(data),
        "class_counts": {str(label): int(count) for label, count in counts.items()},
        "pca_visualization": pca_status,
    }
    report_path = output / f"dataset_report{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Dataset report saved to {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="reports/results")
    parser.add_argument("--split-name")
    args = parser.parse_args()
    make_report(args.data, args.predictions, args.output_dir, args.split_name)


if __name__ == "__main__":
    main()
