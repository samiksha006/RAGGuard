"""Train a TF-IDF + Logistic Regression baseline without test-set leakage."""
import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


def metrics(y_true: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else None
        ),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"answer", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")
    if df["label"].isna().any():
        raise ValueError(f"{path}: label contains missing values")
    labels = pd.to_numeric(df["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{path}: labels must be binary 0/1")
    df = df.copy()
    df["label"] = labels.astype(int)
    if df.empty:
        raise ValueError(f"{path}: dataset is empty")
    return df.reset_index(drop=True)


def make_text(df: pd.DataFrame) -> list[str]:
    question = df["question"] if "question" in df else pd.Series("", index=df.index)
    context = df["context"] if "context" in df else pd.Series("", index=df.index)
    return (
        question.fillna("").astype(str)
        + " [CONTEXT] "
        + context.fillna("").astype(str)
        + " [ANSWER] "
        + df["answer"].fillna("").astype(str)
    ).tolist()


def build_model(seed: int) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=80_000,
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=40_000,
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def split_training_indices(
    labels: np.ndarray, validation_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=seed,
        stratify=labels,
    )
    return train_indices, validation_indices


def select_threshold(y_validation: np.ndarray, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    return float(
        max(
            thresholds,
            key=lambda threshold: f1_score(
                y_validation, probabilities >= threshold, zero_division=0
            ),
        )
    )


def check_split_ids(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    id_column = "id" if "id" in train_df and "id" in test_df else "row_id"
    if id_column not in train_df or id_column not in test_df:
        return
    overlap = set(train_df[id_column].dropna()) & set(test_df[id_column].dropna())
    if overlap:
        raise ValueError(
            f"Training and test CSVs have {len(overlap)} overlapping values in {id_column}; "
            "refusing official-test evaluation."
        )


def fit_and_score(
    training_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    validation_size: float,
    split_name: str,
) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    """Fit on a training-only partition, tune on its validation rows, then score test once."""
    training_labels = training_df["label"].to_numpy()
    train_indices, validation_indices = split_training_indices(
        training_labels, validation_size, seed
    )
    model, predictions_df, result = fit_with_explicit_validation(
        training_df.iloc[train_indices].reset_index(drop=True),
        training_df.iloc[validation_indices].reset_index(drop=True),
        test_df,
        seed=seed,
        split_name=split_name,
    )
    result["n_training_rows"] = len(training_df)
    result["validation_fraction_of_training"] = validation_size
    result["split_strategy"] = "stratified_train_validation"
    return model, predictions_df, result


def save_outputs(
    model: Pipeline,
    predictions: pd.DataFrame,
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    official = result["test_split"] == "official_test"
    predictions_name = "predictions_official_test.csv" if official else "predictions.csv"
    metrics_name = "metrics_official_test.json" if official else "metrics.json"
    joblib.dump(model, output_dir / "model.joblib")
    predictions.to_csv(output_dir / predictions_name, index=False)
    result_text = json.dumps(result, indent=2)
    (output_dir / metrics_name).write_text(result_text, encoding="utf-8")
    if official:
        predictions.to_csv(output_dir / "predictions.csv", index=False)
        (output_dir / "metrics.json").write_text(result_text, encoding="utf-8")
    (output_dir / "selected_threshold.json").write_text(
        json.dumps(
            {
                "threshold": result["selected_threshold"],
                "selected_on": "training_validation",
                "test_split": result["test_split"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="Legacy single CSV; creates internal holdout splits.")
    inputs.add_argument("--train-input", help="Training CSV (official train split).")
    parser.add_argument("--test-input", help="Untouched official test CSV.")
    parser.add_argument("--output-dir", default="artifacts/text_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--group-column", default="")
    args = parser.parse_args()
    if not 0 < args.validation_size < 1:
        parser.error("--validation-size must be between 0 and 1")

    if args.input:
        if args.test_input:
            parser.error("--test-input can only be used with --train-input")
        source = load_dataset(args.input)
        # Retain the original single-input smoke-test interface and its internal holdout.
        all_indices = np.arange(len(source))
        labels = source["label"].to_numpy()
        if args.group_column and args.group_column in source and source[args.group_column].nunique() > 2:
            grouped = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
            train_pool, test_indices = next(
                grouped.split(
                    all_indices,
                    labels,
                    groups=source[args.group_column].fillna("missing"),
                )
            )
            grouped_validation = GroupShuffleSplit(
                n_splits=1, test_size=0.1875, random_state=args.seed + 1
            )
            train_local, validation_local = next(
                grouped_validation.split(
                    np.arange(len(train_pool)),
                    labels[train_pool],
                    groups=source.iloc[train_pool][args.group_column].fillna("missing"),
                )
            )
            training_subset = source.iloc[train_pool[train_local]].reset_index(drop=True)
            validation_subset = source.iloc[train_pool[validation_local]].reset_index(drop=True)
            split_strategy = "source_grouped"
        else:
            train_pool, test_indices = train_test_split(
                all_indices,
                test_size=0.2,
                random_state=args.seed,
                stratify=labels,
            )
            train_local, validation_local = train_test_split(
                np.arange(len(train_pool)),
                test_size=0.1875,
                random_state=args.seed,
                stratify=labels[train_pool],
            )
            training_subset = source.iloc[train_pool[train_local]].reset_index(drop=True)
            validation_subset = source.iloc[train_pool[validation_local]].reset_index(drop=True)
            split_strategy = "stratified_random"

        test_df = source.iloc[test_indices].reset_index(drop=True)
        model, predictions, result = fit_with_explicit_validation(
            training_subset,
            validation_subset,
            test_df,
            seed=args.seed,
            split_name="single_input_holdout",
        )
        result["split_strategy"] = split_strategy
    else:
        if not args.test_input:
            parser.error("--test-input is required with --train-input")
        training_df = load_dataset(args.train_input)
        test_df = load_dataset(args.test_input)
        check_split_ids(training_df, test_df)
        model, predictions, result = fit_and_score(
            training_df,
            test_df,
            seed=args.seed,
            validation_size=args.validation_size,
            split_name="official_test",
        )

    output_dir = Path(args.output_dir)
    save_outputs(model, predictions, result, output_dir)
    print(json.dumps(result, indent=2))
    print(f"Saved model, predictions, metrics, and threshold to {output_dir}")


def fit_with_explicit_validation(
    training_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    split_name: str,
) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    model = build_model(seed)
    training_text = make_text(training_df)
    start = time.perf_counter()
    model.fit(training_text, training_df["label"].to_numpy())
    training_seconds = time.perf_counter() - start
    validation_probabilities = model.predict_proba(make_text(validation_df))[:, 1]
    threshold = select_threshold(
        validation_df["label"].to_numpy(), validation_probabilities
    )
    test_text = make_text(test_df)
    start = time.perf_counter()
    probabilities = model.predict_proba(test_text)[:, 1]
    latency_ms = (time.perf_counter() - start) / max(len(test_df), 1) * 1000
    predictions = (probabilities >= threshold).astype(int)
    id_column = "row_id" if "row_id" in test_df else "id" if "id" in test_df else None
    columns: dict[str, Any] = {
        "row_id": test_df[id_column].to_numpy() if id_column else test_df.index.to_numpy(),
        "y_true": test_df["label"].to_numpy(),
        "probability": probabilities,
        "prediction": predictions,
        "answer": test_df["answer"].to_numpy(),
    }
    for column in ("source_id", "id", "model", "task_type"):
        if column in test_df:
            columns[column] = test_df[column].to_numpy()
    predictions_df = pd.DataFrame(columns)
    result: dict[str, Any] = {
        "model": "TF-IDF word+char / LogisticRegression",
        "dataset": "RAGTruth" if split_name == "official_test" else "single_input",
        "test_split": split_name,
        "threshold_selected_on": "training_validation",
        "selected_threshold": threshold,
        "seed": seed,
        "n_train": len(training_df),
        "n_validation": len(validation_df),
        "n_test": len(test_df),
        "validation_fraction_of_training": len(validation_df)
        / (len(training_df) + len(validation_df)),
        "split_strategy": (
            "stratified_train_validation"
            if split_name == "official_test"
            else "single_input_holdout"
        ),
        "train_label_counts": training_df["label"].value_counts().sort_index().to_dict(),
        "validation_label_counts": validation_df["label"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "test_label_counts": test_df["label"].value_counts().sort_index().to_dict(),
        "training_time_seconds": training_seconds,
        "test_inference_latency_ms_per_sample": latency_ms,
        **metrics(test_df["label"].to_numpy(), probabilities, predictions),
    }
    return model, predictions_df, result


if __name__ == "__main__":
    main()
