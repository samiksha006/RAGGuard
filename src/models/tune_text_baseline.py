
"""Memory-efficient tuning for word + character TF-IDF Logistic Regression."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform

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
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


RANDOM_STATE = 42


def make_text(df: pd.DataFrame) -> pd.Series:
    """Combine available question, context, and answer columns."""
    question = (
        df["question"]
        if "question" in df.columns
        else pd.Series("", index=df.index)
    )
    context = (
        df["context"]
        if "context" in df.columns
        else pd.Series("", index=df.index)
    )
    answer = (
        df["answer"]
        if "answer" in df.columns
        else pd.Series("", index=df.index)
    )

    return (
        question.fillna("").astype(str)
        + " [CONTEXT] "
        + context.fillna("").astype(str)
        + " [ANSWER] "
        + answer.fillna("").astype(str)
    )


def build_pipeline(seed: int = RANDOM_STATE) -> Pipeline:
    """Build a memory-conscious word + character TF-IDF pipeline."""
    features = FeatureUnion(
        transformer_list=[
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=30_000,
                    min_df=2,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=15_000,
                    min_df=2,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ],
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def calculate_metrics(y_true, predictions, probabilities) -> dict:
    """Calculate classification and probability metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )
    else:
        metrics["roc_auc"] = None

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tune word-level and character-level TF-IDF "
            "with Logistic Regression."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/tuning/text_search.csv",
        help="CSV path for hyperparameter search history.",
    )
    parser.add_argument("--iterations", type=int, default=12)
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    # Load and validate the dataset.
    df = pd.read_csv(input_path)

    if "label" not in df.columns:
        parser.error("Input CSV must contain a 'label' column.")

    if not {"question", "context", "answer"}.intersection(df.columns):
        parser.error(
            "Input CSV must contain at least one of: "
            "'question', 'context', or 'answer'."
        )

    df = df.dropna(subset=["label"]).copy()

    try:
        numeric_labels = pd.to_numeric(df["label"], errors="raise")
    except (ValueError, TypeError):
        parser.error("The 'label' column must contain numeric 0/1 labels.")

    if not numeric_labels.isin([0, 1]).all():
        parser.error("The 'label' column must contain only 0 and 1.")

    y = numeric_labels.astype(int).reset_index(drop=True)
    texts = make_text(df).reset_index(drop=True)

    if y.nunique() != 2:
        parser.error("Both label classes (0 and 1) must be present.")

    class_counts = y.value_counts()
    if class_counts.min() < 4:
        parser.error(
            "Each class needs at least 4 samples for the "
            "stratified split and 3-fold cross-validation."
        )

    # Reserve a stratified internal validation set.
    x_train, x_val, y_train, y_val = train_test_split(
        texts,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Limit feature sizes to reduce RAM consumption.
    pipeline = build_pipeline()

    search_space = {
        "features__word__ngram_range": [(1, 1), (1, 2)],
        "features__word__max_features": [20_000, 30_000],
        "features__word__sublinear_tf": [True],
        "features__char__ngram_range": [(3, 5), (2, 5)],
        "features__char__max_features": [10_000, 15_000],
        "features__char__sublinear_tf": [True],
        "clf__C": loguniform(1e-2, 1e2),
    }

    # Run one fit at a time to avoid parallel worker memory spikes.
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        n_iter=args.iterations,
        scoring="f1",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
        pre_dispatch=1,
        refit=True,
        return_train_score=False,
        error_score="raise",
        verbose=1,
    )

    print(f"Dataset: {input_path}")
    print(f"Total rows: {len(df)}")
    print(f"Training rows: {len(x_train)}")
    print(f"Validation rows: {len(x_val)}")
    print(f"Search iterations: {args.iterations}")
    print("Starting memory-conscious hyperparameter search...")

    start_time = time.perf_counter()
    search.fit(x_train, y_train)
    elapsed_seconds = time.perf_counter() - start_time

    # Evaluate the selected model on the internal validation split.
    # The validation data is not used in CV hyperparameter selection.
    best_model = search.best_estimator_
    predictions = best_model.predict(x_val)
    probabilities = best_model.predict_proba(x_val)[:, 1]

    validation_metrics = calculate_metrics(
        y_val, predictions, probabilities
    )

    # Preserve the existing --output behavior.
    output_csv = Path(args.output)
    output_dir = output_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "text_baseline_tuned.joblib"
    metadata_path = output_dir / "experiment_metadata.json"
    metrics_path = output_dir / "validation_metrics.json"

    # Save full CV results, including each tested parameter combination.
    search_history = pd.DataFrame(search.cv_results_)
    search_history.to_csv(output_csv, index=False)

    # Save the fitted best estimator.
    joblib.dump(best_model, model_path, compress=3)

    metadata = {
        "experiment_name": "word_char_tfidf_logistic_regression",
        "input_path": str(input_path),
        "random_state": RANDOM_STATE,
        "iterations_requested": args.iterations,
        "iterations_completed": len(search.cv_results_),
        "cross_validation_folds": 3,
        "selection_metric": "f1",
        "training_rows": int(len(x_train)),
        "validation_rows": int(len(x_val)),
        "training_class_counts": {
            str(k): int(v)
            for k, v in y_train.value_counts().sort_index().items()
        },
        "validation_class_counts": {
            str(k): int(v)
            for k, v in y_val.value_counts().sort_index().items()
        },
        "best_cv_f1": float(search.best_score_),
        "best_parameters": search.best_params_,
        "validation_metrics": validation_metrics,
        "runtime_seconds": round(elapsed_seconds, 3),
        "feature_configuration": {
            "word_max_features": [20_000, 30_000],
            "character_max_features": [10_000, 15_000],
            "word_ngrams": ["(1, 1)", "(1, 2)"],
            "character_ngrams": ["(3, 5)", "(2, 5)"],
            "min_df": 2,
            "dtype": "float32",
        },
        "artifacts": {
            "model": str(model_path),
            "search_history": str(output_csv),
            "metadata": str(metadata_path),
            "validation_metrics": str(metrics_path),
        },
        "notes": [
            "Word and character TF-IDF are fitted inside each CV fold.",
            "Hyperparameters are selected using training-fold CV F1.",
            "Internal validation metrics are reported separately.",
            "The official test set is not loaded by this script.",
            "No classification threshold is selected by this script.",
            "The saved estimator is refitted on the internal training split.",
        ],
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(validation_metrics, indent=2),
        encoding="utf-8",
    )

    print("\n========== TUNING COMPLETE ==========")
    print(f"Best CV F1: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Validation metrics: {validation_metrics}")
    print(f"Runtime: {elapsed_seconds:.2f} seconds")
    print(f"\nSaved model: {model_path}")
    print(f"Saved search history: {output_csv}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved validation metrics: {metrics_path}")


if __name__ == "__main__":
    main()
