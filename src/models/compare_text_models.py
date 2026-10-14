
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.models.tune_text_baseline import make_text


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/ragtruth/train.csv"
TUNED_PATH = ROOT / "artifacts/tuning/text_baseline_tuned.joblib"
OUTPUT_DIR = ROOT / "artifacts/comparison"


def build_original_model():
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=80_000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=40_000,
            sublinear_tf=True,
            min_df=1,
        )),
    ])

    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def score_model(model, x_val, y_val):
    probabilities = model.predict_proba(x_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(precision_score(
            y_val, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_val, predictions, zero_division=0
        )),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_val, probabilities)),
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
        "brier_score": float(brier_score_loss(y_val, probabilities)),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    y = pd.to_numeric(df["label"], errors="raise").astype(int)
    x = make_text(df)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Original baseline: fit with the original feature configuration.
    baseline = build_original_model()
    print("Training original baseline...")
    baseline.fit(x_train, y_train)

    # Tuned estimator: already fitted by the tuning script.
    print("Loading tuned model...")
    tuned = joblib.load(TUNED_PATH)

    results = {
        "original_baseline": score_model(
            baseline, x_val, y_val.to_numpy()
        ),
        "tuned_model": score_model(
            tuned, x_val, y_val.to_numpy()
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "validation_comparison.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Validation comparison (threshold = 0.5) ===")
    print(pd.DataFrame(results).T.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
