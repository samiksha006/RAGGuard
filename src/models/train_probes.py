"""Train layer-wise classical probes without fitting on official-test data."""

import argparse

import json

import time

from pathlib import Path

from typing import Any

import joblib

import matplotlib

import numpy as np

import pandas as pd

import yaml

from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import RandomForestClassifier

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

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

matplotlib.use("Agg")

import matplotlib.pyplot as plt

METRIC_NAMES = (

    "accuracy",

    "precision",

    "recall",

    "f1",

    "pr_auc",

    "roc_auc",

    "brier_score",

)

def load_config(path: str | Path) -> dict[str, Any]:

    with Path(path).open(encoding="utf-8") as file:

        return yaml.safe_load(file) or {}

def load_feature_split(
    feature_path: str | Path,
    data_path: str | Path,
    expected_split: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Load cached features and require exact ID, label, and row-order agreement."""
    started = time.perf_counter()
    print(f"[load:{expected_split}] Reading rows from {data_path}", flush=True)
    data = pd.read_csv(data_path).reset_index(drop=True)

    if "label" not in data:

        raise ValueError(f"{data_path}: label column is required")

    id_column = "row_id" if "row_id" in data else "id" if "id" in data else None

    if id_column is None:

        raise ValueError(f"{data_path}: row_id or id column is required")

    if data[id_column].isna().any() or data[id_column].astype(str).duplicated().any():

        raise ValueError(f"{data_path}: IDs must be present and unique")

    labels = pd.to_numeric(data["label"], errors="raise")

    if labels.isna().any() or not labels.isin([0, 1]).all():

        raise ValueError(f"{data_path}: labels must be binary 0/1")

    data["label"] = labels.astype(int)

    metadata_path = Path(feature_path).with_suffix(".metadata.json")

    metadata: dict[str, Any] = {}

    if metadata_path.exists():

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        observed_split = metadata.get("split")

        if observed_split != expected_split:

            raise ValueError(

                f"{feature_path}: expected {expected_split!r} features, "

                f"metadata identifies {observed_split!r}"

            )

    print(
        f"[load:{expected_split}] Reading cached features from {feature_path}",
        flush=True,
    )
    with np.load(feature_path, allow_pickle=False) as archive:
        keys = [key for key in archive.files if "_layer_" in key or key.startswith("layer_")]
        if not keys:
            raise ValueError(f"{feature_path}: no layer feature arrays found")
        features = {
            key: archive[key].astype("float32", copy=False)
            for key in keys
        }

        if "row_id" not in archive or "labels" not in archive:

            raise ValueError(f"{feature_path}: cache must include row_id and labels arrays")

        cached_ids = archive["row_id"].astype(str)

        cached_labels = archive["labels"].astype(int)

    data_ids = data[id_column].astype(str).to_numpy()

    if len(np.unique(cached_ids)) != len(cached_ids):

        raise ValueError(f"{feature_path}: cached row IDs must be unique")

    if metadata.get("partial_split_cache"):

        positions = pd.Series(np.arange(len(data)), index=data_ids)

        missing_ids = set(cached_ids) - set(data_ids)

        if missing_ids:

            raise ValueError(

                f"{feature_path}: {len(missing_ids)} cached IDs are not present in {data_path}"

            )

        data = data.iloc[positions.loc[cached_ids].to_numpy()].reset_index(drop=True)

        data_ids = data["row_id"].astype(str).to_numpy()

    if len(cached_ids) != len(data) or not np.array_equal(cached_ids, data_ids):

        raise ValueError(

            f"{feature_path}: cached row IDs/order do not exactly match {data_path}"

        )

    if len(cached_labels) != len(data) or not np.array_equal(

        cached_labels, data["label"].to_numpy()

    ):

        raise ValueError(f"{feature_path}: cached labels do not match {data_path}")

    for key, matrix in features.items():

        if matrix.ndim != 2 or matrix.shape[0] != len(data):

            raise ValueError(

                f"{feature_path}: {key} must be a 2-D matrix aligned to all rows"

            )

        if not np.isfinite(matrix).all():

            raise ValueError(f"{feature_path}: {key} contains non-finite features")

    data["row_id"] = data_ids
    feature_bytes = sum(matrix.nbytes for matrix in features.values())
    print(
        f"[load:{expected_split}] Loaded {len(data):,} aligned rows and "
        f"{len(features)} feature arrays ({feature_bytes / 1024**2:.1f} MiB) "
        f"in {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return features, data

def check_disjoint_ids(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:

    overlap = set(train_df["row_id"]) & set(test_df["row_id"])

    if overlap:

        raise ValueError(

            f"Training and official test splits share {len(overlap)} row IDs"

        )

def split_training_indices(

    labels: np.ndarray, validation_size: float, seed: int

) -> tuple[np.ndarray, np.ndarray]:

    indices = np.arange(len(labels))

    return train_test_split(

        indices,

        test_size=validation_size,

        random_state=seed,

        stratify=labels,

    )

def resolve_layers(requested_layers: list[int], total_layers: int) -> list[int]:

    resolved: list[int] = []

    for requested in requested_layers:

        index = requested if requested >= 0 else total_layers + requested

        if not 0 <= index < total_layers:

            raise ValueError(

                f"Configured layer {requested} is unavailable; model has "

                f"{total_layers} hidden-state outputs (valid indices 0..{total_layers - 1})."

            )

        if index not in resolved:

            resolved.append(index)

    return resolved

def build_classifier(name: str, seed: int) -> Any:

    if name == "logistic_regression":

        return make_pipeline(

            StandardScaler(),

            LogisticRegression(

                solver="lbfgs",

                max_iter=1000,

                class_weight="balanced",

                random_state=seed,

            ),

        )

    if name == "linear_svm":

        calibrated = CalibratedClassifierCV(

            LinearSVC(class_weight="balanced", random_state=seed),

            cv=3,

        )

        return make_pipeline(StandardScaler(), calibrated)

    if name == "random_forest":

        return RandomForestClassifier(

            n_estimators=300,

            min_samples_leaf=2,

            class_weight="balanced_subsample",

            n_jobs=-1,

            random_state=seed,

        )

    raise ValueError(f"Unsupported probe classifier: {name}")

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

def calculate_metrics(

    labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray

) -> dict[str, float | None]:

    return {

        "accuracy": float(accuracy_score(labels, predictions)),

        "precision": float(precision_score(labels, predictions, zero_division=0)),

        "recall": float(recall_score(labels, predictions, zero_division=0)),

        "f1": float(f1_score(labels, predictions, zero_division=0)),

        "pr_auc": float(average_precision_score(labels, probabilities)),

        "roc_auc": (

            float(roc_auc_score(labels, probabilities))

            if len(np.unique(labels)) == 2

            else None

        ),

        "brier_score": float(brier_score_loss(labels, probabilities)),

    }

def train_and_compare(
    train_features: dict[str, np.ndarray],
    train_df: pd.DataFrame,
    test_features: dict[str, np.ndarray],
    test_df: pd.DataFrame,
    *,
    requested_layers: list[int],
    classifiers: list[str],
    validation_size: float,
    seed: int,
    evaluation_split: str = "official_test",
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame, Any]:
    """Compare answer-only and combined-segment probes using training validation."""
    run_started = time.perf_counter()
    print("[probes] Checking split IDs and creating the stratified validation split", flush=True)
    check_disjoint_ids(train_df, test_df)

    labels = train_df["label"].to_numpy(dtype=int)
    train_indices, validation_indices = split_training_indices(
        labels, validation_size, seed
    )

    if len(np.unique(labels[train_indices])) != 2:
        raise ValueError("Training partition must contain both classes")

    layer_indices = [
        int(key.rsplit("_", 1)[-1])
        for key in train_features
        if "_layer_" in key and key.rsplit("_", 1)[-1].isdigit()
    ]
    if not layer_indices:
        raise ValueError("Training cache has no layer-indexed feature arrays")
    total_layers = max(layer_indices) + 1
    layers = resolve_layers(requested_layers, total_layers)

    configurations = {
        "answer_only": ("answer",),
        "question_answer": ("question", "answer"),
        "context_answer": ("context", "answer"),
        "question_context_answer": ("question", "context", "answer"),
    }

    train_segments = {
        segment
        for segment in ("question", "context", "answer")
        if any(
            key.startswith(f"{segment}_layer_")
            for key in train_features
        )
    }
    test_segments = {
        segment
        for segment in ("question", "context", "answer")
        if any(
            key.startswith(f"{segment}_layer_")
            for key in test_features
        )
    }
    candidates: list[dict[str, Any]] = []
    skipped_configurations: list[str] = []
    expected_candidates = sum(
        all(segment in train_segments for segment in segments)
        for segments in configurations.values()
    ) * len(layers) * len(classifiers)
    print(
        f"[probes] Evaluating {len(configurations)} feature configurations "
        f"across layers {layers} with classifiers {classifiers}; "
        f"up to {expected_candidates} candidates",
        flush=True,
    )

    for layer in layers:
        print(f"[probes] Layer {layer}: preparing feature configurations", flush=True)
        for config_name, segments in configurations.items():
            feature_names = [f"{segment}_layer_{layer}" for segment in segments]

            missing_train_segments = set(segments) - train_segments
            missing_test_segments = set(segments) - test_segments
            if (
                missing_train_segments
                and missing_train_segments == missing_test_segments
            ):
                skipped_configurations.append(config_name)
                print(
                    f"[probes] Skipping {config_name}: feature segments are absent "
                    "from both caches",
                    flush=True,
                )
                continue
            if missing_train_segments:
                raise ValueError(
                    f"Training cache is missing feature segment(s) "
                    f"{sorted(missing_train_segments)} required by {config_name}"
                )
            if missing_test_segments:
                raise ValueError(
                    f"Official test cache is missing feature segment(s) "
                    f"{sorted(missing_test_segments)} required by {config_name}"
                )
            missing_train = [name for name in feature_names if name not in train_features]
            if missing_train:
                raise ValueError(
                    f"Training cache is missing feature key(s) for {config_name}, "
                    f"layer {layer}: {', '.join(missing_train)}"
                )
            if not all(name in test_features for name in feature_names):
                raise ValueError(
                    f"Official test cache is missing features for {config_name}, "
                    f"layer {layer}"
                )

            for name in feature_names:
                if train_features[name].shape[1] != test_features[name].shape[1]:
                    raise ValueError(f"Train/test dimensions differ for {name}")
                if len(train_features[name]) != len(train_df):
                    raise ValueError(f"Training rows do not align for {name}")
                if len(test_features[name]) != len(test_df):
                    raise ValueError(f"Test rows do not align for {name}")

            configuration_started = time.perf_counter()
            train_matrix = np.concatenate(
                [train_features[name] for name in feature_names], axis=1
            )
            fit_matrix = train_matrix[train_indices]
            validation_matrix = train_matrix[validation_indices]
            print(
                f"[probes] Layer {layer}, {config_name}: "
                f"{train_matrix.shape[1]:,} features; prepared training/validation "
                f"matrices in {time.perf_counter() - configuration_started:.2f}s",
                flush=True,
            )

            for classifier_name in classifiers:
                print(
                    f"[probes] Layer {layer}, {config_name}, {classifier_name}: "
                    f"fitting on {len(train_indices):,} training rows",
                    flush=True,
                )
                classifier = build_classifier(classifier_name, seed)
                started = time.perf_counter()
                classifier.fit(fit_matrix, labels[train_indices])
                training_seconds = time.perf_counter() - started

                validation_started = time.perf_counter()
                validation_probabilities = classifier.predict_proba(
                    validation_matrix
                )[:, 1]
                threshold = select_threshold(
                    labels[validation_indices], validation_probabilities
                )
                validation_predictions = (
                    validation_probabilities >= threshold
                ).astype(int)

                candidate = {
                    "feature_name": config_name,
                    "feature_names": feature_names,
                    "feature_segment": config_name,
                    "layer": layer,
                    "classifier_name": classifier_name,
                    "classifier": classifier,
                    "threshold": threshold,
                    "validation_f1": float(f1_score(
                        labels[validation_indices],
                        validation_predictions,
                        zero_division=0,
                    )),
                    "training_seconds": training_seconds,
                    "validation_seconds": time.perf_counter() - validation_started,
                    "n_train": len(train_indices),
                    "n_validation": len(validation_indices),
                }
                candidates.append(candidate)
                print(
                    f"[probes] Layer {layer}, {config_name}, {classifier_name}: "
                    f"fit {training_seconds:.2f}s; validation "
                    f"{candidate['validation_seconds']:.2f}s; "
                    f"F1={candidate['validation_f1']:.4f}, "
                    f"threshold={threshold:.3f}",
                    flush=True,
                )

            del train_matrix, fit_matrix, validation_matrix

    if not candidates:
        raise ValueError("No feature layer and classifier combinations were selected")

    selected = max(candidates, key=lambda item: item["validation_f1"])
    print(
        f"[probes] Validation selection complete: {selected['feature_segment']} "
        f"layer {selected['layer']} / {selected['classifier_name']} "
        f"(F1={selected['validation_f1']:.4f}, "
        f"threshold={selected['threshold']:.3f}); "
        "beginning held-out test evaluation",
        flush=True,
    )
    comparison: list[dict[str, Any]] = []
    selected_predictions = None
    test_labels = test_df["label"].to_numpy(dtype=int)

    candidates_by_features: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for candidate in candidates:
        candidates_by_features.setdefault(
            tuple(candidate["feature_names"]), []
        ).append(candidate)

    for feature_names, configuration_candidates in candidates_by_features.items():
        test_matrix = np.concatenate(
            [test_features[name] for name in feature_names], axis=1
        )
        for candidate in configuration_candidates:
            evaluation_started = time.perf_counter()
            probabilities = candidate["classifier"].predict_proba(test_matrix)[:, 1]
            evaluation_seconds = time.perf_counter() - evaluation_started
            latency_ms = evaluation_seconds / max(len(test_df), 1) * 1000
            predictions = (probabilities >= candidate["threshold"]).astype(int)

            row = {
                key: candidate[key]
                for key in (
                    "feature_segment", "layer", "classifier_name",
                    "threshold", "validation_f1", "training_seconds",
                    "validation_seconds", "n_train", "n_validation",
                )
            }
            row["feature_names"] = ",".join(candidate["feature_names"])
            row.update(calculate_metrics(test_labels, probabilities, predictions))
            row["evaluation_seconds"] = evaluation_seconds
            row["inference_latency_ms_per_sample"] = latency_ms
            row["split"] = evaluation_split
            row["selected_by_validation"] = candidate is selected
            comparison.append(row)
            print(
                f"[probes] Test evaluation {candidate['feature_segment']} layer "
                f"{candidate['layer']} / {candidate['classifier_name']}: "
                f"{evaluation_seconds:.2f}s",
                flush=True,
            )

            if candidate is selected:
                selected_predictions = pd.DataFrame({
                    "row_id": test_df["row_id"].to_numpy(),
                    "y_true": test_labels,
                    "probability": probabilities,
                    "prediction": predictions,
                    "feature_segment": candidate["feature_segment"],
                    "feature_names": ",".join(candidate["feature_names"]),
                    "layer": candidate["layer"],
                    "classifier": candidate["classifier_name"],
                })
        del test_matrix

    assert selected_predictions is not None

    metadata = {
        "evaluation_split": evaluation_split,
        "model_selection": "highest validation F1 across feature configuration, layer and classifier",
        "threshold_selection": "validation F1 on the training split only",
        "feature_scaling": "StandardScaler fit within each model pipeline on training partition only",
        "representation_interpretation": (
            "Observer-model hidden-state representations; not necessarily "
            "activations from the original answer-generating model."
        ),
        "pca": "not used",
        "test_labels_used_for_fit_or_selection": False,
        "test_features_used_for_fit_or_selection": False,
        "n_train_rows": len(train_df),
        "n_train_partition": len(train_indices),
        "n_validation": len(validation_indices),
        "n_official_test": len(test_df),
        "split_strategy": (
            "Stratified training/validation split using "
            "train_test_split(random_state=seed); official test excluded from "
            "all model and threshold selection."
        ),
        "train_row_ids": train_df["row_id"].tolist(),
        "fit_train_row_ids": train_df.iloc[train_indices]["row_id"].tolist(),
        "validation_row_ids": train_df.iloc[validation_indices]["row_id"].tolist(),
        "official_test_row_ids": test_df["row_id"].tolist(),
        "test_data_scope": "full_split" if evaluation_split == "official_test" else "internal_holdout",
        "selected_probe": {
            "feature_segment": selected["feature_segment"],
            "feature_names": selected["feature_names"],
            "layer": selected["layer"],
            "classifier": selected["classifier_name"],
            "threshold": selected["threshold"],
            "validation_f1": selected["validation_f1"],
        },
        "validation_fraction": validation_size,
        "seed": seed,
        "requested_layers": requested_layers,
        "resolved_layers": layers,
        "requested_configurations": list(configurations),
        "skipped_configurations": sorted(set(skipped_configurations)),
        "candidate_count": len(candidates),
        "total_training_and_validation_seconds": sum(
            candidate["training_seconds"] + candidate["validation_seconds"]
            for candidate in candidates
        ),
        "total_test_evaluation_seconds": sum(
            row["evaluation_seconds"] for row in comparison
        ),
    }

    print(
        f"[probes] Completed {len(candidates)} candidates in "
        f"{time.perf_counter() - run_started:.2f}s",
        flush=True,
    )
    return comparison, metadata, selected_predictions, selected["classifier"]

def load_baseline_metrics(path: str | Path | None) -> dict[str, Any] | None:

    if not path:

        return None

    baseline_path = Path(path)

    if not baseline_path.is_file():

        raise FileNotFoundError(f"Text-baseline metrics not found: {baseline_path}")

    result = json.loads(baseline_path.read_text(encoding="utf-8"))

    return {

        "feature_segment": "text_baseline",

        "layer": None,

        "classifier_name": "tfidf_logistic_regression",

        "threshold": result.get("selected_threshold"),

        "validation_f1": None,

        "training_seconds": result.get("training_time_seconds"),

        "inference_latency_ms_per_sample": result.get(

            "test_inference_latency_ms_per_sample"

        ),

        "split": result.get("test_split", "official_test"),

        "selected_by_validation": False,

        **{metric: result.get(metric) for metric in METRIC_NAMES},

        "source_metrics_file": str(baseline_path),

    }

def write_comparison_plots(

    results: pd.DataFrame, output_dir: Path, split_name: str = "official_test"

) -> None:

    plotting_started = time.perf_counter()
    probe_rows = results[results["feature_segment"] != "text_baseline"]

    baseline_rows = results[results["feature_segment"] == "text_baseline"]

    plot_metrics = (

        "f1",

        "precision",

        "recall",

        "pr_auc",

        "roc_auc",

        "brier_score",

        "inference_latency_ms_per_sample",

    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in plot_metrics:

        figure, axis = plt.subplots(figsize=(8, 5))

        for (segment, classifier), rows in probe_rows.groupby(

            ["feature_segment", "classifier_name"]

        ):

            ordered = rows.sort_values("layer")

            axis.plot(

                ordered["layer"],

                ordered[metric],

                marker="o",

                label=f"{segment} / {classifier}",

            )

        if not baseline_rows.empty and baseline_rows[metric].notna().any():

            axis.axhline(

                float(baseline_rows[metric].dropna().iloc[0]),

                linestyle="--",

                color="black",

                label="TF-IDF text baseline",

            )

        axis.set(

            xlabel="Hidden-state output index",

            ylabel=metric.replace("_", " "),

            title=f"{split_name.replace('_', ' ').title()} layer comparison: "

            f"{metric.replace('_', ' ')}",

        )

        axis.legend(fontsize="small", ncol=1)

        figure.tight_layout()

        figure.savefig(

            output_dir / f"layer_comparison_{metric}_{split_name}.png",

            dpi=160,

        )

        plt.close(figure)
    print(
        f"[output] Wrote comparison plots for {split_name} in "
        f"{time.perf_counter() - plotting_started:.2f}s",
        flush=True,
    )

def run_legacy(

    features_path: str | Path,

    data_path: str | Path,

    output_dir: str | Path,

    *,

    requested_layers: list[int],

    classifiers: list[str],

    validation_size: float,

    seed: int,

) -> dict[str, Any]:

    """Keep the earlier one-file holdout workflow available for smoke experiments."""

    data = pd.read_csv(data_path).reset_index(drop=True)

    full_dataset_rows = len(data)

    if "label" not in data:

        raise ValueError(f"{data_path}: label column is required")

    with np.load(features_path, allow_pickle=False) as archive:

        if "row_index" not in archive:

            raise ValueError(f"{features_path}: legacy feature cache needs row_index")

        row_index = archive["row_index"].astype(int)

        feature_arrays = {

            (

                f"full_sequence_{key}"

                if key.startswith("layer_")

                else key

            ): archive[key].astype("float32")

            for key in archive.files

            if key.startswith("layer_") or "_layer_" in key

        }

        cached_ids = archive["row_id"].astype(str) if "row_id" in archive else None

        cached_labels = archive["labels"].astype(int) if "labels" in archive else None

    if not feature_arrays:

        raise ValueError(f"{features_path}: no legacy layer_N arrays found")

    if row_index.ndim != 1 or len(np.unique(row_index)) != len(row_index):

        raise ValueError(f"{features_path}: row_index must be unique and one-dimensional")

    if (row_index < 0).any() or (row_index >= len(data)).any():

        raise ValueError(f"{features_path}: row_index contains an invalid data row")

    data = data.iloc[row_index].reset_index(drop=True)

    labels = pd.to_numeric(data["label"], errors="raise")

    if labels.isna().any() or not labels.isin([0, 1]).all():

        raise ValueError(f"{data_path}: labels must be binary 0/1")

    data["label"] = labels.astype(int)

    id_column = "row_id" if "row_id" in data else "id" if "id" in data else None

    if id_column is not None:

        row_ids = data[id_column].astype(str).to_numpy()

    else:

        row_ids = np.asarray([f"legacy-row-{index}" for index in row_index])

    if len(np.unique(row_ids)) != len(row_ids):

        row_ids = np.asarray([f"legacy-row-{index}" for index in row_index])

    data["row_id"] = row_ids

    if cached_ids is not None and not np.array_equal(cached_ids, row_ids):

        raise ValueError(f"{features_path}: row IDs do not align with {data_path}")

    if cached_labels is not None and not np.array_equal(

        cached_labels, data["label"].to_numpy()

    ):

        raise ValueError(f"{features_path}: labels do not align with {data_path}")

    for key, matrix in feature_arrays.items():

        if matrix.ndim != 2 or matrix.shape[0] != len(data):

            raise ValueError(f"{features_path}: {key} is not aligned with data rows")

    labels_array = data["label"].to_numpy()

    train_indices, test_indices = train_test_split(

        np.arange(len(data)),

        test_size=0.2,

        random_state=seed,

        stratify=labels_array,

    )

    train_features = {

        key: matrix[train_indices]

        for key, matrix in feature_arrays.items()

    }

    test_features = {

        key: matrix[test_indices]

        for key, matrix in feature_arrays.items()

    }

    train_df = data.iloc[train_indices].reset_index(drop=True)

    test_df = data.iloc[test_indices].reset_index(drop=True)

    comparison, metadata, predictions, selected_model = train_and_compare(

        train_features,

        train_df,

        test_features,

        test_df,

        requested_layers=requested_layers,

        classifiers=classifiers,

        validation_size=validation_size,

        seed=seed,

        evaluation_split="single_input_holdout",

    )

    metadata["feature_cache_directory"] = str(Path(features_path).parent)

    metadata["full_dataset_rows"] = full_dataset_rows

    metadata["baseline_metrics"] = None

    output = Path(output_dir)

    model_dir = output / "models"

    model_dir.mkdir(parents=True, exist_ok=True)

    selected = metadata["selected_probe"]

    model_path = model_dir / (
        f"{selected['feature_names'][0]}_"
        f"{selected['classifier']}.joblib"
    )

    joblib.dump(selected_model, model_path)

    metadata["selected_model_file"] = str(model_path)

    result_frame = pd.DataFrame(comparison)

    result_frame.to_csv(output / "comparison_single_input_holdout.csv", index=False)

    (output / "comparison_single_input_holdout.json").write_text(

        json.dumps({"metadata": metadata, "results": comparison}, indent=2),

        encoding="utf-8",

    )

    predictions.to_csv(

        output / "selected_probe_predictions_single_input_holdout.csv", index=False

    )

    pd.DataFrame(

        {

            "row_id": train_df["row_id"].tolist() + test_df["row_id"].tolist(),

            "split": ["train"] * len(train_df) + ["internal_holdout"] * len(test_df),

            "is_validation": [

                row_id in set(metadata["validation_row_ids"])

                for row_id in train_df["row_id"]

            ]

            + [False] * len(test_df),

        }

    ).to_csv(output / "split_membership_single_input_holdout.csv", index=False)

    write_comparison_plots(

        result_frame, output, split_name="single_input_holdout"

    )

    return metadata

def run_paired(

    train_features_path: str | Path,

    test_features_path: str | Path,

    train_data_path: str | Path,

    test_data_path: str | Path,

    output_dir: str | Path,

    *,

    requested_layers: list[int],

    classifiers: list[str],

    validation_size: float,

    seed: int,

    baseline_metrics_path: str | Path | None,

) -> dict[str, Any]:

    train_features, train_df = load_feature_split(

        train_features_path, train_data_path, "train"

    )

    test_features, test_df = load_feature_split(

        test_features_path, test_data_path, "test"

    )

    check_disjoint_ids(train_df, test_df)

    comparison, metadata, selected_predictions, selected_model = train_and_compare(

        train_features,

        train_df,

        test_features,

        test_df,

        requested_layers=requested_layers,

        classifiers=classifiers,

        validation_size=validation_size,

        seed=seed,

    )

    output = Path(output_dir)

    model_dir = output / "models"

    model_dir.mkdir(parents=True, exist_ok=True)

    selected_info = metadata["selected_probe"]

    selected_key = selected_info["feature_names"][0]

    selected_model_path = model_dir / f"{selected_key}_{selected_info['classifier']}.joblib"
    output_started = time.perf_counter()
    joblib.dump(selected_model, selected_model_path)
    print(
        f"[output] Serialized selected model in "
        f"{time.perf_counter() - output_started:.2f}s",
        flush=True,
    )

    metadata["selected_model_file"] = str(selected_model_path)

    metadata["feature_cache_directory"] = str(Path(train_features_path).parent)

    test_feature_metadata_path = Path(test_features_path).with_suffix(

        ".metadata.json"

    )

    train_feature_metadata_path = Path(train_features_path).with_suffix(

        ".metadata.json"

    )

    if train_feature_metadata_path.exists():

        metadata["train_feature_metadata"] = json.loads(

            train_feature_metadata_path.read_text(encoding="utf-8")

        )

    if test_feature_metadata_path.exists():

        test_feature_metadata = json.loads(

            test_feature_metadata_path.read_text(encoding="utf-8")

        )

        metadata["test_feature_metadata"] = test_feature_metadata

        metadata["test_data_scope"] = (

            "smoke_subset"

            if test_feature_metadata.get("partial_split_cache")

            else "full_split"

        )

        metadata["test_source_row_count"] = test_feature_metadata.get(

            "source_row_count"

        )

    metadata["baseline_metrics_file"] = (

        str(baseline_metrics_path) if baseline_metrics_path else None

    )

    metadata["baseline_metrics"] = load_baseline_metrics(baseline_metrics_path)

    rows = list(comparison)

    baseline = metadata["baseline_metrics"]

    if baseline is not None:

        rows.append(baseline)

    result_frame = pd.DataFrame(rows)
    output_started = time.perf_counter()
    result_frame.to_csv(output / "comparison_official_test.csv", index=False)

    (output / "comparison_official_test.json").write_text(

        json.dumps(

            {"metadata": metadata, "results": comparison},

            indent=2,

            allow_nan=False,

        ),
        encoding="utf-8",
    )
    print(
        f"[output] Wrote comparison CSV/JSON in "
        f"{time.perf_counter() - output_started:.2f}s",
        flush=True,
    )

    output_started = time.perf_counter()
    selected_predictions.to_csv(
        output / "selected_probe_predictions_official_test.csv", index=False
    )

    membership = pd.DataFrame(

        {

            "row_id": metadata["train_row_ids"]

            + metadata["official_test_row_ids"],

            "source_split": ["official_train"] * len(metadata["train_row_ids"])

            + ["official_test"] * len(metadata["official_test_row_ids"]),

            "partition": [

                "validation"

                if row_id in set(metadata["validation_row_ids"])

                else "train"

                for row_id in metadata["train_row_ids"]

            ]

            + ["official_test"] * len(metadata["official_test_row_ids"]),

        }

    )

    membership.to_csv(output / "split_membership.csv", index=False)

    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"[output] Wrote predictions and split metadata in "
        f"{time.perf_counter() - output_started:.2f}s",
        flush=True,
    )

    write_comparison_plots(result_frame, output)

    print(json.dumps(metadata, indent=2))

    print(f"Official-test comparison saved under {output}")

    return metadata

def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--train-features")

    parser.add_argument("--test-features")

    parser.add_argument("--train-data")

    parser.add_argument("--test-data")

    parser.add_argument("--features", help="Legacy single-cache input for smoke tests.")

    parser.add_argument("--data", help="Legacy single CSV input for smoke tests.")

    parser.add_argument("--output-dir")

    parser.add_argument("--config", default="configs/experiment_matrix.yaml")

    parser.add_argument("--baseline-metrics")

    parser.add_argument(

        "--no-baseline",

        action="store_true",

        help="Do not add full-test text-baseline metrics (use for smoke subsets).",

    )

    parser.add_argument("--seed", type=int)

    parser.add_argument("--validation-size", type=float)

    parser.add_argument("--layers", type=int, nargs="+")

    parser.add_argument("--models", nargs="+")

    args = parser.parse_args()

    config = load_config(args.config)

    probe_config = config.get("probe_training", {})

    training_config = config.get("training", {})

    requested_layers = args.layers or probe_config.get("layers", [-1, -2, -4, -6])

    classifiers = args.models or probe_config.get(

        "models", ["logistic_regression", "linear_svm"]

    )

    validation_size = args.validation_size or float(

        probe_config.get("validation_size", 0.2)

    )

    seed = args.seed if args.seed is not None else int(

        probe_config.get("seed", training_config.get("random_state", 42))

    )

    output_dir = args.output_dir or probe_config.get(

        "output_dir", "artifacts/probes/ragtruth"

    )

    baseline_metrics = (

        None

        if args.no_baseline

        else args.baseline_metrics

        or probe_config.get(

            "baseline_metrics", "artifacts/text_baseline/metrics_official_test.json"

        )

    )

    if not 0 < validation_size < 1:

        parser.error("--validation-size must be between 0 and 1")

    if args.features or args.data:

        if not args.features or not args.data:

            parser.error("Legacy mode requires both --features and --data.")

        if any((args.train_features, args.test_features, args.train_data, args.test_data)):

            parser.error("Use either legacy or paired mode, not both.")

        metadata = run_legacy(

            args.features,

            args.data,

            output_dir,

            requested_layers=requested_layers,

            classifiers=classifiers,

            validation_size=validation_size,

            seed=seed,

        )

        print(json.dumps(metadata, indent=2))

        return

    required = {

        "--train-features": args.train_features,

        "--test-features": args.test_features,

        "--train-data": args.train_data,

        "--test-data": args.test_data,

    }

    missing = [name for name, value in required.items() if not value]

    if missing:

        parser.error(f"Paired mode requires: {', '.join(missing)}")

    run_paired(

        args.train_features,

        args.test_features,

        args.train_data,

        args.test_data,

        output_dir,

        requested_layers=requested_layers,

        classifiers=classifiers,

        validation_size=validation_size,

        seed=seed,

        baseline_metrics_path=baseline_metrics,

    )

if __name__ == "__main__":

    main()
