import json
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.extract_hidden_states import mean_pool_hidden_states
from src.models import train_probes


def test_mean_pooling_uses_every_attended_token():
    import torch

    hidden = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool_hidden_states(hidden, mask)
    assert pooled.tolist() == [[2.0]]


def test_feature_cache_must_match_csv_row_id_order_and_labels(tmp_path):
    data = pd.DataFrame(
        {
            "row_id": ["train-1", "train-2"],
            "question": ["q1", "q2"],
            "context": ["c1", "c2"],
            "answer": ["a1", "a2"],
            "label": [0, 1],
        }
    )
    data_path = tmp_path / "train.csv"
    data.to_csv(data_path, index=False)
    feature_path = tmp_path / "train.npz"
    np.savez_compressed(
        feature_path,
        row_id=np.asarray(["train-1", "train-2"]),
        labels=np.asarray([0, 1]),
        row_index=np.asarray([0, 1]),
        answer_layer_0=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype="float32"),
    )
    feature_path.with_suffix(".metadata.json").write_text(
        json.dumps({"split": "train"}), encoding="utf-8"
    )

    features, loaded_data = train_probes.load_feature_split(
        feature_path, data_path, "train"
    )
    assert features["answer_layer_0"].shape == (2, 2)
    assert loaded_data["row_id"].tolist() == ["train-1", "train-2"]

    np.savez_compressed(
        feature_path,
        row_id=np.asarray(["train-2", "train-1"]),
        labels=np.asarray([0, 1]),
        row_index=np.asarray([0, 1]),
        answer_layer_0=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype="float32"),
    )
    with pytest.raises(ValueError, match="row IDs/order"):
        train_probes.load_feature_split(feature_path, data_path, "train")


def test_feature_cache_rejects_split_metadata_mismatch(tmp_path):
    data = pd.DataFrame(
        {
            "row_id": ["train-1", "train-2"],
            "answer": ["a1", "a2"],
            "label": [0, 1],
        }
    )
    data_path = tmp_path / "train.csv"
    data.to_csv(data_path, index=False)
    feature_path = tmp_path / "train.npz"
    np.savez_compressed(
        feature_path,
        row_id=np.asarray(["train-1", "train-2"]),
        labels=np.asarray([0, 1]),
        answer_layer_0=np.asarray([[0.1], [0.2]], dtype="float32"),
    )
    feature_path.with_suffix(".metadata.json").write_text(
        json.dumps({"split": "test"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="expected 'train'"):
        train_probes.load_feature_split(feature_path, data_path, "train")


def test_partial_smoke_cache_aligns_to_ids_in_source_csv(tmp_path):
    data = pd.DataFrame(
        {
            "row_id": ["train-1", "train-2", "train-3"],
            "answer": ["a1", "a2", "a3"],
            "label": [0, 1, 0],
        }
    )
    data_path = tmp_path / "train.csv"
    data.to_csv(data_path, index=False)
    feature_path = tmp_path / "train.npz"
    np.savez_compressed(
        feature_path,
        row_id=np.asarray(["train-1", "train-2"]),
        labels=np.asarray([0, 1]),
        row_index=np.asarray([0, 1]),
        answer_layer_0=np.asarray([[0.1], [0.2]], dtype="float32"),
    )
    feature_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "split": "train",
                "partial_split_cache": True,
                "source_row_count": 3,
            }
        ),
        encoding="utf-8",
    )
    features, loaded_data = train_probes.load_feature_split(
        feature_path, data_path, "train"
    )
    assert len(features["answer_layer_0"]) == 2
    assert loaded_data["row_id"].tolist() == ["train-1", "train-2"]
    assert loaded_data["label"].tolist() == [0, 1]


class RecordingClassifier:
    def __init__(self):
        self.fit_features = None
        self.probability_inputs = []

    def fit(self, features, labels):
        self.fit_features = np.asarray(features).copy()
        assert len(self.fit_features) == len(labels)
        return self

    def predict_proba(self, features):
        matrix = np.asarray(features)
        self.probability_inputs.append(matrix.copy())
        probability = np.where(matrix[:, 0] > 50, 0.8, 0.2)
        return np.column_stack((1 - probability, probability))


def test_probes_fit_and_select_only_from_training_then_score_test(monkeypatch):
    classifier = RecordingClassifier()
    monkeypatch.setattr(
        train_probes, "build_classifier", Mock(return_value=classifier)
    )
    train_labels = np.tile([0, 1], 20)
    test_labels = np.tile([0, 1], 4)
    train_df = pd.DataFrame(
        {"row_id": [f"train-{i}" for i in range(40)], "label": train_labels}
    )
    test_df = pd.DataFrame(
        {"row_id": [f"test-{i}" for i in range(8)], "label": test_labels}
    )
    train_matrix = np.column_stack(
        (np.zeros(40), np.arange(40, dtype="float32"))
    ).astype("float32")
    test_matrix = np.column_stack(
        (np.full(8, 100, dtype="float32"), np.arange(8, dtype="float32"))
    )

    comparison, metadata, predictions, selected_model = train_probes.train_and_compare(
        {"answer_layer_0": train_matrix},
        train_df,
        {"answer_layer_0": test_matrix},
        test_df,
        requested_layers=[-1],
        classifiers=["logistic_regression"],
        validation_size=0.2,
        seed=13,
    )

    assert np.max(classifier.fit_features[:, 0]) < 50
    assert len(classifier.probability_inputs) == 2
    assert np.max(classifier.probability_inputs[0][:, 0]) < 50
    assert np.all(classifier.probability_inputs[1][:, 0] == 100)
    assert metadata["test_labels_used_for_fit_or_selection"] is False
    assert metadata["selected_probe"]["validation_f1"] == comparison[0]["validation_f1"]
    assert predictions["row_id"].tolist() == test_df["row_id"].tolist()
    assert selected_model is classifier
    assert comparison[0]["split"] == "official_test"


def test_combined_probe_grid_contains_all_requested_configs_and_layers(monkeypatch):
    monkeypatch.setattr(
        train_probes,
        "build_classifier",
        Mock(side_effect=[RecordingClassifier() for _ in range(16)]),
    )
    train_labels = np.tile([0, 1], 20)
    test_labels = np.tile([0, 1], 4)
    train_df = pd.DataFrame(
        {"row_id": [f"train-{i}" for i in range(40)], "label": train_labels}
    )
    test_df = pd.DataFrame(
        {"row_id": [f"test-{i}" for i in range(8)], "label": test_labels}
    )
    requested_layers = [-1, -2, -4, -6]
    layer_indices = [6, 5, 3, 1]
    train_features = {}
    test_features = {}
    for segment_index, segment in enumerate(("question", "context", "answer")):
        for layer in layer_indices:
            key = f"{segment}_layer_{layer}"
            train_features[key] = np.column_stack(
                (np.full(40, segment_index, dtype="float32"), np.arange(40))
            )
            test_features[key] = np.column_stack(
                (np.full(8, segment_index, dtype="float32"), np.arange(8))
            )

    comparison, metadata, predictions, _ = train_probes.train_and_compare(
        train_features,
        train_df,
        test_features,
        test_df,
        requested_layers=requested_layers,
        classifiers=["logistic_regression"],
        validation_size=0.2,
        seed=13,
    )

    assert metadata["candidate_count"] == 16
    assert metadata["resolved_layers"] == layer_indices
    assert metadata["requested_configurations"] == [
        "answer_only",
        "question_answer",
        "context_answer",
        "question_context_answer",
    ]
    assert {row["layer"] for row in comparison} == set(layer_indices)
    assert {row["feature_segment"] for row in comparison} == set(
        metadata["requested_configurations"]
    )
    selected = metadata["selected_probe"]
    assert selected["feature_segment"] in metadata["requested_configurations"]
    assert selected["layer"] in layer_indices
    assert selected["classifier"] == "logistic_regression"
    assert selected["threshold"] is not None
    assert selected["validation_f1"] is not None
    fit_ids = set(metadata["fit_train_row_ids"])
    validation_ids = set(metadata["validation_row_ids"])
    assert fit_ids.isdisjoint(validation_ids)
    assert fit_ids | validation_ids == set(train_df["row_id"])
    assert len(fit_ids) == len(train_df) - len(validation_ids)
    assert metadata["seed"] == 13
    assert metadata["test_labels_used_for_fit_or_selection"] is False
    assert predictions["row_id"].tolist() == test_df["row_id"].tolist()


def test_probe_comparison_rejects_missing_test_feature_key():
    labels = np.tile([0, 1], 20)
    train = pd.DataFrame(
        {"row_id": [f"train-{i}" for i in range(40)], "label": labels}
    )
    test = pd.DataFrame(
        {"row_id": [f"test-{i}" for i in range(8)], "label": np.tile([0, 1], 4)}
    )
    train_features = {
        "answer_layer_0": np.ones((40, 2), dtype="float32"),
        "question_layer_0": np.ones((40, 2), dtype="float32"),
    }
    test_features = {"answer_layer_0": np.ones((8, 2), dtype="float32")}

    with pytest.raises(ValueError, match="Official test cache is missing feature segment"):
        train_probes.train_and_compare(
            train_features,
            train,
            test_features,
            test,
            requested_layers=[-1],
            classifiers=["logistic_regression"],
            validation_size=0.2,
            seed=13,
        )


def test_probe_comparison_rejects_missing_requested_training_layer():
    labels = np.tile([0, 1], 20)
    train = pd.DataFrame(
        {"row_id": [f"train-{i}" for i in range(40)], "label": labels}
    )
    test = pd.DataFrame(
        {"row_id": [f"test-{i}" for i in range(8)], "label": np.tile([0, 1], 4)}
    )
    train_features = {
        "answer_layer_1": np.ones((40, 2), dtype="float32"),
    }
    test_features = {
        "answer_layer_0": np.ones((8, 2), dtype="float32"),
        "answer_layer_1": np.ones((8, 2), dtype="float32"),
    }

    with pytest.raises(ValueError, match="Training cache is missing feature key"):
        train_probes.train_and_compare(
            train_features,
            train,
            test_features,
            test,
            requested_layers=[-2],
            classifiers=["logistic_regression"],
            validation_size=0.2,
            seed=13,
        )


def test_probe_comparison_rejects_mismatched_feature_dimensions():
    labels = np.tile([0, 1], 20)
    train = pd.DataFrame(
        {"row_id": [f"train-{i}" for i in range(40)], "label": labels}
    )
    test = pd.DataFrame(
        {"row_id": [f"test-{i}" for i in range(8)], "label": np.tile([0, 1], 4)}
    )

    with pytest.raises(ValueError, match="Train/test dimensions differ for answer_layer_0"):
        train_probes.train_and_compare(
            {"answer_layer_0": np.ones((40, 2), dtype="float32")},
            train,
            {"answer_layer_0": np.ones((8, 3), dtype="float32")},
            test,
            requested_layers=[-1],
            classifiers=["logistic_regression"],
            validation_size=0.2,
            seed=13,
        )


def test_train_test_ids_must_not_overlap():
    train = pd.DataFrame({"row_id": ["shared"], "label": [0]})
    test = pd.DataFrame({"row_id": ["shared"], "label": [0]})
    with pytest.raises(ValueError, match="share 1 row IDs"):
        train_probes.check_disjoint_ids(train, test)


def test_paired_run_writes_separate_probe_artifacts_and_baseline_comparison(tmp_path):
    train_count = 40
    test_count = 12
    train = pd.DataFrame(
        {
            "row_id": [f"train-{index}" for index in range(train_count)],
            "question": ["q"] * train_count,
            "context": ["c"] * train_count,
            "answer": ["a"] * train_count,
            "label": np.tile([0, 1], train_count // 2),
        }
    )
    test = pd.DataFrame(
        {
            "row_id": [f"test-{index}" for index in range(test_count)],
            "question": ["q"] * test_count,
            "context": ["c"] * test_count,
            "answer": ["a"] * test_count,
            "label": np.tile([0, 1], test_count // 2),
        }
    )
    train_data = tmp_path / "train.csv"
    test_data = tmp_path / "test.csv"
    train.to_csv(train_data, index=False)
    test.to_csv(test_data, index=False)
    train_features_path = tmp_path / "feature-cache" / "train.npz"
    test_features_path = tmp_path / "feature-cache" / "test.npz"
    train_features_path.parent.mkdir()
    values = np.column_stack(
        (train["label"].to_numpy(dtype="float32"), np.arange(train_count))
    )
    test_values = np.column_stack(
        (test["label"].to_numpy(dtype="float32"), np.arange(test_count))
    )
    for feature_path, frame, matrix, split in (
        (train_features_path, train, values, "train"),
        (test_features_path, test, test_values, "test"),
    ):
        np.savez_compressed(
            feature_path,
            row_id=frame["row_id"].to_numpy(dtype=str),
            labels=frame["label"].to_numpy(dtype="int64"),
            row_index=np.arange(len(frame)),
            answer_layer_0=matrix.astype("float32"),
        )
        feature_path.with_suffix(".metadata.json").write_text(
            json.dumps(
                {
                    "split": split,
                    "partial_split_cache": False,
                    "source_row_count": len(frame),
                }
            ),
            encoding="utf-8",
        )

    metadata = train_probes.run_paired(
        train_features_path,
        test_features_path,
        train_data,
        test_data,
        tmp_path / "probe-artifacts",
        requested_layers=[-1],
        classifiers=["logistic_regression"],
        validation_size=0.2,
        seed=9,
        baseline_metrics_path=None,
    )
    output = tmp_path / "probe-artifacts"
    assert metadata["test_data_scope"] == "full_split"
    assert metadata["test_labels_used_for_fit_or_selection"] is False
    assert (output / "models" / "answer_layer_0_logistic_regression.joblib").is_file()
    saved_model = joblib.load(
        output / "models" / "answer_layer_0_logistic_regression.joblib"
    )
    assert saved_model.predict_proba(values[:2]).shape == (2, 2)
    assert (output / "comparison_official_test.csv").is_file()
    assert (output / "comparison_official_test.json").is_file()
    saved_metadata = json.loads((output / "run_metadata.json").read_text())
    selected = saved_metadata["selected_probe"]
    assert selected["feature_segment"] == "answer_only"
    assert selected["layer"] == 0
    assert selected["classifier"] == "logistic_regression"
    assert selected["threshold"] is not None
    assert selected["validation_f1"] is not None
    assert saved_metadata["seed"] == 9
    assert saved_metadata["validation_row_ids"]
    assert saved_metadata["fit_train_row_ids"]
    assert (output / "split_membership.csv").is_file()
    assert (output / "selected_probe_predictions_official_test.csv").is_file()
    assert (output / "layer_comparison_f1_official_test.png").is_file()
    assert (tmp_path / "feature-cache" / "train.npz").is_file()


def test_calibrated_linear_svm_produces_probabilities():
    labels = np.tile([0, 1], 20)
    features = np.column_stack(
        (
            labels.astype("float32") * 2 - 1,
            np.linspace(-1, 1, len(labels), dtype="float32"),
            np.ones(len(labels), dtype="float32"),
        )
    )
    classifier = train_probes.build_classifier("linear_svm", 42)
    classifier.fit(features[:32], labels[:32])
    probabilities = classifier.predict_proba(features[32:])
    assert probabilities.shape == (8, 2)
    assert np.isfinite(probabilities).all()
