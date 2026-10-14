import numpy as np
import pandas as pd
import pytest

from scripts.prepare_ragtruth import aggregate_label, parse_labels, prepare_dataframe
from src.models import train_text_baseline
from src.evaluation.evaluate import evaluate_predictions
from src.evaluation.make_report import make_report


def test_annotation_parser_accepts_dict_json_and_python_literal():
    assert parse_labels({"evident_conflict": 1, "baseless_info": 0}) == {
        "evident_conflict": 1,
        "baseless_info": 0,
    }
    assert aggregate_label('{"evident_conflict": 0, "baseless_info": 2}') == 1
    assert aggregate_label("{'evident_conflict': 0, 'baseless_info': 0}") == 0


@pytest.mark.parametrize(
    ("conflict", "baseless", "expected"),
    [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)],
)
def test_annotation_aggregation(conflict, baseless, expected):
    assert aggregate_label(
        {"evident_conflict": conflict, "baseless_info": baseless}
    ) == expected


@pytest.mark.parametrize(
    "annotation",
    [
        None,
        "not an annotation",
        "__import__('os').system('echo unsafe')",
        {"evident_conflict": 1},
        {"evident_conflict": -1, "baseless_info": 0},
        {"evident_conflict": 0, "baseless_info": "unknown"},
        {"evident_conflict": True, "baseless_info": 0},
    ],
)
def test_invalid_annotations_raise(annotation):
    with pytest.raises(ValueError):
        aggregate_label(annotation)


def test_dataframe_reports_invalid_annotation_rows():
    frame = pd.DataFrame(
        {
            "id": ["ok", "invalid"],
            "query": ["q1", "q2"],
            "context": ["c1", "c2"],
            "output": ["a1", "a2"],
            "model": ["m", "m"],
            "task_type": ["qa", "qa"],
            "hallucination_labels_processed": [
                {"evident_conflict": 0, "baseless_info": 0},
                {"evident_conflict": 0},
            ],
        }
    )
    with pytest.raises(ValueError, match="1 invalid annotation row"):
        prepare_dataframe(frame, "test")


def test_prepare_maps_text_and_preserves_ids_and_metadata():
    frame = pd.DataFrame(
        {
            "id": ["ragtruth-1"],
            "query": [" question "],
            "context": [" context "],
            "output": [" answer "],
            "model": ["model-a"],
            "task_type": ["qa"],
            "hallucination_labels_processed": [
                {"evident_conflict": 0, "baseless_info": 1}
            ],
        }
    )
    prepared = prepare_dataframe(frame, "train")
    row = prepared.iloc[0]
    assert row["id"] == row["row_id"] == row["source_id"] == "ragtruth-1"
    assert row["question"] == "question"
    assert row["context"] == "context"
    assert row["answer"] == "answer"
    assert row["label"] == 1
    assert row["model"] == "model-a"
    assert row["task_type"] == "qa"


class SpyModel:
    def __init__(self):
        self.fit_text = []
        self.fit_labels = None
        self.prediction_calls = []

    def fit(self, text, labels):
        self.fit_text = list(text)
        self.fit_labels = np.asarray(labels)
        return self

    def predict_proba(self, text):
        text = list(text)
        self.prediction_calls.append(text)
        probability = np.asarray(
            [0.8 if "positive" in item else 0.2 for item in text]
        )
        return np.column_stack((1 - probability, probability))


def test_official_test_is_only_used_after_training_and_threshold_selection(monkeypatch):
    spy_model = SpyModel()
    monkeypatch.setattr(
        train_text_baseline, "build_model", lambda _: spy_model
    )
    training = pd.DataFrame(
        {
            "id": [f"train-{i}" for i in range(40)],
            "question": [f"training question {i}" for i in range(40)],
            "context": ["training context"] * 40,
            "answer": [
                f"training {'positive' if i % 2 else 'negative'} {i}"
                for i in range(40)
            ],
            "label": [i % 2 for i in range(40)],
        }
    )
    official_test = pd.DataFrame(
        {
            "id": [f"official-test-{i}" for i in range(8)],
            "question": ["official test question"] * 8,
            "context": ["official test context"] * 8,
            "answer": [
                f"official-test {'positive' if i % 2 else 'negative'} {i}"
                for i in range(8)
            ],
            "label": [i % 2 for i in range(8)],
        }
    )

    train_text = train_text_baseline.make_text(training)
    test_text = train_text_baseline.make_text(official_test)
    assert not (set(train_text) & set(test_text))
    train_text_baseline.check_split_ids(training, official_test)
    _, predictions, result = train_text_baseline.fit_and_score(
        training,
        official_test,
        seed=17,
        validation_size=0.2,
        split_name="official_test",
    )

    assert len(spy_model.fit_text) == result["n_train"]
    assert all("official-test" not in text for text in spy_model.fit_text)
    assert len(spy_model.prediction_calls) == 2
    assert all(
        "official-test" not in text for text in spy_model.prediction_calls[0]
    )
    assert all(
        "official-test" in text for text in spy_model.prediction_calls[1]
    )
    assert result["threshold_selected_on"] == "training_validation"
    assert result["test_split"] == "official_test"
    assert len(predictions) == len(official_test)


def test_train_test_id_overlap_is_rejected():
    training = pd.DataFrame({"id": ["shared"]})
    test = pd.DataFrame({"id": ["shared"]})
    with pytest.raises(ValueError, match="overlapping"):
        train_text_baseline.check_split_ids(training, test)


def test_official_test_evaluation_outputs_have_split_names_and_metadata(tmp_path):
    predictions = pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1],
            "probability": [0.1, 0.9, 0.3, 0.7],
            "prediction": [0, 1, 0, 1],
            "answer": ["a", "b", "c", "d"],
        }
    )
    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    result = evaluate_predictions(
        predictions_path, tmp_path / "evaluation", "official_test"
    )

    assert result["evaluation_split"] == "official_test"
    for filename in (
        "confusion_matrix_official_test.png",
        "roc_curve_official_test.png",
        "pr_curve_official_test.png",
        "reliability_official_test.png",
        "evaluation_metrics_official_test.json",
    ):
        assert (tmp_path / "evaluation" / filename).is_file()

    data = pd.DataFrame(
        {
            "question": ["q1", "q2"],
            "context": ["c1", "c2"],
            "answer": ["a1", "a2"],
            "label": [0, 1],
        }
    )
    data_path = tmp_path / "test.csv"
    data.to_csv(data_path, index=False)
    report = make_report(
        data_path, predictions_path, tmp_path / "report", "official_test"
    )
    assert report["evaluation_split"] == "official_test"
    assert report["pca_visualization"].startswith("omitted")
    assert (tmp_path / "report" / "dataset_report_official_test.json").is_file()
    assert (tmp_path / "report" / "class_distribution_official_test.png").is_file()
