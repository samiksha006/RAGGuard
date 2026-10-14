"""Prepare the official RAGTruth train and test Parquet files."""
import argparse
import ast
import json
import math
from numbers import Real
from pathlib import Path

import pandas as pd


RAW_FILES = (
    "train-00000-of-00001.parquet",
    "test-00000-of-00001.parquet",
)
REQUIRED_COLUMNS = {
    "id",
    "query",
    "context",
    "output",
    "model",
    "task_type",
    "hallucination_labels_processed",
}
ANNOTATION_KEYS = {"evident_conflict", "baseless_info"}


def parse_labels(value: object) -> dict[str, object]:
    """Parse and validate one annotation dictionary without evaluating code."""
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError("annotation is not valid JSON or a Python literal") from exc

    if not isinstance(parsed, dict):
        raise ValueError("annotation must be a dictionary")
    missing = ANNOTATION_KEYS - parsed.keys()
    if missing:
        raise ValueError(f"annotation is missing keys: {', '.join(sorted(missing))}")
    for key in ANNOTATION_KEYS:
        count = parsed[key]
        if isinstance(count, bool) or not isinstance(count, Real):
            raise ValueError(f"{key} must be a non-negative number")
        if not math.isfinite(float(count)) or count < 0:
            raise ValueError(f"{key} must be a finite non-negative number")
    return parsed


def aggregate_label(value: object) -> int:
    """Apply the verified response-level RAGTruth binary-label rule."""
    labels = parse_labels(value)
    return int(labels["evident_conflict"] > 0 or labels["baseless_info"] > 0)


def prepare_dataframe(df: pd.DataFrame, split_name: str = "input") -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{split_name}: missing required columns: {', '.join(sorted(missing))}")

    if df["id"].isna().any():
        rows = df.index[df["id"].isna()].tolist()
        raise ValueError(f"{split_name}: {len(rows)} rows have a missing id; examples: {rows[:10]}")

    labels: list[int] = []
    invalid: list[tuple[object, str]] = []
    for index, value in df["hallucination_labels_processed"].items():
        try:
            labels.append(aggregate_label(value))
        except ValueError as exc:
            invalid.append((index, str(exc)))
            labels.append(-1)

    if invalid:
        examples = "; ".join(f"row {index}: {reason}" for index, reason in invalid[:10])
        raise ValueError(
            f"{split_name}: {len(invalid)} invalid annotation row(s); "
            f"no output was written. Examples: {examples}"
        )

    result = pd.DataFrame(
        {
            "id": df["id"].to_numpy(),
            "row_id": df["id"].to_numpy(),
            "question": df["query"].fillna("").astype(str).str.strip(),
            "context": df["context"].fillna("").astype(str).str.strip(),
            "answer": df["output"].fillna("").astype(str).str.strip(),
            "label": labels,
            "source_id": df["id"].to_numpy(),
            "model": df["model"].fillna("").astype(str),
            "task_type": df["task_type"].fillna("").astype(str),
        }
    )
    return result


def convert_files(raw_dir: Path, output_dir: Path) -> None:
    prepared: list[tuple[Path, pd.DataFrame]] = []
    for filename in RAW_FILES:
        source = raw_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Dataset file not found: {source}")
        source_df = pd.read_parquet(source)
        prepared.append((source, prepare_dataframe(source_df, filename)))

    ids_by_split = [
        set(frame["id"].tolist())
        for _, frame in prepared
    ]
    overlap = ids_by_split[0] & ids_by_split[1]
    if overlap:
        raise ValueError(
            f"Train/test contain {len(overlap)} overlapping IDs; "
            "refusing to write split outputs."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for (source, frame), split in zip(prepared, ("train", "test")):
        destination = output_dir / f"{split}.csv"
        frame.to_csv(destination, index=False)
        print(f"Saved: {destination}")
        print(f"Rows: {len(frame)} | Invalid annotations: 0")
        print(f"Labels: {frame['label'].value_counts().sort_index().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ragtruth"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/ragtruth"))
    args = parser.parse_args()
    convert_files(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
