"""Extract pooled observer-model features, separately for question, context and answer."""
import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SEGMENTS = ("question", "context", "answer")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def validate_dataframe(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    required = {*SEGMENTS, "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError(f"{path}: dataset is empty")
    labels = pd.to_numeric(df["label"], errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{path}: label values must be non-missing binary 0/1")

    result = df.copy().reset_index(drop=True)
    result["label"] = labels.astype("int64").to_numpy()
    id_column = "row_id" if "row_id" in result else "id" if "id" in result else None
    if id_column is None:
        raise ValueError(f"{path}: expected a stable row_id or id column")
    if result[id_column].isna().any():
        raise ValueError(f"{path}: {id_column} contains missing values")
    result["row_id"] = result[id_column].astype(str)
    if result["row_id"].duplicated().any():
        raise ValueError(f"{path}: row IDs must be unique within a split")
    for segment in SEGMENTS:
        result[segment] = result[segment].fillna("").astype(str)
    return result


def mean_pool_hidden_states(hidden_states: Any, attention_mask: Any) -> Any:
    """Mean-pool all attended tokens, including tokenizer-added special tokens."""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def extract_features(
    df: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[str, np.ndarray]:
    layers: dict[str, list[np.ndarray]] = {
        f"{segment}_layer_{index}": []
        for segment in SEGMENTS
        for index in range(model.config.num_hidden_layers + 1)
    }
    with torch.no_grad():
        for segment in SEGMENTS:
            texts = df[segment].tolist()
            for start_index in range(0, len(texts), batch_size):
                batch = texts[start_index : start_index + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                outputs = model(**encoded, output_hidden_states=True, return_dict=True)
                for layer_index, hidden in enumerate(outputs.hidden_states):
                    pooled = mean_pool_hidden_states(hidden, encoded["attention_mask"])
                    layers[f"{segment}_layer_{layer_index}"].append(
                        pooled.detach().cpu().numpy().astype("float32")
                    )
                print(
                    f"Encoded {segment}: "
                    f"{min(start_index + len(batch), len(texts))}/{len(texts)}"
                )

    result: dict[str, np.ndarray] = {
        key: np.concatenate(parts, axis=0)
        if parts
        else np.empty((0, model.config.hidden_size), dtype="float32")
        for key, parts in layers.items()
    }
    result["labels"] = df["label"].to_numpy(dtype="int64")
    result["row_index"] = np.arange(len(df), dtype="int64")
    result["row_id"] = df["row_id"].to_numpy(dtype=str)
    return result


def save_split(
    df: pd.DataFrame,
    split_name: str,
    output_path: Path,
    *,
    model_name: str,
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: str,
    batch_size: int,
    max_length: int,
    source_row_count: int,
) -> None:
    started = time.perf_counter()
    payload = extract_features(
        df, tokenizer, model, torch, device, batch_size, max_length
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    metadata = {
        "split": split_name,
        "rows": len(df),
        "source_row_count": source_row_count,
        "partial_split_cache": len(df) != source_row_count,
        "row_ids": df["row_id"].tolist(),
        "label_counts": {
            str(label): int(count)
            for label, count in df["label"].value_counts().sort_index().items()
        },
        "model_name": model_name,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_name": getattr(tokenizer, "name_or_path", model_name),
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "max_length_per_segment": max_length,
        "batch_size": batch_size,
        "device": device,
        "pooling": (
            "Per-field attention-mask mean pooling. Question, context and answer "
            "are tokenized independently; each representation averages every "
            "attended token in that field, including tokenizer-added special "
            "tokens. Each field truncates independently at max_length."
        ),
        "representation_type": (
            "Observer-model representations; not necessarily the original "
            "answer-generating model's activations."
        ),
        "feature_keys": [
            key for key in payload if "_layer_" in key
        ],
        "row_order": "row_index and row_id correspond to this split CSV order",
        "extraction_seconds": time.perf_counter() - started,
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "row_ids"}, indent=2))


def load_transformers() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Hidden-state extraction requires torch and transformers. "
            "Install the project requirements first."
        ) from exc
    return torch, AutoTokenizer, AutoModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Legacy single CSV input.")
    input_group.add_argument("--train-input", help="Training split CSV.")
    parser.add_argument("--test-input", help="Official test split CSV.")
    parser.add_argument("--output", help="Legacy single NPZ destination.")
    parser.add_argument(
        "--output-dir",
        help="Paired split cache directory (separate from probe model artifacts).",
    )
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--model-name")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit each split for smoke tests.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    feature_config = config.get("features", {})
    model_name = args.model_name or feature_config.get(
        "model_name", "distilbert/distilbert-base-uncased"
    )
    batch_size = args.batch_size or int(feature_config.get("batch_size", 4))
    max_length = args.max_length or int(feature_config.get("max_length", 384))
    if batch_size < 1 or max_length < 2:
        parser.error("--batch-size must be positive and --max-length must be at least 2")
    if args.max_rows < 0:
        parser.error("--max-rows must be zero or positive")

    paired = args.train_input is not None
    if paired:
        if not args.test_input or not args.output_dir or args.output:
            parser.error("Paired mode requires --test-input and --output-dir, not --output")
        train_df = validate_dataframe(pd.read_csv(args.train_input), args.train_input)
        test_df = validate_dataframe(pd.read_csv(args.test_input), args.test_input)
        source_row_counts = {"train": len(train_df), "test": len(test_df)}
        overlap = set(train_df["row_id"]) & set(test_df["row_id"])
        if overlap:
            parser.error(f"Train/test CSVs share {len(overlap)} row IDs.")
        if args.max_rows:
            train_df = train_df.head(args.max_rows).reset_index(drop=True)
            test_df = test_df.head(args.max_rows).reset_index(drop=True)
        splits = [("train", train_df), ("test", test_df)]
        output_dir = Path(args.output_dir)
        output_paths = {
            "train": output_dir / "train.npz",
            "test": output_dir / "test.npz",
        }
    else:
        if not args.output or args.test_input or args.output_dir:
            parser.error("Single-input mode requires --output only.")
        single_df = validate_dataframe(pd.read_csv(args.input), args.input)
        source_row_count = len(single_df)
        if args.max_rows:
            single_df = single_df.head(args.max_rows).reset_index(drop=True)
        source_row_counts = {"unspecified": source_row_count}
        splits = [("unspecified", single_df)]
        output_paths = {"unspecified": Path(args.output)}

    torch, tokenizer_class, model_class = load_transformers()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(
        model_name, output_hidden_states=True
    ).to(device)
    model.eval()

    for split_name, df in splits:
        save_split(
            df,
            split_name,
            output_paths[split_name],
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            source_row_count=source_row_counts[split_name],
        )

    if paired:
        pair_metadata = {
            "train_feature_file": str(output_paths["train"]),
            "test_feature_file": str(output_paths["test"]),
            "train_rows": len(splits[0][1]),
            "test_rows": len(splits[1][1]),
            "train_source_rows": source_row_counts["train"],
            "test_source_rows": source_row_counts["test"],
            "partial_split_cache": args.max_rows > 0,
            "train_test_row_id_overlap": 0,
            "model_name": model_name,
            "purpose": "Separate official train and test split caches.",
        }
        (Path(args.output_dir) / "splits.metadata.json").write_text(
            json.dumps(pair_metadata, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
