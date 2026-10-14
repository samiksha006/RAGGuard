"""Load flexible RAGTruth exports, normalize fields and create a clean response-level table."""
import argparse, json
from pathlib import Path
import pandas as pd

ALIASES = {
    "question": ["question", "query", "prompt"],
    "context": ["context", "retrieved_context", "evidence", "documents", "source_text"],
    "answer": ["response", "answer", "generated_response", "output", "text", "model_response"],
    "label": ["label", "hallucination", "is_hallucinated", "hallucination_label", "has_hallucination"],
    "source_id": ["source_id", "source", "doc_id", "document_id", "source_doc_id"],
}
POS = {"1", "true", "yes", "positive", "hallucinated", "hallucination", "contains_hallucination"}
NEG = {"0", "false", "no", "negative", "grounded", "non_hallucinated", "no_hallucination"}

def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        return pd.DataFrame(obj if isinstance(obj, list) else obj.get("data", obj))
    if p.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(p, sep="\t" if p.suffix.lower()==".tsv" else ",")
    raise ValueError(f"Unsupported file type: {p.suffix}. Use CSV, TSV, JSON or JSONL.")

def _find_col(df, target, aliases):
    low = {str(c).strip().lower(): c for c in df.columns}
    for name in aliases:
        if name.lower() in low:
            return low[name.lower()]
    return None

def normalize_label(value):
    if pd.isna(value): return None
    # Common numeric/boolean response-level labels.
    if isinstance(value, bool): return int(value)
    s = str(value).strip().lower()
    if s in POS: return 1
    if s in NEG: return 0
    # Span/annotation arrays are intentionally not guessed here.
    return None

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    mapped = {}
    for target, aliases in ALIASES.items():
        col = _find_col(df, target, aliases)
        if col is not None: mapped[target] = col
    # RAGTruth exports may provide annotation lists instead of binary response labels.
    if "label" not in mapped:
        span_candidates = [c for c in df.columns if any(k in str(c).lower() for k in ["annotation", "hallucination_spans", "span_label"])]
        if span_candidates:
            raise ValueError("A likely span-annotation column was found but no binary response label. Review RAGTruth annotation semantics and explicitly aggregate spans before preparing data.")
        raise ValueError(f"Could not find a binary label column. Columns found: {list(df.columns)}. Map the correct annotation to a response-level label after reviewing the dataset schema.")
    if "answer" not in mapped:
        raise ValueError(f"Could not find answer/response text column. Columns found: {list(df.columns)}")
    out = pd.DataFrame()
    for target in ["question", "context", "answer", "source_id"]:
        col = mapped.get(target)
        out[target] = df[col].fillna("").astype(str) if col else ""
    out["label"] = df[mapped["label"]].map(normalize_label)
    before = len(out)
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    for col in ["question", "context", "answer"]:
        out[col] = out[col].astype(str).str.strip()
    out = out[(out.answer.str.len() > 0)].drop_duplicates(subset=["question", "context", "answer", "label"])
    out = out.reset_index(drop=True)
    out["row_id"] = range(len(out))
    out.attrs["rows_removed"] = before - len(out)
    return out[["row_id", "question", "context", "answer", "label", "source_id"]]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args=ap.parse_args()
    raw=read_table(args.input); clean=prepare_dataframe(raw)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output,index=False)
    print(f"Input rows: {len(raw)} | Clean rows: {len(clean)} | Removed/unknown: {len(raw)-len(clean)}")
    print("Class counts:", clean.label.value_counts().sort_index().to_dict())
    print("Output:", args.output)
if __name__ == "__main__": main()
