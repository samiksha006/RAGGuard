# RAGGuard — RAG Hallucination Detection with Classical Probes

Research-oriented project scaffold for the 19MAM56 Advanced Machine Learning Lab CAT-II rubric. It compares text-only baselines with classical classifiers trained on hidden-state features from an observer language model, and includes source-grouped evaluation, layer comparisons, calibration, efficiency tracking, visualizations, FastAPI, Streamlit, Docker, tests, and report guidance.

> **Scientific caveat:** RAGTruth's historical answers do not include the original generator's hidden states. This repository's default hidden-state pipeline therefore extracts *observer-model* representations by feeding the question/context/answer through a selected open-weight model. Do not describe these as the original answer generator's activations. The detector estimates hallucination risk; it does not prove a claim is false.

## Rubric coverage

| Rubric area | Included implementation |
|---|---|
| Data engineering & visualization | Flexible RAGTruth loader, label normalization, missing-data checks, class balance, PCA, confusion matrix, ROC/PR, reliability, layer plots |
| Advanced modelling & optimization | TF-IDF + Logistic Regression, hidden-state Logistic Regression / Linear SVM / Random Forest / optional XGBoost, Optuna optional, config-driven runs |
| Benchmarking & comparison | Shared splits, precision/recall/F1/ROC-AUC/PR-AUC, text-only vs probe, source-grouped holdout option |
| Deployment & containerization | FastAPI validated `/predict`, Streamlit UI, Dockerfile, health endpoint |
| Documentation | Report template, experiment protocol, limitations, references, runbook |

## Quick start (CPU-friendly)

Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_demo_data.py --rows 600
python -m src.data.prepare --input data/demo/ragtruth_demo.csv --output data/processed/dataset.csv
python -m src.models.train_text_baseline --input data/processed/dataset.csv
python -m src.evaluation.evaluate --predictions artifacts/text_baseline/predictions.csv --output-dir reports/results
```

The synthetic dataset is only for smoke-testing the software. **Never report demo-data metrics as research results.**

## Use the real RAGTruth dataset

The converter expects the already-downloaded Parquet files at
`data/raw/ragtruth/train-00000-of-00001.parquet` and
`data/raw/ragtruth/test-00000-of-00001.parquet`. It does not download or alter
the source files. The verified response-level label is `1` when
`evident_conflict > 0` **or** `baseless_info > 0`, and `0` when both are zero.
Missing or malformed annotation dictionaries cause preparation to fail with
the invalid row count and examples; they are never treated as grounded.

Run the following from the repository directory in Windows PowerShell:

```powershell
Set-Location "C:\Users\samik\Downloads\RAGGuard_Project\ragguard_project"
.\.venv\Scripts\Activate.ps1
python scripts/prepare_ragtruth.py
python -m src.models.train_text_baseline --train-input data/processed/ragtruth/train.csv --test-input data/processed/ragtruth/test.csv --output-dir artifacts/text_baseline --seed 42
python -m src.evaluation.evaluate --predictions artifacts/text_baseline/predictions_official_test.csv --output-dir reports/results --split-name official_test
python -m src.evaluation.make_report --data data/processed/ragtruth/test.csv --predictions artifacts/text_baseline/predictions_official_test.csv --output-dir reports/results --split-name official_test
```

Preparation writes `train.csv` and `test.csv`, retaining the RAGTruth ID,
question, context, answer, binary label, model, and task type. The paired-input
trainer splits **only the training CSV** into a stratified training and
validation partition. TF-IDF and Logistic Regression are fitted on the training
partition; the F1 threshold is selected on validation; the official test CSV is
used once for final predictions and metrics. No final refit uses validation or
test data. The legacy `--input` option remains available for synthetic smoke
tests only; never report its holdout metrics as RAGTruth research results.

The trained model, official test predictions, full metrics (including PR-AUC,
ROC-AUC, Brier score, training time, and per-sample inference latency), and
validation-selected threshold are saved under `artifacts/text_baseline/`.
Evaluation plots are named with `_official_test` and include the confusion
matrix, ROC curve, precision-recall curve, and reliability diagram. Dataset
report metadata identifies the official test split. PCA is intentionally
omitted from that report so a projection is not fitted on the official test
data.

The configured default encoder is DistilBERT as a reproducible observer baseline. An encoder swap changes the observer and should be recorded with its exact model revision and extraction settings.

## Research protocol

- **RQ1:** Do hidden-state probes outperform TF-IDF and sentence/text embeddings?
- **RQ2:** Which layers are most predictive? Compare each layer using identical splits.
- **RQ3:** Does performance hold for unseen source documents? Use group-based split by `source_id` where available.
- **RQ4:** Are scores calibrated? Report Brier score, reliability diagram, and risk/coverage when abstaining on uncertain predictions.
- **RQ5:** What is the accuracy/latency/memory trade-off for layer selection and feature dimensionality?

Prevent leakage: split before fitting vectorizers/scalers/PCA; tune only on validation data; keep the test set untouched; avoid duplicate responses across splits; group by source/question where possible. Report seed, model ID, library versions, sample counts, split strategy, and limitations.

## Hidden-state extraction and probes

These are observer-model representations: the selected Hugging Face model
re-encodes each RAGTruth question, retrieved context, and answer. They are not
the original answer generator's activations unless that same generator produced
the answer and its activations were captured at generation time. Features are
extracted separately for question, context, and answer. Each field is tokenized
independently, truncated independently to `max_length`, then every hidden-state
output is mean-pooled over all attended tokens, including tokenizer-added
special tokens. No answer-only pooling is implied.

The initial model, batch size, maximum length, and probe settings are in
`configs/model_config.yaml` and `configs/experiment_matrix.yaml`. Extraction
creates split-specific compressed feature caches in `data/features/`; probe
models and evaluation results are written to `artifacts/probes/`, not to the
existing `artifacts/text_baseline/` directory. The official test CSV and labels
are only used for final probe inference and metric reporting. Scalers are part
of each model pipeline and fit only on the training partition; the calibrated
Linear SVM's calibration folds also come from training rows. Probe and text
baseline metrics use the same full official test split, with threshold/model
selection based on training validation F1.

Run a small, 64-row-per-split smoke extraction first from Windows PowerShell.
This invokes the configured Hugging Face model and may fetch model files on its
first use. It does not run full-dataset extraction. `--no-baseline` is important
here because smoke-subset scores are not comparable to the existing full-test
text baseline.

```powershell
Set-Location "C:\Users\samik\Downloads\RAGGuard_Project\ragguard_project"
.\.venv\Scripts\Activate.ps1

python -m src.features.extract_hidden_states --train-input data/processed/ragtruth/train.csv --test-input data/processed/ragtruth/test.csv --output-dir data/features/ragtruth_smoke --max-rows 64 --max-length 128 --batch-size 8

python -m src.models.train_probes --train-features data/features/ragtruth_smoke/train.npz --test-features data/features/ragtruth_smoke/test.npz --train-data data/processed/ragtruth/train.csv --test-data data/processed/ragtruth/test.csv --output-dir artifacts/probes/ragtruth_smoke --layers -1 --models logistic_regression --no-baseline
```

After the smoke test succeeds, run full extraction and probe training:

```powershell
python -m src.features.extract_hidden_states --train-input data/processed/ragtruth/train.csv --test-input data/processed/ragtruth/test.csv --output-dir data/features/ragtruth

python -m src.models.train_probes --train-features data/features/ragtruth/train.npz --test-features data/features/ragtruth/test.npz --train-data data/processed/ragtruth/train.csv --test-data data/processed/ragtruth/test.csv --output-dir artifacts/probes/ragtruth
```

The extractor writes `train.npz` and `test.npz` separately, each carrying
`row_id`, `labels`, `row_index`, and per-segment/per-layer feature arrays, plus
metadata JSON containing the split name, IDs, counts, model revision, pooling
rule, and extraction settings. The probe trainer verifies exact CSV/cache ID and
label alignment and disjoint split IDs. It uses a stratified validation split
from official training rows to select the best segment/layer/classifier by F1
and selects each candidate's threshold on that validation split. It then
evaluates candidate predictions on the untouched official test rows, saves the
validation-selected probe model and predictions, and writes
`comparison_official_test.csv`/`.json`, split membership, and layer-comparison
plots under `artifacts/probes/ragtruth/`. The comparison includes F1, precision,
recall, PR-AUC, ROC-AUC, Brier score, training time, and per-sample test
inference latency for probes and (when available) the full official-test text
baseline metrics.

`--max-rows` marks feature caches as partial in their metadata; smoke results
must not be represented as full official-test experiment results. To select a
different compatible Hugging Face encoder, use `--model-name MODEL_ID` or edit
`features.model_name` in the model config. Hidden-state indices are configured
as negative indices in the experiment config, with `-1` meaning the last
hidden-state output. Extraction and probe training have not been run on the
full dataset as part of this code change; no hidden-state performance results
are claimed.

## Visualizations and report

```bash
python -m src.evaluation.make_report --data data/processed/dataset.csv --predictions artifacts/text_baseline/predictions.csv --output-dir reports/results
```

For probe results, pass the relevant predictions CSV. The script writes PNGs and a JSON metric summary. The report skeleton is `reports/REPORT_TEMPLATE.md`; replace placeholders with actual results, add your architecture diagram, and keep the final report within the course's page limit.

## Run the app

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
streamlit run ui/app.py
```

The API loads the trained text baseline by default. To serve hidden-state probes, set `RAGGUARD_MODE=probe` and `RAGGUARD_PROBE_PATH=artifacts/probe/model.joblib`, plus configure the feature extractor/model. Probe inference requires loading the same observer model used during feature extraction; the default API intentionally refuses to pretend a text model is a hidden-state detector. See `api/main.py` for the extension hook.

## Docker

```bash
docker build -t ragguard:latest .
docker run --rm -p 8000:8000 ragguard:latest
```

For public deployment, push the image to a registry and deploy to a platform you control, or deploy Streamlit to Hugging Face Spaces. No cloud URL is included because deployment requires your account credentials and a real build/deploy action.

## Project structure

```text
configs/              model/run configuration
src/data/             dataset loading and preparation
src/features/         hidden-state extraction
src/models/            baselines, probes, tuning
src/evaluation/        metrics, error analysis, plots
api/                   FastAPI service
ui/                    Streamlit UI
scripts/               demo-data generator and RAGTruth preparation
 tests/                smoke tests
reports/               report template and generated outputs
artifacts/             model outputs (ignored by git)
```

## Important limitations

- Hidden-state probes are model/checkpoint dependent.
- Observer-model activations are not the original generator's internal states unless you regenerate the answers with that same model and record its activations.
- Dataset annotation conventions and class balance can materially affect metrics.
- Span-level localization is not implemented as a validated result in this starter; response-level detection is the primary target.
- A working local API/Docker setup is included, but cloud hosting, benchmark results, and full-scale training must be completed and verified on your machine.

## Deployment runbook

See [`README_DEPLOYMENT.md`](README_DEPLOYMENT.md) for Docker Hub and Hugging Face Spaces steps. `Dockerfile.streamlit` is provided for a separate UI container. Deployment is not claimed complete until you deploy using your own account and verify the public URL.
