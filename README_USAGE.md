# AI-RHEUM RAG Evaluation Usage

This repository currently runs an official No-RAG vs RAG comparison for AI-RHEUM labels.

## What `evaluate` does

Running `python3 -m evaluate` executes `src/eval_official.py`, which:

1. Loads the AI-RHEUM config and label set.
2. Builds/loads a corpus for retrieval.
3. Runs predictions per case with:
   - No-RAG
   - RAG (top-k retrieved context)
4. Writes run artifacts:
   - `results/results.json`
   - `results/predictions.csv`
   - `results/summary.md`

Default project paths are centralized in `config/paths.py`.
Default ontology constants are centralized in `config/constants.py`.

## Code Organization

The evaluation stack is now split by responsibility:

- `config/constants.py`
  - shared constants (for example `DEFAULT_ONTOLOGY`)
- `config/paths.py`
  - project-relative path defaults
- `src/schemas.py`
  - typed artifact schemas (`LabelSet`, `PredictionRow`, results sections)
- `src/eval_types.py`
  - shared evaluation config dataclass (`RunConfig`)
- `src/eval_pipeline.py`
  - stage-based runtime pipeline:
    - `prepare_runtime(...)`
    - `run_prediction_stage(...)`
    - `assemble_results(...)`
- `src/eval_official.py`
  - thin CLI/orchestrator and summary rendering
- `src/bootstrap.py`
  - shared import bootstrap helper for direct script execution

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional env vars:

```bash
# BioPortal (recommended for label/corpus generation)
export BIOPORTAL_API_KEY="..."

# Hugging Face backend
export USE_HUGGINGFACE="true"
export HF_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# Optional generation controls
export LLM_DO_SAMPLE="false"
export LLM_TEMPERATURE="0"
export LLM_TOP_P="1.0"
```

## Required vs Auto-Generated Inputs

`evaluate` now auto-generates the label set if missing:

- If `data/ai_rheum_label_set.json` does not exist, eval automatically calls:
  - `python3 src/fetch_label_uris.py --ontology AI-RHEUM --output data/ai_rheum_label_set.json`

Still required before eval:

- `data/seed_cases_ai_rheum.csv`
  - Must include columns: `chart_text`, `gold_label`

Corpus behavior:

- Eval calls `load_or_build_corpus(...)` for `data/ai_rheum_corpus.jsonl`.
- If BioPortal is reachable, corpus is (re)built from label IDs.
- If BioPortal is unavailable, eval falls back to an existing corpus file on disk.
- If neither works, eval exits with an error.

## Run Commands

Default (embeddings retrieval):

```bash
python3 -m evaluate
```

Fully local retrieval (TF-IDF):

```bash
python3 -m evaluate --retrieval tfidf
```

Other useful flags:

```bash
python3 -m evaluate --k 3 --max-context-chars 1800 --results-dir results
```

## Manual Data Utilities

Regenerate label set manually:

```bash
python3 src/fetch_label_uris.py --ontology AI-RHEUM --output data/ai_rheum_label_set.json
```

Regenerate corpus manually:

```bash
python3 src/build_corpus.py
```

## Output Files

- `results/results.json`
  - Run metadata, agreement metrics, paired t-test, coercion counts
- `results/predictions.csv`
  - Per-case gold label, No-RAG prediction, RAG prediction, ddx/evidence fields
- `results/summary.md`
  - Human-readable run summary

## Notes

- `src/fetch_label_uris.py` is currently curated for `AI-RHEUM`.
- First embedding run is slower because retriever/model caches are created.
