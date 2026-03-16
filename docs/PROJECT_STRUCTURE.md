# Project Structure

## Top-level layout

```
bime/550/
├── classes/                  # Core domain modules (ontology, retrievers, llm)
├── config/                   # Shared constants and path defaults
├── data/                     # Input artifacts + caches
├── docs/                     # Project documentation
├── exploratory/              # Diagnostics scripts
├── results/                  # Evaluation outputs
├── src/                      # Entrypoints + orchestration pipeline
├── evaluate.py               # Root entrypoint for official eval
└── run.py                    # Legacy rag_exp runner
```

## Evaluation stack (official path)

`python3 -m evaluate` executes the following flow:

1. `evaluate.py`
2. `src/eval_official.py`
3. `src/eval_pipeline.py`
   - `prepare_runtime(...)`
   - `run_prediction_stage(...)`
   - `assemble_results(...)`

## Key modules

- `config/constants.py`
  - shared constants (for example `DEFAULT_ONTOLOGY`)
- `config/paths.py`
  - centralized project-relative paths
- `src/eval_types.py`
  - typed run config (`RunConfig`)
- `src/schemas.py`
  - typed artifact schemas (`LabelSet`, `PredictionRow`, results sections)
- `src/eval_pipeline.py`
  - stage-based evaluation logic
- `src/eval_official.py`
  - CLI and output rendering/writing
- `classes/llm_interface.py`
  - backend abstraction + response validation
- `classes/retrievers.py`
  - embedding/FAISS/TF-IDF retrieval
- `classes/corpus.py`
  - BioPortal ingestion and corpus normalization

## Data-generation helpers

- `src/fetch_label_uris.py`
  - generate label set JSON from BioPortal
- `src/build_corpus.py`
  - generate corpus JSONL from label set
- `exploratory/retrieval_diagnostics.py`
  - inspect retriever outputs for a single query

## Primary artifacts

- `data/ai_rheum_label_set.json`
- `data/seed_cases_ai_rheum.csv`
- `data/ai_rheum_corpus.jsonl`
- `results/results.json`
- `results/predictions.csv`
- `results/summary.md`
