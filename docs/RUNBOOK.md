# Runbook (AI‑RHEUM, Single‑Ontology)

Reproduces the AI‑RHEUM-only evaluation (mechanical exact agreement; no clinical claims).

## Artifacts (AI‑RHEUM)

- Label set: [data/ai_rheum_label_set.json](../data/ai_rheum_label_set.json)
- Seed dataset: [data/seed_cases_ai_rheum.csv](../data/seed_cases_ai_rheum.csv)
- Corpus cache: [data/ai_rheum_corpus.jsonl](../data/ai_rheum_corpus.jsonl)
- Retriever cache dir: [data/retriever_cache/ai_rheum/](../data/retriever_cache/)
- Outputs: [results/](../results/)

## Environment variables

Recommended (BioPortal):
- `BIOPORTAL_API_KEY`

Optional LLM backends:
- `USE_HUGGINGFACE=true` (local; optional `HF_MODEL`)
- `GOOGLE_API_KEY` (Gemini; optional `GEMINI_MODEL`)
- `OPENAI_API_KEY` (OpenAI)

Optional reproducibility knob:
- `LLM_TEMPERATURE` (recommend `0`)

## Local reproduction

### 0) Setup

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 1) Build/refresh the AI‑RHEUM corpus

Writes [data/ai_rheum_corpus.jsonl](../data/ai_rheum_corpus.jsonl).

```bash
python3 exploratory/build_corpus.py
```

### 2) Smoke test retrieval (optional)

```bash
PYTHONPATH=src python3 - <<'PY'
from tco_corpus import load_tco_corpus
from retrievers import create_retriever

corpus = load_tco_corpus("data/ai_rheum_corpus.jsonl")
retriever = create_retriever(
    corpus,
    top_k=3,
    prefer_embeddings=True,
    cache_dir="data/retriever_cache/ai_rheum",
)

hits = retriever.retrieve("joint pain with morning stiffness and swelling")
for i, d in enumerate(hits, 1):
    print(i, d.get("label"), d.get("tco_id"))
PY
```

Note: the corpus identifier field is `tco_id` (legacy naming; values are AI‑RHEUM URIs).

### 3) Run official evaluation on the AI‑RHEUM seed dataset

This matches the paths recorded in [results/results.json](../results/results.json).

```bash
python3 evaluate.py \
  --ontology-key ai_rheum \
  --label-set data/ai_rheum_label_set.json \
  --dataset data/seed_cases_ai_rheum.csv \
  --corpus data/ai_rheum_corpus.jsonl \
  --retriever-cache-dir data/retriever_cache/ai_rheum \
  --results-dir results
```

Outputs:
- `results/results.json`
- `results/predictions.csv`
- `results/summary.md`

## Colab sketch (optional)

This repo includes [src/colab_setup.py](../src/colab_setup.py) for a basic Colab install flow.

Checklist:
- [ ] Upload/clone repo into Colab
- [ ] `pip install -r requirements.txt`
- [ ] Set `BIOPORTAL_API_KEY` and one LLM backend env var
- [ ] Run the same corpus build + evaluation commands as in “Local reproduction”
