# Runbook (AI‑RHEUM, Single‑Ontology)

This runbook reproduces the AI‑RHEUM-only evaluation.

- Ontology: AI‑RHEUM (BioPortal)
- Allowed labels: [data/label_set.json](../data/label_set.json) (AI‑RHEUM IDs ∪ NONE)
- Corpus cache: [data/ai_rheum_corpus.jsonl](../data/ai_rheum_corpus.jsonl)
- Dataset: [data/synthetic_charts.csv](../data/synthetic_charts.csv)
- Metric: exact percent agreement only

## Environment variables

Recommended:
- BIOPORTAL_API_KEY (higher BioPortal rate limits)

Optional LLM backends:
- GOOGLE_API_KEY (Gemini)
- OPENAI_API_KEY (OpenAI)
- USE_HUGGINGFACE=true (local HF)
- GEMINI_MODEL (optional override)
- HF_MODEL (optional override)

## 0) Setup

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r [requirements.txt](http://_vscodecontentref_/3)