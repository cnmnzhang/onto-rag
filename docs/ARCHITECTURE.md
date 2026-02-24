# Architecture (AI‑RHEUM, Single‑Ontology)

This repo is an ontology-grounded NLP methods experiment (not clinical decision-making). The system is constrained-label prediction over synthetic text, comparing No‑RAG vs RAG using a single ontology: AI‑RHEUM.

## Non‑negotiables

- Single ontology: AI‑RHEUM only (BioPortal-accessed).
- Constrain predictions to: selected AI‑RHEUM IDs ∪ NONE sentinel.
- Evaluation is mechanical: exact percent agreement only (no clinical claims, no stratified analyses).

## 2026-02 Scope Update: 12-Label Differential + Auxiliary Outputs

This repo’s **official metric remains unchanged**:
- **Scored field**: `predicted_label` only
- **Metric**: exact percent agreement, `predicted_label == gold_label`

The following are **auxiliary, non-scored** outputs used only for qualitative inspection:
- `ddx_top3` (up to 3 alternatives)
- `next_step` (one sentence)
- `evidence` (optional retrieved doc ids)

No label discovery is allowed. All outputs must map back to the constrained label set ∪ `NONE`.

## Identifier rule (most important)

IDs in [data/ai_rheum_label_set.json](../data/ai_rheum_label_set.json) **must exactly equal** the corpus identifier field in [data/ai_rheum_corpus.jsonl](../data/ai_rheum_corpus.jsonl) (byte-for-byte string match).

Important legacy naming note: the corpus currently stores the class identifier under the field name `tco_id`. For AI‑RHEUM runs, `tco_id` contains AI‑RHEUM URIs; the name is legacy.

## Key artifacts (expected)

- Label set: [data/ai_rheum_label_set.json](../data/ai_rheum_label_set.json) (AI‑RHEUM IDs + NONE)
- Corpus cache: [data/ai_rheum_corpus.jsonl](../data/ai_rheum_corpus.jsonl)
- Dataset: [data/seed_cases_ai_rheum.csv](../data/seed_cases_ai_rheum.csv)
- Retriever cache directory: [data/retriever_cache/ai_rheum/](../data/retriever_cache/)
- LLM cache: [data/llm_cache.json](../data/llm_cache.json)
- Official evaluation outputs:
  - [results/results.json](../results/results.json)
  - [results/predictions.csv](../results/predictions.csv)
  - [results/summary.md](../results/summary.md)

Done criteria:
- [ ] Label set contains **exactly 12 unique AI‑RHEUM URIs** under `labels`
- [ ] `none_label` is exactly the string `NONE`
- [ ] All gold labels in the evaluated dataset are in `labels ∪ {NONE}`
- [ ] Evaluation scoring uses `predicted_label` only (aux fields never affect agreement)

## Component map (module-level)

### 1) Ontology ingest (BioPortal → AI‑RHEUM terms)

Purpose: fetch ontology class metadata for a small, explicit set of AI‑RHEUM IDs (labels, synonyms, definitions, parents) and write an indexable JSONL corpus.

Where it lives:
- Ontology configuration registry: [src/classes/onto_config.py](../src/classes/onto_config.py)
- BioPortal fetch + JSONL normalization/caching: [src/classes/corpus.py](../src/classes/corpus.py)

Notes (legacy naming):
- The ingest module is named [src/classes/corpus.py](../src/classes/corpus.py). In the AI‑RHEUM-only setup it is treated as the generic BioPortal corpus builder, and the ontology-specific artifact is [data/ai_rheum_corpus.jsonl](../data/ai_rheum_corpus.jsonl).

Done criteria:
- [ ] AI‑RHEUM config exists in [src/classes/onto_config.py](../src/classes/onto_config.py) (acronym matches BioPortal)
- [ ] Building the corpus succeeds online (or falls back to an existing JSONL offline)
- [ ] Corpus records include stable fields used by retrievers (next section)

---

### 2) Corpus (ontology “documents”)

Purpose: store one “document” per allowed AI‑RHEUM ID for retrieval.

Where it lives:
- Corpus writer/loader + record normalization: [src/classes/corpus.py](../src/classes/corpus.py)

Artifact:
- [data/ai_rheum_corpus.jsonl](../data/ai_rheum_corpus.jsonl)

In-memory representation:
- The in-memory corpus is a `list[dict]` with (at minimum) keys: `tco_id`, `label`, `text`, `synonyms` (and optionally `definition`, `parent_labels`).

Data contract (per JSONL record; canonical field names in this repo):
- tco_id: string (**identifier**, must match entries in the label set JSON)
- label: string
- synonyms: list[string]
- text: string (primary indexable field)

Optional enrichment fields:
- definition: string
- parent_labels: list[string]

Naming note:
- `tco_id` is legacy naming from earlier TCO runs. For AI‑RHEUM, it stores AI‑RHEUM class URIs.

Done criteria:
- [ ] Each record has tco_id, label, text
- [ ] tco_id values are unique and match the label set IDs exactly
- [ ] Corpus length equals number of labels in [data/ai_rheum_label_set.json](../data/ai_rheum_label_set.json)

---

### 3) Retriever

Purpose: given chart text, retrieve top-k ontology documents to inject into the RAG prompt.

Where it lives:
- Retriever implementations + caching: [src/classes/retrievers.py](../src/classes/retrievers.py)

Behavior:
- Preferred: sentence-transformers embeddings (optionally FAISS) with cache under [data/retriever_cache/ai_rheum/](../data/retriever_cache/)
- Fallback: TF‑IDF cosine similarity (deterministic, no heavy deps)

Done criteria:
- [ ] create_retriever(corpus) returns a working retriever without external services
- [ ] retriever.retrieve(query) returns k documents with label/text/id
- [ ] Optional cache files appear under [data/retriever_cache/ai_rheum/](../data/retriever_cache/) on first run

---

### 4) RAG context builder

Purpose: format retrieved ontology docs into a bounded context string.

Where it lives:
- Context formatting and clipping: [src/rag_context.py](../src/rag_context.py)

Contract:
- build_rag_context(chart_text, retriever, config, top_k, max_chars) returns a string capped by max_chars

Done criteria:
- [ ] Returned context is non-empty for typical chart text
- [ ] Context is bounded (never exceeds the configured max_chars)

---

### 5) LLM interface (constrained output + caching)

Purpose: produce structured JSON predictions while enforcing the constrained label set.

Where it lives:
- LLM backend abstraction + caching + validation: [src/classes/llm_interface.py](../src/classes/llm_interface.py)

Enforcement:
- Output must be within allowed AI‑RHEUM IDs ∪ NONE
- Invalid/parse-failed outputs are coerced to NONE
- Calls are cached to [data/llm_cache.json](../data/llm_cache.json)

Done criteria:
- [ ] predict() returns JSON-like dict with predicted_label, rationale (top3 optional)
- [ ] Any out-of-set label is coerced to NONE
- [ ] Cache grows on first run, hits on reruns

---

### 6) Data generation (synthetic)

Purpose: produce the experiment dataset (synthetic charts) with gold labels drawn from the constrained label set ∪ NONE.

Where it lives:
- Synthetic dataset generator: [src/synthetic_data.py](../src/synthetic_data.py)

Artifact:
- [data/synthetic_charts.csv](../data/synthetic_charts.csv)

Contract (evaluation assumes at minimum):
- chart_text: string
- gold_label: string (must be in allowed ∪ NONE)

Done criteria:
- [ ] Dataset loads via pandas
- [ ] All gold_label values are in allowed ∪ NONE
- [ ] Enough rows to be meaningful for agreement (project-specific threshold)

---

### 7) Evaluation (mechanical scoring)

Purpose: run No‑RAG vs RAG and compute exact percent agreement.

Where it lives:
- Official evaluation runner: [src/eval_official.py](../src/eval_official.py)
- Entrypoint: [evaluate.py](../evaluate.py)

Determinism note (official runs):
- For official runs, set temperature=0 (or backend equivalent) where applicable; disk caching further enforces repeatability across reruns.

Outputs:
- Official: [results/results.json](../results/results.json), [results/predictions.csv](../results/predictions.csv), [results/summary.md](../results/summary.md)

Metric:
Let N be number of evaluated rows and I[·] indicator:

$$
\\text{Agreement} = \\frac{1}{N}\\sum_{i=1}^{N} I[\\hat{y}_i = y_i] \\times 100
$$

Done criteria:
- [ ] Writes results.json + predictions.csv + summary.md
- [ ] Enforces “predictions in allowed ∪ NONE” (coercion recorded)
- [ ] Filters or rejects rows whose gold_label is outside allowed ∪ NONE (explicit in results metadata)

---

### 8) Validators (mechanical checks)

Purpose: keep datasets and label sets consistent with constraints.

Where it lives:
- Dataset/label validators: [src/validators.py](../src/validators.py)

Done criteria:
- [ ] data/ai_rheum_label_set.json meets size and uniqueness constraints
- [ ] CSV gold labels validate against the AI‑RHEUM label set (allowed ∪ NONE)


1) Identifier sanity rule (do this first)
IDs in data/ai_rheum_label_set.json must exactly equal the `tco_id` field in data/ai_rheum_corpus.jsonl. If you see exclusions/coercions, check this first.

2) Smoke test: build (or load) the AI‑RHEUM corpus cache
PYTHONPATH=src python3 - <<'PY'
import json
from pathlib import Path

from onto_config import get_config
from tco_corpus import ensure_tco_corpus

labels = json.loads(Path("data/ai_rheum_label_set.json").read_text())["labels"]
cfg = get_config("ai_rheum")

corpus = ensure_tco_corpus(
    config=cfg,
    label_ids=labels,
    output_path="data/ai_rheum_corpus.jsonl",
    prefer_bioportal=True,
)
print("OK corpus records:", len(corpus))
print("Wrote/verified: data/ai_rheum_corpus.jsonl")
PY

3) Smoke test: retrieval works (and cache is ontology-specific)
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

4) Run official evaluation
Outputs:

results/results.json
results/predictions.csv
results/summary.md
Determinism note:

For official runs, set temperature=0 (or backend equivalent) where applicable; caching enforces repeatability on reruns.