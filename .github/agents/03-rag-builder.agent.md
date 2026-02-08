---
name: 03-rag-builder
description: Implement/refactor TCO ingestion, corpus building, and retrieval for RAG(TCO) using Hugging Face embeddings with a TF-IDF fallback.
tools: ["read", "search", "edit", "execute"]
model: ["GPT-5.2 (copilot)", "Claude Sonnet 4.5 (copilot)"]
handoffs:
  - label: Run evaluation + report results
    agent: 04-evaluator
    prompt: Evaluate no-RAG vs RAG using the current pipeline; produce percent agreement and save artifacts to results/.
    send: false
---

## Role
You implement the RAG substrate: ontology → corpus → retriever → context builder. You do not design the dataset schema (consume docs/SYNTHETIC_DATA_SCHEMA.md and data/label_set.json).

## Required inputs
- docs/PROPOSAL.md
- docs/SYNTHETIC_DATA_SCHEMA.md
- data/label_set.json
- existing src modules (llm_interface.py, onto_config.py, retrievers.py, rag_exp.py)

## Implementation targets (prefer extending current code)
1) Ontology ingestion:
   - BioPortal API path (BIOPORTAL_API_KEY)
   - graceful fallback if offline: use existing data/tco_corpus.jsonl if present
2) Corpus builder:
   - output: data/tco_corpus.jsonl with stable fields:
     - tco_id, label, synonyms, text
3) Retrieval:
   - primary: sentence-transformers embeddings + FAISS (if available)
   - fallback: scikit-learn TF-IDF cosine
4) Context builder:
   - takes chart_text → returns top-k chunks formatted for prompting

## Non-negotiables
- Deterministic builds (seed where applicable).
- Do not introduce hard dependencies that break Colab.
- Keep artifacts in data/.
- Add caching for embeddings/index if it materially improves runtime.

## Done criteria
- Running a single script or notebook cell builds corpus + index end-to-end.
- Retrieval returns coherent chunks for a chart-style query.
- RAG context builder returns bounded-length context (avoid giant prompts).