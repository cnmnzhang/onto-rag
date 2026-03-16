"""Minimal end-to-end runner for the RAG substrate.

This script builds/loads:
- `data/tco_corpus.jsonl` (from BioPortal if possible, else cached)
- retrieval index/embeddings (cached in `data/retriever_cache/` when possible)

Then demonstrates retrieval + bounded `build_rag_context(text) -> context`.

Run:
- From repo root: `python run.py`
- Or directly: `PYTHONPATH=src python -m rag_exp`
"""

from __future__ import annotations

import json
from pathlib import Path

from classes.onto_config import get_config
from classes.retrievers import create_retriever
from rag_context import build_rag_context as _build_rag_context
from classes.corpus import load_or_build_corpus


_CONFIG = get_config("ai_rheum")
_RETRIEVER = None


def _load_label_ids(label_set_path: str | Path = "data/label_set.json") -> list[str]:
    data = json.loads(Path(label_set_path).read_text())
    return list(data.get("labels") or [])


def _init_retriever() -> None:
    global _RETRIEVER
    if _RETRIEVER is not None:
        return

    label_ids = _load_label_ids()
    corpus = load_or_build_corpus(config=_CONFIG, label_ids=label_ids, output_path=f"data/{_CONFIG.corpus_filename}")

    # Cache embeddings/index under data/ to speed up reruns.
    _RETRIEVER = create_retriever(
        corpus,
        top_k=3,
        prefer_embeddings=True,
        cache_dir="data/retriever_cache",
        model_name="all-MiniLM-L6-v2",
    )


def build_rag_context(text: str) -> str:
    """Public helper: chart_text -> bounded RAG context."""

    _init_retriever()
    assert _RETRIEVER is not None
    return _build_rag_context(text, _RETRIEVER, _CONFIG, top_k=3, max_chars=1800)


def main() -> None:
    _init_retriever()

    sample_text = """46-year-old F with 3-month anterior neck swelling.
Symptoms: mild dysphagia and intermittent hoarseness.
Exam: firm thyroid nodule.
Ultrasound: hypoechoic nodule with microcalcifications.
FNA cytology: suspicious for papillary carcinoma.
""".strip()
    print("Using built-in sample text.")

    context = build_rag_context(sample_text)
    print("\n--- RAG CONTEXT (bounded) ---")
    print(context)
    print("\nContext length (chars):", len(context))


if __name__ == "__main__":
    main()
