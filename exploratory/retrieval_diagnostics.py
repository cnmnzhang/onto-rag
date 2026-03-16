#!/usr/bin/env python3
"""Quick retrieval diagnostics for top-k label hits and similarity scores.

Usage:
  EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 python3 scripts/retrieval_diagnostics.py \
    --query "joint pain with morning stiffness" \
    --ontology-key ai_rheum \
    --label-set data/ai_rheum_label_set.json \
    --corpus data/ai_rheum_corpus.jsonl \
    --cache-dir data/retriever_cache/ai_rheum
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.bootstrap import ensure_repo_on_sys_path  # noqa: E402

ensure_repo_on_sys_path()

from classes.onto_config import get_config  # noqa: E402
from classes.retrievers import create_retriever  # noqa: E402
from classes.corpus import load_or_build_corpus, load_corpus  # noqa: E402


def _sanitize_model_name_for_path(model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model_name).strip())
    return safe.strip("-") or "default"


def _load_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels") or []
    return [str(x) for x in labels]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect retrieved ontology docs for a single query")
    parser.add_argument("--query", required=True)
    parser.add_argument("--ontology-key", default="ai_rheum")
    parser.add_argument("--label-set", default="data/ai_rheum_label_set.json")
    parser.add_argument("--corpus", default="data/ai_rheum_corpus.jsonl")
    parser.add_argument("--cache-dir", default="data/retriever_cache/ai_rheum")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--prefer-embeddings", action="store_true", default=True)
    args = parser.parse_args()

    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"
    model_cache_dir = Path(args.cache_dir) / _sanitize_model_name_for_path(embedding_model)

    label_set_path = Path(args.label_set)
    corpus_path = Path(args.corpus)
    labels = _load_labels(label_set_path)

    if corpus_path.exists():
        corpus = load_corpus(str(corpus_path))
    else:
        config = get_config(args.ontology_key)
        corpus = load_or_build_corpus(
            config=config,
            label_ids=labels,
            output_path=corpus_path,
            prefer_bioportal=True,
        )

    retriever = create_retriever(
        corpus,
        top_k=int(args.k),
        prefer_embeddings=bool(args.prefer_embeddings),
        cache_dir=str(model_cache_dir),
        model_name=embedding_model,
    )

    print(f"Embedding model: {embedding_model}")
    print(f"Retriever cache: {model_cache_dir}")
    print(f"Query: {args.query}")
    print()

    hits = retriever.retrieve(str(args.query))
    for i, hit in enumerate(hits, 1):
        label = hit.get("label") or ""
        doc_id = hit.get("tco_id") or hit.get("doc_id") or hit.get("id") or ""
        score = hit.get("similarity_score")
        if score is None:
            score_text = "NA"
        else:
            score_text = f"{float(score):.4f}"
        print(f"{i}. score={score_text} id={doc_id}")
        print(f"   label={label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
