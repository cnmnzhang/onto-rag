#!/usr/bin/env python3
"""src/retriever.py

Embedding-based retriever using sentence-transformers.
Supports two models: all-MiniLM-L6-v2 and BAAI/bge-small-en-v1.5.
Caches embeddings to disk to avoid recomputing.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
    # Allow full names too
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query vector and matrix of doc vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-10
    b_normed = b / b_norms
    return b_normed @ a_norm


def _corpus_hash(corpus: list[dict]) -> str:
    """Stable hash of corpus ids to detect changes."""
    ids = [r.get("id", "") for r in corpus]
    return hashlib.md5(json.dumps(ids, sort_keys=True).encode()).hexdigest()[:12]


class Retriever:
    """Retrieves top-k corpus records by cosine similarity to query embedding."""

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        cache_dir: str | None = "data/retriever_cache",
        field: str = "text",
    ) -> None:
        # self.corpus = corpus

        # Filter to only finding and diagnosis chunks — domain chunks are too generic
        self.corpus = [
            r for r in corpus
            if r.get("chunk_type") != "domain_chunk"
        ]

        self.model_name = SUPPORTED_MODELS.get(model_name, model_name)
        self.top_k = top_k
        self.field = field
        self._model = None
        self._embeddings: np.ndarray | None = None
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._build_index()

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _cache_path(self) -> Path | None:
        if self._cache_dir is None:
            return None
        safe_name = self.model_name.replace("/", "_").replace("-", "_")
        corpus_hash = _corpus_hash(self.corpus)
        fname = f"embeddings_{safe_name}_{corpus_hash}.pkl"
        return self._cache_dir / fname

    def _build_index(self) -> None:
        cache_path = self._cache_path()

        # Try loading from cache
        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as f:
                self._embeddings = pickle.load(f)
            print(f"  [Retriever] Loaded embeddings from cache: {cache_path.name}")
            return

        # Build embeddings
        texts = [str(doc.get(self.field, "")) for doc in self.corpus]
        print(f"  [Retriever] Building embeddings for {len(texts)} docs with {self.model_name}...")
        model = self._get_model()
        self._embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Save to cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self._embeddings, f)
            print(f"  [Retriever] Saved embeddings to cache: {cache_path.name}")

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return top-k most similar corpus records for query."""
        k = top_k or self.top_k
        model = self._get_model()
        q_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        assert self._embeddings is not None
        scores = _cosine_similarity(q_emb, self._embeddings)
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            doc = dict(self.corpus[idx])
            doc["retrieval_score"] = float(scores[idx])
            results.append(doc)
        return results


def create_retriever(
    corpus: list[dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5,
    cache_dir: str = "data/retriever_cache",
) -> Retriever:
    """Factory function matching existing codebase interface."""
    return Retriever(corpus, model_name=model_name, top_k=top_k, cache_dir=cache_dir)
