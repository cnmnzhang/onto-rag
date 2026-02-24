"""Shared types for official evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.paths import (
    AI_RHEUM_CORPUS_PATH,
    AI_RHEUM_LABEL_SET_PATH,
    AI_RHEUM_RETRIEVER_CACHE_DIR,
    AI_RHEUM_SEED_CASES_PATH,
    LLM_CACHE_PATH,
    RESULTS_DIR,
)


DEFAULT_SEED = 42
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class RunConfig:
    seed: int = DEFAULT_SEED
    top_k: int = DEFAULT_TOP_K
    max_context_chars: int = 1800
    prefer_embeddings: bool = True
    ontology_key: str = "ai_rheum"
    label_set_path: Path = AI_RHEUM_LABEL_SET_PATH
    dataset_path: Path = AI_RHEUM_SEED_CASES_PATH
    llm_cache_path: Path = LLM_CACHE_PATH
    corpus_path: Path = AI_RHEUM_CORPUS_PATH
    retriever_cache_dir: Path = AI_RHEUM_RETRIEVER_CACHE_DIR
    embedding_model: str = "all-MiniLM-L6-v2"
    results_dir: Path = RESULTS_DIR

