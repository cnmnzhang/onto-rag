"""Centralized project-relative path constants."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path("data")
SRC_DIR = Path("src")
RESULTS_DIR = Path("results")

AI_RHEUM_LABEL_SET_PATH = DATA_DIR / "ai_rheum_label_set.json"
AI_RHEUM_SEED_CASES_PATH = DATA_DIR / "seed_cases_ai_rheum.csv"
AI_RHEUM_CORPUS_PATH = DATA_DIR / "ai_rheum_corpus.jsonl"
AI_RHEUM_RETRIEVER_CACHE_DIR = DATA_DIR / "retriever_cache" / "ai_rheum"

LLM_CACHE_PATH = DATA_DIR / "llm_cache.json"

RESULTS_JSON_PATH = RESULTS_DIR / "results.json"
PREDICTIONS_CSV_PATH = RESULTS_DIR / "predictions.csv"
SUMMARY_MD_PATH = RESULTS_DIR / "summary.md"

FETCH_LABEL_URIS_SCRIPT_PATH = SRC_DIR / "fetch_label_uris.py"
BUILD_CORPUS_SCRIPT_PATH = SRC_DIR / "build_corpus.py"

