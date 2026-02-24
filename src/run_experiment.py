#!/usr/bin/env python3
"""src/run_experiment.py

Main experiment runner.

Runs 4 conditions for every test case:
  A) No RAG   | Embedding model 1 (MiniLM)
  B) RAG      | Embedding model 1 (MiniLM)
  C) No RAG   | Embedding model 2 (BGE)
  D) RAG      | Embedding model 2 (BGE)

Since conditions A and C produce identical LLM output (no RAG context),
we run the LLM twice per case: once with no RAG, once with RAG.
Then do this for both embedding models.

Outputs:
  results/responses_minilm.csv   -- full narrative responses, model 1
  results/responses_bge.csv      -- full narrative responses, model 2
  results/all_responses.csv      -- combined for blinded review generation
  results/rag_contexts.csv       -- what context was retrieved for each case

Usage:
  python src/run_experiment.py
  python src/run_experiment.py --models minilm          # one model only
  python src/run_experiment.py --cases data/test_cases.csv
  python src/run_experiment.py --top-k 5
  python src/run_experiment.py --dry-run                # test pipeline without API
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent))

from corpus_builder import load_corpus, build_rich_corpus, save_corpus
from retriever import Retriever
from rag_context import build_rag_context
from llm_client import LLMClient

MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_corpus(ttl_path: Path, corpus_path: Path) -> list[dict]:
    if corpus_path.exists():
        print(f"[Corpus] Loading existing corpus: {corpus_path}")
        return load_corpus(corpus_path)
    print(f"[Corpus] Building corpus from {ttl_path} ...")
    if not ttl_path.exists():
        raise FileNotFoundError(
            f"TTL file not found: {ttl_path}\n"
            f"Copy AI-RHEUM.ttl to {ttl_path}"
        )
    records = build_rich_corpus(ttl_path)
    save_corpus(records, corpus_path)
    print(f"[Corpus] Saved {len(records)} records to {corpus_path}")
    return records


def _build_retriever(corpus: list[dict], model_key: str, top_k: int) -> Retriever:
    model_name = MODELS[model_key]
    print(f"[Retriever] Initialising {model_key} ({model_name}) ...")
    return Retriever(
        corpus,
        model_name=model_name,
        top_k=top_k,
        cache_dir=f"data/retriever_cache/{model_key}",
    )


def run_experiment(
    cases_path: Path,
    ttl_path: Path,
    corpus_path: Path,
    results_dir: Path,
    model_keys: list[str],
    top_k: int,
    llm_cache_path: Path,
    dry_run: bool = False,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load test cases
    print(f"\n[Data] Loading test cases from {cases_path}")
    df = pd.read_csv(cases_path)
    print(f"[Data] {len(df)} cases loaded")

    # Build/load corpus
    corpus = _ensure_corpus(ttl_path, corpus_path)
    print(f"[Corpus] {len(corpus)} records available")

    # Build retrievers for each model
    retrievers: dict[str, Retriever] = {}
    for mk in model_keys:
        retrievers[mk] = _build_retriever(corpus, mk, top_k)

    # Init LLM
    backend = None
    if dry_run:
        backend = "dry_run"
    llm = LLMClient(cache_path=llm_cache_path, backend=backend)

    # ------- Run experiment -------
    # We collect rows for each model separately, then combine

    context_rows = []  # What RAG retrieved
    all_response_rows = []  # Combined for blinded review

    for model_key in model_keys:
        model_name = MODELS[model_key]
        retriever = retrievers[model_key]
        response_rows = []

        print(f"\n{'='*60}")
        print(f"Model: {model_key} ({model_name})")
        print(f"{'='*60}")

        for _, case in df.iterrows():
            case_id = str(case["case_id"])
            chart_text = str(case["chart_text"])
            gold_label = str(case.get("gold_label", "UNKNOWN"))
            category = str(case.get("category", ""))
            notes = str(case.get("notes", ""))

            print(f"\n  Case {case_id} | {notes[:50]}")

            # --- Retrieve context ---
            retrieved_docs = retriever.retrieve(chart_text, top_k=top_k)
            rag_ctx = build_rag_context(retrieved_docs)

            # Save retrieved context
            context_rows.append({
                "case_id": case_id,
                "model_key": model_key,
                "model_name": model_name,
                "rag_context": rag_ctx,
                "top_k": top_k,
                "retrieved_chunks": json.dumps(
                    [{"label": d.get("label"), "score": round(d.get("retrieval_score", 0), 4)}
                     for d in retrieved_docs],
                    ensure_ascii=False,
                ),
            })

            # --- Condition: No RAG ---
            print(f"    Generating No-RAG response...", end="", flush=True)
            resp_no_rag = llm.generate(chart_text, rag_context=None)
            print(" done")

            # --- Condition: RAG ---
            print(f"    Generating RAG response...", end="", flush=True)
            resp_rag = llm.generate(chart_text, rag_context=rag_ctx)
            print(" done")

            row = {
                "case_id": case_id,
                "model_key": model_key,
                "model_name": model_name,
                "gold_label": gold_label,
                "category": category,
                "notes": notes,
                "chart_text": chart_text,
                "response_no_rag": resp_no_rag,
                "response_rag": resp_rag,
                "rag_context_chars": len(rag_ctx),
            }
            response_rows.append(row)
            all_response_rows.append(row)

        # Save per-model CSV
        model_df = pd.DataFrame(response_rows)
        model_path = results_dir / f"responses_{model_key}.csv"
        model_df.to_csv(model_path, index=False)
        print(f"\n[Output] Saved {model_path}")

    # Save RAG contexts
    ctx_df = pd.DataFrame(context_rows)
    ctx_path = results_dir / "rag_contexts.csv"
    ctx_df.to_csv(ctx_path, index=False)
    print(f"[Output] Saved {ctx_path}")

    # Save combined responses
    all_df = pd.DataFrame(all_response_rows)
    all_path = results_dir / "all_responses.csv"
    all_df.to_csv(all_path, index=False)
    print(f"[Output] Saved {all_path}")

    # Save run metadata
    meta = {
        "timestamp": _now(),
        "cases_path": str(cases_path),
        "corpus_path": str(corpus_path),
        "corpus_size": len(corpus),
        "n_cases": len(df),
        "model_keys": model_keys,
        "model_names": [MODELS[k] for k in model_keys],
        "top_k": top_k,
        "llm_backend": llm.backend,
        "llm_model": llm.model,
        "dry_run": dry_run,
    }
    meta_path = results_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[Output] Saved {meta_path}")

    print(f"\n{'='*60}")
    print(f"Experiment complete.")
    print(f"  Cases: {len(df)}")
    print(f"  Models: {model_keys}")
    print(f"  Results: {results_dir}/")
    print(f"  Next step: python src/generate_review_sheet.py")


def main() -> None:
    p = argparse.ArgumentParser(description="Run RAG vs No-RAG rheumatology experiment")
    p.add_argument("--cases", default="data/test_cases.csv", help="Test cases CSV")
    p.add_argument("--ttl", default="data/AI-RHEUM.ttl", help="AI-RHEUM TTL ontology file")
    p.add_argument("--corpus", default="data/ai_rheum_corpus_rich.jsonl", help="Corpus JSONL (built from TTL)")
    p.add_argument("--results-dir", default="results", help="Output directory")
    p.add_argument("--models", default="minilm,bge", help="Comma-separated model keys: minilm,bge")
    p.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    p.add_argument("--llm-cache", default="data/llm_cache.json", help="LLM response cache")
    p.add_argument("--dry-run", action="store_true", help="Use dry-run LLM (no API key needed)")
    args = p.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip() in MODELS]
    if not model_keys:
        print(f"Invalid --models. Choose from: {list(MODELS.keys())}")
        sys.exit(1)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    run_experiment(
        cases_path=Path(args.cases),
        ttl_path=Path(args.ttl),
        corpus_path=Path(args.corpus),
        results_dir=Path(args.results_dir),
        model_keys=model_keys,
        top_k=args.top_k,
        llm_cache_path=Path(args.llm_cache),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
