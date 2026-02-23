#!/usr/bin/env python3
"""Small ablation sweep for retrieval context settings.

Sweeps top_k and max_context_chars and writes a compact CSV table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval_official import RunConfig, run_official_eval  # noqa: E402


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablation sweep over k and max_context_chars")
    parser.add_argument("--ontology-key", default="ai_rheum")
    parser.add_argument("--label-set", default="data/ai_rheum_label_set.json")
    parser.add_argument("--dataset", default="data/seed_cases_ai_rheum.csv")
    parser.add_argument("--corpus", default="data/ai_rheum_corpus.jsonl")
    parser.add_argument("--retriever-cache-dir", default="data/retriever_cache/ai_rheum")
    parser.add_argument("--results-dir", default="results/ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", default="1,2,3")
    parser.add_argument("--max-context-values", default="800,1200,1800")
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for k in _parse_int_list(args.k_values):
        for max_chars in _parse_int_list(args.max_context_values):
            combo_dir = results_dir / f"k{k}_c{max_chars}"
            cfg = RunConfig(
                seed=int(args.seed),
                top_k=int(k),
                max_context_chars=int(max_chars),
                ontology_key=str(args.ontology_key),
                label_set_path=Path(args.label_set),
                dataset_path=Path(args.dataset),
                corpus_path=Path(args.corpus),
                retriever_cache_dir=Path(args.retriever_cache_dir),
                embedding_model=str(args.embedding_model),
                results_dir=combo_dir,
            )
            out = run_official_eval(cfg)
            rows.append(
                {
                    "k": k,
                    "max_context_chars": max_chars,
                    "agreement_no_rag": out["metrics"]["agreement_no_rag"],
                    "agreement_rag": out["metrics"]["agreement_rag"],
                    "n_cases": out["metrics"]["n_cases"],
                    "embedding_model": out["run"].get("embedding_model", ""),
                    "results_json": str(combo_dir / "results.json"),
                }
            )

    table = pd.DataFrame(rows).sort_values(["k", "max_context_chars"]).reset_index(drop=True)
    table_path = results_dir / "ablation_table.csv"
    table.to_csv(table_path, index=False)

    summary = {
        "rows": len(table),
        "table": str(table_path),
        "k_values": _parse_int_list(args.k_values),
        "max_context_values": _parse_int_list(args.max_context_values),
    }
    (results_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote ablation table: {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
