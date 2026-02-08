"""Seeded evaluation: No-RAG vs RAG(TCO) on data/seed_cases.csv.

Outputs:
- results/results_seed.json
- results/predictions_seed.csv
- results/summary.md

Requirements enforced:
- Constrained label set from data/label_set.json
- Strict JSON parsing via LLMInterface backends (parse failures -> NONE)
- Invalid labels coerced to NONE (double-checked here)
- Disk caching via data/llm_cache.json
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm_interface import LLMInterface
from onto_config import get_config
from rag_context import build_rag_context
from retrievers import create_retriever
from tco_corpus import ensure_tco_corpus


DEFAULT_SEED = 42
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class RunConfig:
    seed: int = DEFAULT_SEED
    top_k: int = DEFAULT_TOP_K
    max_context_chars: int = 1800
    label_set_path: Path = Path("data/label_set.json")
    seed_cases_path: Path = Path("data/seed_cases.csv")
    llm_cache_path: Path = Path("data/llm_cache.json")
    corpus_path: Path = Path("data/tco_corpus.jsonl")
    retriever_cache_dir: Path = Path("data/retriever_cache")
    results_dir: Path = Path("results")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _load_label_set(path: Path) -> tuple[list[str], str]:
    data = json.loads(path.read_text())
    labels = list(data.get("labels") or [])
    none_label = str(data.get("none_label") or "NONE")
    return labels, none_label


def _exact_agreement(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return 100.0 * correct / len(gold)


def _coerce(pred: str, allowed: set[str], none_label: str) -> tuple[str, bool]:
    if pred in allowed:
        return pred, False
    return none_label, True


def run_seed_eval(cfg: RunConfig) -> dict[str, Any]:
    # Reproducibility (mostly affects any stochastic backends; also future-proof).
    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))
    np.random.seed(cfg.seed)

    labels, none_label = _load_label_set(cfg.label_set_path)
    allowed = set(labels) | {none_label}

    config = get_config("tco")

    # Build/load normalized corpus restricted to label_set labels.
    corpus = ensure_tco_corpus(
        config=config,
        label_ids=labels,
        output_path=cfg.corpus_path,
        prefer_bioportal=True,
    )

    # Retriever with caching.
    retriever = create_retriever(
        corpus,
        top_k=cfg.top_k,
        prefer_embeddings=True,
        cache_dir=str(cfg.retriever_cache_dir),
        model_name="all-MiniLM-L6-v2",
    )

    # LLM with disk cache.
    llm = LLMInterface(labels, cache_file=str(cfg.llm_cache_path), config=config)

    df = pd.read_csv(cfg.seed_cases_path)
    if "case_id" not in df.columns:
        raise ValueError("seed_cases.csv must include case_id")

    preds_no_rag: list[str] = []
    preds_rag: list[str] = []
    coerced_no_rag = 0
    coerced_rag = 0

    for _, row in df.iterrows():
        text = str(row["chart_text"])

        # No-RAG
        out0 = llm.predict(text, rag_context=None)
        p0, did0 = _coerce(str(out0.get("predicted_label", none_label)), allowed, none_label)
        if did0:
            coerced_no_rag += 1
        preds_no_rag.append(p0)

        # RAG
        ctx = build_rag_context(text, retriever, config, top_k=cfg.top_k, max_chars=cfg.max_context_chars)
        out1 = llm.predict(text, rag_context=ctx)
        p1, did1 = _coerce(str(out1.get("predicted_label", none_label)), allowed, none_label)
        if did1:
            coerced_rag += 1
        preds_rag.append(p1)

    gold = [str(x) for x in df["gold_label"].tolist()]
    agreement_no_rag = _exact_agreement(gold, preds_no_rag)
    agreement_rag = _exact_agreement(gold, preds_rag)

    # Save predictions CSV (exact requested columns).
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(
        {
            "case_id": df["case_id"].astype(str),
            "gold_label": gold,
            "pred_no_rag": preds_no_rag,
            "pred_rag": preds_rag,
        }
    )
    pred_path = cfg.results_dir / "predictions_seed.csv"
    pred_df.to_csv(pred_path, index=False)

    # Results JSON
    commit = _git_commit_hash()

    model_name = llm.backend
    if llm.backend == "gemini":
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    elif llm.backend == "huggingface":
        model_name = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    elif llm.backend == "openai":
        model_name = "gpt-3.5-turbo"

    results = {
        "run": {
            "timestamp": _now_iso(),
            "git_commit": commit,
            "model_backend": llm.backend,
            "model_name": model_name,
            "k": cfg.top_k,
            "seed": cfg.seed,
            "n_cases": int(len(df)),
            "label_set_path": str(cfg.label_set_path),
            "seed_cases_path": str(cfg.seed_cases_path),
            "llm_cache_path": str(cfg.llm_cache_path),
            "corpus_path": str(cfg.corpus_path),
        },
        "metrics": {
            "agreement_no_rag": float(agreement_no_rag),
            "agreement_rag": float(agreement_rag),
        },
        "coercions": {
            "coerced_invalid_labels_no_rag": int(coerced_no_rag),
            "coerced_invalid_labels_rag": int(coerced_rag),
        },
    }

    results_path = cfg.results_dir / "results_seed.json"
    results_path.write_text(json.dumps(results, indent=2))

    # Summary markdown (short table + 5–10 examples)
    summary_path = cfg.results_dir / "summary.md"
    summary_path.write_text(_render_summary(df, pred_df, results, none_label))

    return results


def _render_summary(df_seed: pd.DataFrame, pred_df: pd.DataFrame, results: dict[str, Any], none_label: str) -> str:
    n = int(results["run"]["n_cases"])
    a0 = float(results["metrics"]["agreement_no_rag"])
    a1 = float(results["metrics"]["agreement_rag"])
    backend = str(results["run"]["model_backend"])
    model_name = str(results["run"].get("model_name") or backend)
    k = int(results["run"]["k"])

    merged = df_seed.merge(pred_df, on=["case_id", "gold_label"], how="left")
    merged["ok_no_rag"] = merged["gold_label"] == merged["pred_no_rag"]
    merged["ok_rag"] = merged["gold_label"] == merged["pred_rag"]

    # pick a few examples: 3 successes, 3 failures (rag), and fill up to 10 total.
    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def add_rows(mask, limit: int):
        nonlocal examples
        for _, r in merged[mask].head(limit).iterrows():
            cid = str(r["case_id"])
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            examples.append(
                {
                    "case_id": cid,
                    "register": r.get("register", ""),
                    "gold": r["gold_label"],
                    "pred_no_rag": r["pred_no_rag"],
                    "pred_rag": r["pred_rag"],
                }
            )

    add_rows(merged["ok_rag"] == True, 3)
    add_rows(merged["ok_rag"] == False, 3)
    # if still short, add some that are NONE gold (distractors)
    if len(examples) < 8:
        add_rows(merged["gold_label"] == none_label, 2)

    examples = examples[:10]

    lines: list[str] = []
    lines.append("# Seed Evaluation Summary")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Condition | Exact agreement | N | k | Backend |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append(f"| No-RAG | {a0:.1f}% | {n} | {k} | {backend} |")
    lines.append(f"| RAG(TCO) | {a1:.1f}% | {n} | {k} | {backend} |")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    lines.append(f"- Seed: {results['run']['seed']}")
    lines.append(f"- Label set: {results['run']['label_set_path']}")
    lines.append(f"- Cache: {results['run']['llm_cache_path']}")
    lines.append(f"- Model: {model_name}")
    if results["run"].get("git_commit"):
        lines.append(f"- Git commit: {results['run']['git_commit']}")
    lines.append("")
    lines.append("## Example Cases (5–10)")
    lines.append("")
    for ex in examples:
        lines.append(
            f"- {ex['case_id']} ({ex['register']}): gold={ex['gold']} | no-rag={ex['pred_no_rag']} | rag={ex['pred_rag']}"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    cfg = RunConfig()
    run_seed_eval(cfg)
    print("✓ Wrote results/results_seed.json")
    print("✓ Wrote results/predictions_seed.csv")
    print("✓ Wrote results/summary.md")


if __name__ == "__main__":
    main()
