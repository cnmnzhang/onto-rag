"""Official evaluation runner (mechanical scoring).

Reads:
- data/synthetic_charts.csv
- data/label_set.json

Writes:
- results/results.json
- results/predictions.csv
- results/summary.md

Rules:
- Primary metric: exact percent agreement (predicted_label == gold_label)
- Predictions must be in allowed label set or NONE; otherwise coerce to NONE
- Keep LLM calls cached via data/llm_cache.json (LLMInterface)

This is intentionally minimal and deterministic.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
from corpus import ensure_corpus


DEFAULT_SEED = 42
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class RunConfig:
    seed: int = DEFAULT_SEED
    top_k: int = DEFAULT_TOP_K
    max_context_chars: int = 1800
    ontology_key: str = "tco"
    label_set_path: Path = Path("data/label_set.json")
    dataset_path: Path = Path("data/synthetic_charts.csv")
    llm_cache_path: Path = Path("data/llm_cache.json")
    corpus_path: Path = Path("data/tco_corpus.jsonl")
    retriever_cache_dir: Path = Path("data/retriever_cache")
    embedding_model: str = "all-MiniLM-L6-v2"
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
    print(f"⚠️  Invalid prediction label coerced to {none_label}: {pred[:80]}")
    return none_label, True


def _sanitize_model_name_for_path(model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model_name).strip())
    return safe.strip("-") or "default"


def _model_name_for_backend(llm: LLMInterface) -> str:
    if llm.backend == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    if llm.backend == "huggingface":
        return os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    if llm.backend == "openai":
        return "gpt-3.5-turbo"
    return "dry_run"


def run_official_eval(cfg: RunConfig) -> dict[str, Any]:
    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))
    np.random.seed(cfg.seed)

    labels, none_label = _load_label_set(cfg.label_set_path)
    allowed = set(labels) | {none_label}

    config = get_config(cfg.ontology_key)
    embedding_model = os.getenv("EMBEDDING_MODEL", cfg.embedding_model).strip() or "all-MiniLM-L6-v2"

    # Corpus restricted to the allowed TCO labels.
    corpus = ensure_corpus(
        config=config,
        label_ids=labels,
        output_path=cfg.corpus_path,
        prefer_bioportal=True,
    )

    model_cache_dir = cfg.retriever_cache_dir / _sanitize_model_name_for_path(embedding_model)
    retriever = create_retriever(
        corpus,
        top_k=cfg.top_k,
        prefer_embeddings=True,
        cache_dir=str(model_cache_dir),
        model_name=embedding_model,
    )

    llm = LLMInterface(labels, cache_file=str(cfg.llm_cache_path), config=config)

    df = pd.read_csv(cfg.dataset_path)
    required = {"chart_text", "gold_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    # Create a stable case_id.
    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].astype(str)
    elif "chart_id" in df.columns:
        df["case_id"] = df["chart_id"].apply(lambda x: f"chart-{int(x)}")
    else:
        df["case_id"] = [f"case-{i:04d}" for i in range(len(df))]

    # Filter to compatible gold labels.
    is_valid_gold = df["gold_label"].astype(str).isin(allowed)
    excluded_df = df.loc[~is_valid_gold, ["case_id", "gold_label"]]
    df = df.loc[is_valid_gold].copy()
    if df.empty:
        raise RuntimeError(
            "No rows have gold_label in the allowed set ∪ NONE. "
            "Update data/label_set.json or the dataset gold labels."
        )

    preds_no_rag: list[str] = []
    preds_rag: list[str] = []
    ddx_no_rag: list[str] = []
    ddx_rag: list[str] = []
    evidence_no_rag: list[str] = []
    evidence_rag: list[str] = []
    coerced_no_rag = 0
    coerced_rag = 0

    for _, row in df.iterrows():
        text = str(row["chart_text"])

        out0 = llm.predict(text, rag_context=None)
        p0, did0 = _coerce(str(out0.get("predicted_label", none_label)), allowed, none_label)
        coerced_no_rag += int(did0)
        preds_no_rag.append(p0)
        ddx_no_rag.append(json.dumps(out0.get("ddx_top3") or [], ensure_ascii=False))
        evidence_no_rag.append(json.dumps(out0.get("evidence") or [], ensure_ascii=False))

        ctx = build_rag_context(text, retriever, config, top_k=cfg.top_k, max_chars=cfg.max_context_chars)
        out1 = llm.predict(text, rag_context=ctx)
        p1, did1 = _coerce(str(out1.get("predicted_label", none_label)), allowed, none_label)
        coerced_rag += int(did1)
        preds_rag.append(p1)
        ddx_rag.append(json.dumps(out1.get("ddx_top3") or [], ensure_ascii=False))
        evidence_rag.append(json.dumps(out1.get("evidence") or [], ensure_ascii=False))

    gold = [str(x) for x in df["gold_label"].tolist()]
    agreement_no_rag = _exact_agreement(gold, preds_no_rag)
    agreement_rag = _exact_agreement(gold, preds_rag)

    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame(
        {
            "case_id": df["case_id"].astype(str),
            "gold_label": gold,
            "pred_no_rag": preds_no_rag,
            "pred_rag": preds_rag,
            "ddx_no_rag": ddx_no_rag,
            "ddx_rag": ddx_rag,
            "evidence_no_rag": evidence_no_rag,
            "evidence_rag": evidence_rag,
        }
    )
    pred_path = cfg.results_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    results = {
        "run": {
            "timestamp": _now_iso(),
            "git_commit": _git_commit_hash(),
            "model_backend": llm.backend,
            "model_name": _model_name_for_backend(llm),
            "k": cfg.top_k,
            "seed": cfg.seed,
            "n_cases": int(len(df)),
            "ontology_key": cfg.ontology_key,
            "dataset_path": str(cfg.dataset_path),
            "label_set_path": str(cfg.label_set_path),
            "corpus_path": str(cfg.corpus_path),
            "retriever_cache_dir": str(cfg.retriever_cache_dir),
            "retriever_model_cache_dir": str(model_cache_dir),
            "embedding_model": embedding_model,
        },
        "metrics": {
            "agreement_no_rag": float(agreement_no_rag),
            "agreement_rag": float(agreement_rag),
            "n_cases": int(len(df)),
        },
        "coercions": {
            "coerced_invalid_labels_no_rag": int(coerced_no_rag),
            "coerced_invalid_labels_rag": int(coerced_rag),
        },
    }

    if len(excluded_df) > 0:
        # Keep a small, stable summary of exclusions.
        results["run"]["excluded_gold_rows"] = int(len(excluded_df))
        results["run"]["excluded_gold_label_counts"] = (
            excluded_df["gold_label"].astype(str).value_counts().to_dict()
        )

    results_path = cfg.results_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))

    summary_path = cfg.results_dir / "summary.md"
    summary_path.write_text(_render_summary(pred_df, results))

    return results


def _render_summary(pred_df: pd.DataFrame, results: dict[str, Any]) -> str:
    n = int(results["run"]["n_cases"])
    a0 = float(results["metrics"]["agreement_no_rag"])
    a1 = float(results["metrics"]["agreement_rag"])
    backend = str(results["run"]["model_backend"])
    model_name = str(results["run"].get("model_name") or backend)
    embedding_model = str(results["run"].get("embedding_model") or "")
    k = int(results["run"]["k"])
    ontology_key = str(results["run"].get("ontology_key") or "")
    dataset_path = str(results["run"].get("dataset_path") or "")

    pred_df = pred_df.copy()
    pred_df["ok_no_rag"] = pred_df["gold_label"] == pred_df["pred_no_rag"]
    pred_df["ok_rag"] = pred_df["gold_label"] == pred_df["pred_rag"]

    examples: list[dict[str, str]] = []

    def add(mask, limit: int):
        for _, r in pred_df.loc[mask].head(limit).iterrows():
            examples.append(
                {
                    "case_id": str(r["case_id"]),
                    "gold": str(r["gold_label"]),
                    "no_rag": str(r["pred_no_rag"]),
                    "rag": str(r["pred_rag"]),
                }
            )

    # Prefer a mix of correct/incorrect under RAG.
    add(pred_df["ok_rag"] == True, 4)
    add(pred_df["ok_rag"] == False, 4)

    # Deduplicate while keeping order.
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for ex in examples:
        if ex["case_id"] in seen:
            continue
        seen.add(ex["case_id"])
        deduped.append(ex)

    deduped = deduped[:10]

    lines: list[str] = []
    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Condition | Exact agreement | N | k | Backend |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append(f"| No-RAG | {a0:.1f}% | {n} | {k} | {backend} |")
    lines.append(f"| RAG({ontology_key or 'ontology'}) | {a1:.1f}% | {n} | {k} | {backend} |")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    if ontology_key:
        lines.append(f"- Ontology: {ontology_key}")
    if dataset_path:
        lines.append(f"- Dataset: {dataset_path}")
    lines.append(f"- Seed: {results['run']['seed']}")
    lines.append(f"- Model: {model_name}")
    if embedding_model:
        lines.append(f"- Embedding model: {embedding_model}")
    if results["run"].get("git_commit"):
        lines.append(f"- Git commit: {results['run']['git_commit']}")
    if results["run"].get("excluded_gold_rows"):
        lines.append(f"- Excluded rows (gold label outside allowed set): {results['run']['excluded_gold_rows']}")
    lines.append("")
    lines.append("## Example Cases (5–10)")
    lines.append("")
    for ex in deduped:
        lines.append(
            f"- {ex['case_id']}: gold={ex['gold']} | no-rag={ex['no_rag']} | rag={ex['rag']}"
        )
    if "ddx_rag" in pred_df.columns:
        lines.append("")
        lines.append("## Differential Diagnosis Snippets (RAG)")
        lines.append("")
        shown = 0
        for _, row in pred_df.iterrows():
            if shown >= 5:
                break
            ddx_text = str(row.get("ddx_rag", "")).strip()
            if not ddx_text or ddx_text == "[]":
                continue
            lines.append(f"- {row['case_id']}: {ddx_text}")
            shown += 1
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    # Convenience: load .env if python-dotenv is available.
    try:  # pragma: no cover
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Official evaluation runner (exact agreement)")
    p.add_argument("--ontology-key", default="tco", help="Key for onto_config.get_config (e.g., ai_rheum)")
    p.add_argument("--label-set", default="data/label_set.json", help="Path to label set JSON")
    p.add_argument("--dataset", default="data/synthetic_charts.csv", help="Path to dataset CSV")
    p.add_argument("--corpus", default="data/tco_corpus.jsonl", help="Path to corpus JSONL")
    p.add_argument("--retriever-cache-dir", default="data/retriever_cache", help="Retriever cache dir")
    p.add_argument("--results-dir", default="results", help="Results output directory")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--max-context-chars", type=int, default=1800)
    p.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        help="Embedding model name for retrieval (or set EMBEDDING_MODEL)",
    )
    args = p.parse_args()

    cfg = RunConfig(
        seed=int(args.seed),
        top_k=int(args.k),
        max_context_chars=int(args.max_context_chars),
        ontology_key=str(args.ontology_key),
        label_set_path=Path(args.label_set),
        dataset_path=Path(args.dataset),
        corpus_path=Path(args.corpus),
        retriever_cache_dir=Path(args.retriever_cache_dir),
        embedding_model=str(args.embedding_model),
        results_dir=Path(args.results_dir),
    )
    run_official_eval(cfg)
    print("✓ Wrote results/results.json")
    print("✓ Wrote results/predictions.csv")
    print("✓ Wrote results/summary.md")


if __name__ == "__main__":
    main()
