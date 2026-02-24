"""Official evaluation runner (mechanical scoring).

Reads (defaults):
- data/seed_cases_ai_rheum.csv
- data/ai_rheum_label_set.json

Writes:
- results/results.json
- results/predictions.csv
- results/summary.md
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from eval_pipeline import assemble_results, prepare_runtime, run_prediction_stage
from eval_types import DEFAULT_SEED, DEFAULT_TOP_K, RunConfig
from schemas import predictions_to_frame


def run_official_eval(cfg: RunConfig) -> dict[str, Any]:
    runtime = prepare_runtime(cfg)
    batch = run_prediction_stage(runtime)
    pred_df = predictions_to_frame(batch.rows)
    results_doc = assemble_results(runtime, batch)
    results = results_doc.to_dict()

    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    pred_path = cfg.results_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

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
    retrieval_mode = str(results["run"].get("retrieval_mode") or "")
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

    add(pred_df["ok_rag"] == True, 4)
    add(pred_df["ok_rag"] == False, 4)

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
    ttest = (results.get("inferential") or {}).get("paired_ttest_correctness") or {}
    if ttest:
        delta = float(ttest.get("mean_delta_agreement_points") or 0.0)
        improved = int(ttest.get("improved_cases") or 0)
        worse = int(ttest.get("worse_cases") or 0)
        p_value = ttest.get("p_value")
        p_str = f"{float(p_value):.4g}" if isinstance(p_value, (float, int)) else "N/A"
        lines.append("")
        lines.append("## Inferential (Paired t-test)")
        lines.append("")
        lines.append(f"- Mean agreement delta (RAG - No-RAG): {delta:.2f} points")
        lines.append(f"- Improved / worse / unchanged: {improved} / {worse} / {int(ttest.get('unchanged_cases') or 0)}")
        lines.append(f"- t-statistic: {ttest.get('t_stat')}")
        lines.append(f"- p-value: {p_str}")
        lines.append(f"- df: {ttest.get('degrees_freedom')}")
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
    if retrieval_mode:
        lines.append(f"- Retrieval mode: {retrieval_mode}")
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
    try:  # pragma: no cover
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    defaults = RunConfig()

    p = argparse.ArgumentParser(description="Official evaluation runner (exact agreement)")
    p.add_argument("--ontology-key", default=defaults.ontology_key, help="Key for onto_config.get_config (e.g., ai_rheum)")
    p.add_argument("--label-set", default=str(defaults.label_set_path), help="Path to label set JSON")
    p.add_argument("--dataset", default=str(defaults.dataset_path), help="Path to dataset CSV")
    p.add_argument("--corpus", default=str(defaults.corpus_path), help="Path to corpus JSONL")
    p.add_argument("--retriever-cache-dir", default=str(defaults.retriever_cache_dir), help="Retriever cache dir")
    p.add_argument("--results-dir", default=str(defaults.results_dir), help="Results output directory")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--max-context-chars", type=int, default=defaults.max_context_chars)
    p.add_argument(
        "--retrieval",
        choices=["embeddings", "tfidf"],
        default="embeddings",
        help="Retriever backend. Use 'tfidf' for fully local/offline runs.",
    )
    p.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", defaults.embedding_model),
        help="Embedding model name for retrieval (or set EMBEDDING_MODEL)",
    )
    args = p.parse_args()

    cfg = RunConfig(
        seed=int(args.seed),
        top_k=int(args.k),
        max_context_chars=int(args.max_context_chars),
        prefer_embeddings=str(args.retrieval) == "embeddings",
        ontology_key=str(args.ontology_key),
        label_set_path=Path(args.label_set),
        dataset_path=Path(args.dataset),
        corpus_path=Path(args.corpus),
        retriever_cache_dir=Path(args.retriever_cache_dir),
        embedding_model=str(args.embedding_model),
        results_dir=Path(args.results_dir),
    )
    run_official_eval(cfg)
    print(f"✓ Wrote {cfg.results_dir / 'results.json'}")
    print(f"✓ Wrote {cfg.results_dir / 'predictions.csv'}")
    print(f"✓ Wrote {cfg.results_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

