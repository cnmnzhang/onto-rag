#!/usr/bin/env python3
"""retrieval_experiment.py

Compares MiniLM vs BiomedBERT embedding models on retrieval quality.

Metric: precision@k — fraction of top-k retrieved chunks that are linked
to the gold-label diagnosis in the corpus (via linked_dx_codes).

Secondary metrics:
  - rank of first relevant chunk (lower is better)
  - fraction of retrieved chunks that have WHY content (not label-only)

Results are stratified by:
  - category (classic / atypical / early / mimicker / none / ambiguous)
  - ontology_support_level (rich / none)

Usage:
  python retrieval_experiment.py
  python retrieval_experiment.py --cases data/test_cases_v2.csv --ks 1,3,5,10
  python retrieval_experiment.py --responses results/responses_minilm.csv --ks 1,3,5,10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from retriever import Retriever
from corpus_builder import load_corpus

MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "biomedbert": "neuml/pubmedbert-base-embeddings",
}

AIR_PREFIX = "http://purl.bioontology.org/ontology/AIR/"


def _gold_codes(gold_label: str) -> set[str]:
    """Extract dx codes from a gold label URI, handling pipe-separated ambiguous cases."""
    codes = set()
    for part in str(gold_label).split("|"):
        part = part.strip()
        if part.startswith(AIR_PREFIX):
            codes.add(part.replace(AIR_PREFIX, ""))
        elif part not in ("NONE", ""):
            codes.add(part)
    return codes


def precision_at_k(retrieved: list[dict], gold_codes: set[str]) -> float:
    if not retrieved or not gold_codes:
        return 0.0
    hits = sum(
        1 for d in retrieved
        if gold_codes & set(d.get("linked_dx_codes", []))
    )
    return hits / len(retrieved)


def rank_first_relevant(retrieved: list[dict], gold_codes: set[str]) -> int | None:
    """1-indexed rank of first relevant chunk. None if none found."""
    for i, d in enumerate(retrieved, 1):
        if gold_codes & set(d.get("linked_dx_codes", [])):
            return i
    return None


def frac_why_rich(retrieved: list[dict]) -> float:
    """Fraction of retrieved chunks that have a WHY section (not label-only)."""
    if not retrieved:
        return 0.0
    return sum(1 for d in retrieved if d.get("definition_why", "").strip()) / len(retrieved)


def run_experiment(
    cases_df: pd.DataFrame,
    corpus: list[dict],
    ks: list[int],
    cache_dir: str = "data/retriever_cache",
) -> pd.DataFrame:
    """
    Retrieve at max(ks) once per model/case, then slice results for each k.
    This avoids re-encoding queries multiple times.
    """
    max_k = max(ks)
    rows = []

    for model_key, model_name in MODELS.items():
        print(f"\n[{model_key}] Building retriever ({model_name}), max_k={max_k} ...")
        retriever = Retriever(
            corpus,
            model_name=model_name,
            top_k=max_k,
            cache_dir=f"{cache_dir}/{model_key}",
            filter_eval_only=True,
        )

        for _, case in cases_df.iterrows():
            case_id = str(case["case_id"])
            chart_text = str(case["chart_text"])
            gold_label = str(case.get("gold_label", "NONE"))
            category = str(case.get("category", ""))
            support = str(case.get("ontology_support_level", "unknown"))

            codes = _gold_codes(gold_label)
            is_none_case = not codes

            # Retrieve once at max_k, slice per k value
            all_retrieved = retriever.retrieve(chart_text, top_k=max_k)

            for k in ks:
                retrieved = all_retrieved[:k]
                p_at_k = precision_at_k(retrieved, codes) if not is_none_case else None
                rank = rank_first_relevant(retrieved, codes) if not is_none_case else None
                why_frac = frac_why_rich(retrieved)

                rows.append({
                    "model_key": model_key,
                    "case_id": case_id,
                    "gold_label": gold_label,
                    "category": category,
                    "ontology_support_level": support,
                    "is_none_case": is_none_case,
                    "k": k,
                    "precision_at_k": p_at_k,
                    "rank_first_relevant": rank,
                    "frac_why_rich": why_frac,
                    "retrieved_chunks": json.dumps(
                        [{"label": d.get("label"), "score": round(d.get("retrieval_score", 0), 4),
                          "linked_dx_codes": d.get("linked_dx_codes", []),
                          "has_why": bool(d.get("definition_why", "").strip())}
                         for d in retrieved]
                    ),
                })

            # Print once per case at max_k
            p_max = precision_at_k(all_retrieved, codes) if not is_none_case else None
            status = f"p@{max_k}={p_max:.2f}" if p_max is not None else "none-case"
            print(f"  {case_id:10s} | {category:10s} | {status}")

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, ks: list[int]) -> None:
    scored = df[~df["is_none_case"]].copy()

    # ── Precision@k curve ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PRECISION@K CURVE")
    print(f"{'='*60}")
    header = f"{'Model':15s}" + "".join(f"  p@{k:>2}" for k in ks) + "  rank(mean)"
    print(header)
    for mk in MODELS:
        m = scored[scored["model_key"] == mk]
        row = f"  {mk:13s}"
        for k in ks:
            mk_k = m[m["k"] == k]
            p = mk_k["precision_at_k"].mean()
            row += f"  {p:5.3f}"
        # rank uses max k (most generous window)
        max_k = max(ks)
        r = m[m["k"] == max_k]["rank_first_relevant"].dropna().mean()
        row += f"  {r:10.2f}"
        print(row)

    # ── Delta at each k ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DELTA (biomedbert − minilm) BY K")
    print(f"{'='*60}")
    for k in ks:
        sub = scored[scored["k"] == k]
        pivot = sub.pivot_table(index="case_id", columns="model_key", values="precision_at_k")
        if "biomedbert" in pivot.columns and "minilm" in pivot.columns:
            delta = pivot["biomedbert"] - pivot["minilm"]
            print(f"  k={k:2d}  delta={delta.mean():+.3f}  "
                  f"improved={( delta > 0).sum()}  worse={(delta < 0).sum()}  "
                  f"same={(delta == 0).sum()}")

    # ── By category (at median k) ────────────────────────────────────────
    mid_k = ks[len(ks) // 2]
    sub_mid = scored[scored["k"] == mid_k]
    print(f"\n{'='*60}")
    print(f"BY CATEGORY  (k={mid_k})")
    print(f"{'='*60}")
    for cat in sorted(sub_mid["category"].unique()):
        sub = sub_mid[sub_mid["category"] == cat]
        n = sub["case_id"].nunique() // len(MODELS)
        print(f"\n  {cat} (n={n})")
        for mk in MODELS:
            p = sub[sub["model_key"] == mk]["precision_at_k"].mean()
            print(f"    {mk:13s}  p@{mid_k}={p:.3f}")

    # ── By ontology support level (at median k) ──────────────────────────
    print(f"\n{'='*60}")
    print(f"BY ONTOLOGY SUPPORT LEVEL  (k={mid_k})")
    print(f"{'='*60}")
    for lvl in ["rich", "partial", "none"]:
        sub = sub_mid[sub_mid["ontology_support_level"] == lvl]
        if sub.empty:
            continue
        n = sub["case_id"].nunique() // len(MODELS)
        print(f"\n  support={lvl} (n={n})")
        for mk in MODELS:
            p = sub[sub["model_key"] == mk]["precision_at_k"].mean()
            print(f"    {mk:13s}  p@{mid_k}={p:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/ai_rheum_corpus_v3.jsonl")
    p.add_argument("--cases", default="data/test_cases_v2.csv",
                   help="Test cases CSV. If --responses is set, cases are read from there instead.")
    p.add_argument("--responses", default=None,
                   help="Optional: use an existing responses CSV (e.g. results/responses_minilm.csv) "
                        "to read chart_text and gold_label from — avoids needing test_cases_v2.csv.")
    p.add_argument("--ks", default="1,3,5,10",
                   help="Comma-separated list of k values to evaluate, e.g. 1,3,5,10")
    p.add_argument("--output-dir", default="results/retrieval_experiment")
    p.add_argument("--cache-dir", default="data/retriever_cache")
    args = p.parse_args()

    ks = sorted(set(int(x.strip()) for x in args.ks.split(",") if x.strip()))
    if not ks:
        print("No valid k values provided.")
        sys.exit(1)

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        sys.exit(1)

    print(f"[Corpus] Loading {corpus_path} ...")
    corpus = load_corpus(corpus_path)
    print(f"[Corpus] {len(corpus)} records")
    print(f"[Config] k values: {ks}")

    if args.responses:
        cases_path = Path(args.responses)
        print(f"[Cases] Reading from responses file: {cases_path}")
        cases_df = pd.read_csv(cases_path)
        cases_df = cases_df.drop_duplicates(subset="case_id").reset_index(drop=True)
    else:
        cases_path = Path(args.cases)
        print(f"[Cases] Reading from: {cases_path}")
        cases_df = pd.read_csv(cases_path)

    required = {"case_id", "chart_text", "gold_label"}
    missing = required - set(cases_df.columns)
    if missing:
        print(f"Missing columns: {missing}")
        sys.exit(1)

    print(f"[Cases] {len(cases_df)} cases loaded")

    results_df = run_experiment(cases_df, corpus, ks=ks, cache_dir=args.cache_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "retrieval_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[Output] Saved: {results_path}")

    print_summary(results_df, ks=ks)

    # Markdown summary
    scored = results_df[~results_df["is_none_case"]]
    summary_lines = ["# Retrieval Experiment Summary", ""]
    summary_lines.append("## Precision@k by model")
    summary_lines.append("")
    header = "| Model |" + "".join(f" p@{k} |" for k in ks)
    sep = "|---|" + "".join("---:|" for _ in ks)
    summary_lines += [header, sep]
    for mk in MODELS:
        m = scored[scored["model_key"] == mk]
        row = f"| {mk} |"
        for k in ks:
            p = m[m["k"] == k]["precision_at_k"].mean()
            row += f" {p:.3f} |"
        summary_lines.append(row)

    summary_lines += ["", "## Delta (biomedbert − minilm)", ""]
    for k in ks:
        sub = scored[scored["k"] == k]
        pivot = sub.pivot_table(index="case_id", columns="model_key", values="precision_at_k")
        if "biomedbert" in pivot.columns and "minilm" in pivot.columns:
            delta = pivot["biomedbert"] - pivot["minilm"]
            summary_lines.append(
                f"- k={k}: mean delta={delta.mean():+.3f}  "
                f"(improved={( delta>0).sum()}, worse={(delta<0).sum()}, same={(delta==0).sum()})"
            )

    md_path = output_dir / "retrieval_summary.md"
    md_path.write_text("\n".join(summary_lines))
    print(f"[Output] Saved: {md_path}")


if __name__ == "__main__":
    main()
