#!/usr/bin/env python3
"""summarize_retrieval.py

Reads retrieval_results.csv files from results/retrieval_experiment/k*/
and produces k-curve plots for all three metrics.

Outputs:
  results/retrieval_experiment/all_metrics.csv
  results/retrieval_experiment/figures.html

Usage:
  python src/summarize_retrieval.py
  python src/summarize_retrieval.py --results-dir results/retrieval_experiment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_all(results_dir: Path) -> pd.DataFrame:
    frames = []
    for k_dir in sorted(results_dir.glob("k*"), key=lambda p: int(p.name[1:])):
        csv_path = k_dir / "retrieval_results.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        # Infer top_k from directory name if column absent
        if "top_k" not in df.columns:
            df["top_k"] = int(k_dir.name[1:])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No retrieval_results.csv files found under {results_dir}/k*/")
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-case rows into per-(k, model) summary stats."""
    scored = df[~df["is_none_case"]].copy()

    rows = []
    for (k, model), g in scored.groupby(["top_k", "model_key"]):
        p_mean = g["precision_at_k"].mean()
        p_std  = g["precision_at_k"].std()
        rank_mean = g["rank_first_relevant"].dropna().mean()
        rank_std  = g["rank_first_relevant"].dropna().std()

        # WHY-rich: include all cases (not just scored ones)
        all_g = df[(df["top_k"] == k) & (df["model_key"] == model)]
        why_mean = all_g["frac_why_rich"].mean()
        why_std  = all_g["frac_why_rich"].std()

        # Delta counts
        rows.append({
            "k": int(k),
            "model": model,
            "precision_at_k_mean": p_mean,
            "precision_at_k_std":  p_std,
            "rank_first_mean":     rank_mean,
            "rank_first_std":      rank_std,
            "frac_why_rich_mean":  why_mean,
            "frac_why_rich_std":   why_std,
            "n_cases":             g["case_id"].nunique(),
        })

    return pd.DataFrame(rows).sort_values(["k", "model"]).reset_index(drop=True)


def delta_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-k delta (biomedbert - minilm) for each metric."""
    scored = df[~df["is_none_case"]]
    rows = []
    for k, g in scored.groupby("top_k"):
        bge = g[g["model_key"] == "biomedbert"]["precision_at_k"]
        mil = g[g["model_key"] == "minilm"]["precision_at_k"]
        # align on case_id
        pivot = g.pivot_table(index="case_id", columns="model_key", values="precision_at_k")
        if "biomedbert" in pivot.columns and "minilm" in pivot.columns:
            delta = pivot["biomedbert"] - pivot["minilm"]
            rows.append({
                "k": int(k),
                "mean_delta": delta.mean(),
                "improved": int((delta > 0).sum()),
                "worse":    int((delta < 0).sum()),
                "same":     int((delta == 0).sum()),
            })
    return pd.DataFrame(rows)


def print_summary(agg: pd.DataFrame, deltas: pd.DataFrame) -> None:
    print(f"\n{'='*65}")
    print("ALL METRICS BY K")
    print(f"{'='*65}")
    print(f"{'k':>3}  {'model':>12}  {'P@k':>6}  {'Rank':>6}  {'WHY%':>6}")
    for _, r in agg.iterrows():
        print(f"  {int(r.k):>2}  {r.model:>12}  "
              f"{r.precision_at_k_mean:6.3f}  "
              f"{r.rank_first_mean:6.2f}  "
              f"{r.frac_why_rich_mean:6.3f}")

    print(f"\n{'='*65}")
    print("DELTA (biomedbert − minilm) on P@k")
    print(f"{'='*65}")
    for _, r in deltas.iterrows():
        print(f"  k={int(r.k):>2}  delta={r.mean_delta:+.3f}  "
              f"improved={int(r.improved)}  worse={int(r.worse)}  same={int(r.same)}")


def render_html(agg: pd.DataFrame, deltas: pd.DataFrame) -> str:
    models = agg["model"].unique().tolist()
    ks = sorted(agg["k"].unique().tolist())

    def series(model, col):
        sub = agg[agg["model"] == model].sort_values("k")
        return sub[col + "_mean"].tolist()

    def series_std(model, col):
        sub = agg[agg["model"] == model].sort_values("k")
        return sub[col + "_std"].tolist()

    colors = {"minilm": "#2c5f8a", "biomedbert": "#e07b39"}
    fill_colors = {"minilm": "rgba(44,95,138,0.12)", "biomedbert": "rgba(224,123,57,0.12)"}

    def dataset(model, col, label, dashed=False):
        color = colors.get(model, "#888")
        fill = fill_colors.get(model, "rgba(0,0,0,0.05)")
        vals = series(model, col)
        stds = series_std(model, col)
        upper = [v + s for v, s in zip(vals, stds)]
        lower = [v - s for v, s in zip(vals, stds)]
        border_dash = "[6,3]" if dashed else "[]"
        return f"""
        {{
          label: '{label}',
          data: {vals},
          borderColor: '{color}',
          backgroundColor: '{color}',
          borderDash: {border_dash},
          borderWidth: 2.5,
          pointRadius: 4,
          tension: 0.3,
          fill: false,
        }}"""

    def error_band(model, col, label):
        color = fill_colors.get(model, "rgba(0,0,0,0.05)")
        vals = series(model, col)
        stds = series_std(model, col)
        upper = [round(v + s, 4) for v, s in zip(vals, stds)]
        lower = [round(max(v - s, 0), 4) for v, s in zip(vals, stds)]
        return upper, lower, color

    # Build chart configs
    delta_vals = deltas.sort_values("k")["mean_delta"].tolist()
    delta_colors = ['"#27ae60"' if d >= 0 else '"#e74c3c"' for d in delta_vals]

    # rank: lower is better, so invert for visual clarity note
    rank_note = "Lower is better"
    why_note  = "Higher = more substantive chunks retrieved"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Retrieval Experiment — k Curves</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1000px; margin: 40px auto; padding: 0 24px;
       background: #f8f9fa; color: #222; }}
h1 {{ color: #1a3a5c; border-bottom: 3px solid #1a3a5c; padding-bottom: 10px; }}
h2 {{ color: #2c5f8a; margin-top: 36px; font-size: 1.1em; }}
.chart-wrap {{ background: white; border-radius: 10px; padding: 24px;
               margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.note {{ font-size: 0.82em; color: #888; margin-top: 6px; font-style: italic; }}
.legend {{ display: flex; gap: 24px; margin-bottom: 10px; font-size: 0.88em; }}
.leg-dot {{ display: inline-block; width: 14px; height: 14px;
            border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; margin-top: 12px; }}
th {{ background: #1a3a5c; color: white; padding: 8px 12px; text-align: left; }}
td {{ padding: 7px 12px; border-bottom: 1px solid #e8e8e8; }}
tr:nth-child(even) {{ background: #f5f7fa; }}
.pos {{ color: #27ae60; font-weight: bold; }}
.neg {{ color: #e74c3c; font-weight: bold; }}
</style>
</head>
<body>
<h1>🔬 Retrieval Experiment — MiniLM vs BiomedBERT</h1>
<div class="legend">
  <span><span class="leg-dot" style="background:#2c5f8a"></span>MiniLM</span>
  <span><span class="leg-dot" style="background:#e07b39"></span>BiomedBERT</span>
</div>

<h2>Precision@k (fraction of retrieved chunks linked to gold diagnosis)</h2>
<div class="chart-wrap">
  <canvas id="precChart" height="80"></canvas>
  <p class="note">Higher is better. Excludes NONE/mimicker cases (no gold diagnosis to match).</p>
</div>

<h2>Rank of First Relevant Chunk</h2>
<div class="chart-wrap">
  <canvas id="rankChart" height="80"></canvas>
  <p class="note">{rank_note}. A rank of 1 means the top result was relevant.</p>
</div>

<h2>Fraction of Retrieved Chunks with WHY Content</h2>
<div class="chart-wrap">
  <canvas id="whyChart" height="80"></canvas>
  <p class="note">{why_note}. Label-only chunks score 0.</p>
</div>

<h2>Delta Precision@k (BiomedBERT − MiniLM)</h2>
<div class="chart-wrap">
  <canvas id="deltaChart" height="70"></canvas>
  <p class="note">Positive = BiomedBERT retrieved more relevant chunks at this k.</p>
</div>

<h2>Full Results Table</h2>
<div class="chart-wrap">
<table>
<tr><th>k</th><th>Model</th><th>P@k</th><th>Rank (mean)</th><th>WHY-rich%</th></tr>
{"".join(
    f"<tr><td>{int(r.k)}</td><td>{r.model}</td>"
    f"<td>{r.precision_at_k_mean:.3f}</td>"
    f"<td>{r.rank_first_mean:.2f}</td>"
    f"<td>{r.frac_why_rich_mean:.3f}</td></tr>"
    for _, r in agg.iterrows()
)}
</table>
</div>

<script>
const ks = {ks};
const minilm_prec  = {series('minilm',      'precision_at_k')};
const biomed_prec  = {series('biomedbert',  'precision_at_k')};
const minilm_rank  = {series('minilm',      'rank_first')};
const biomed_rank  = {series('biomedbert',  'rank_first')};
const minilm_why   = {series('minilm',      'frac_why_rich')};
const biomed_why   = {series('biomedbert',  'frac_why_rich')};
const delta_vals   = {delta_vals};
const delta_colors = [{",".join(delta_colors)}];

const commonOpts = (yLabel, min=0) => ({{
  responsive: true,
  plugins: {{ legend: {{ position: 'top' }} }},
  scales: {{
    x: {{ title: {{ display: true, text: 'k (chunks retrieved)' }} }},
    y: {{ title: {{ display: true, text: yLabel }}, min }},
  }}
}});

new Chart(document.getElementById('precChart'), {{
  type: 'line',
  data: {{
    labels: ks,
    datasets: [
      {{ label: 'MiniLM',      data: minilm_prec, borderColor: '#2c5f8a',
         backgroundColor: '#2c5f8a', borderWidth: 2.5, pointRadius: 4, tension: 0.3 }},
      {{ label: 'BiomedBERT',  data: biomed_prec, borderColor: '#e07b39',
         backgroundColor: '#e07b39', borderWidth: 2.5, pointRadius: 4, tension: 0.3 }},
    ]
  }},
  options: commonOpts('Precision@k', 0),
}});

new Chart(document.getElementById('rankChart'), {{
  type: 'line',
  data: {{
    labels: ks,
    datasets: [
      {{ label: 'MiniLM',     data: minilm_rank, borderColor: '#2c5f8a',
         backgroundColor: '#2c5f8a', borderWidth: 2.5, pointRadius: 4, tension: 0.3 }},
      {{ label: 'BiomedBERT', data: biomed_rank, borderColor: '#e07b39',
         backgroundColor: '#e07b39', borderWidth: 2.5, pointRadius: 4, tension: 0.3 }},
    ]
  }},
  options: commonOpts('Mean rank of first relevant chunk', 1),
}});

new Chart(document.getElementById('whyChart'), {{
  type: 'line',
  data: {{
    labels: ks,
    datasets: [
      {{ label: 'MiniLM',     data: minilm_why, borderColor: '#2c5f8a',
         backgroundColor: '#2c5f8a', borderWidth: 2.5, pointRadius: 4, tension: 0.3 }},
      {{ label: 'BiomedBERT', data: biomed_why, borderColor: '#e07b39',
         backgroundColor: '#e07b39', borderWidth: 2.5, pointRadius: 4, tension: 0.3 }},
    ]
  }},
  options: commonOpts('Fraction WHY-rich', 0),
}});

new Chart(document.getElementById('deltaChart'), {{
  type: 'bar',
  data: {{
    labels: ks,
    datasets: [{{
      label: 'Delta P@k (BiomedBERT − MiniLM)',
      data: delta_vals,
      backgroundColor: delta_colors,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'top' }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'k' }} }},
      y: {{ title: {{ display: true, text: 'Delta' }} }},
    }}
  }},
}});
</script>
</body>
</html>"""
    return html


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/retrieval_experiment")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    print(f"[Load] Reading from {results_dir}/k*/retrieval_results.csv ...")
    df = load_all(results_dir)
    print(f"[Load] {len(df)} rows across {df['top_k'].nunique()} k values")

    agg = aggregate(df)
    deltas = delta_table(df)

    print_summary(agg, deltas)

    # Save aggregated CSV
    csv_path = results_dir / "all_metrics.csv"
    agg.to_csv(csv_path, index=False)
    print(f"\n[Output] Saved: {csv_path}")

    # Save HTML figures
    html = render_html(agg, deltas)
    html_path = results_dir / "figures.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[Output] Saved: {html_path}")
    print(f"\nOpen in browser: {html_path}")


if __name__ == "__main__":
    main()
