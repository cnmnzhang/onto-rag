#!/usr/bin/env python3
"""src/score_results.py

Takes clinician_ratings.json (from review sheet) + review_key.json
and produces a scored analysis comparing RAG vs No-RAG.

Outputs:
  results/scored_results.csv     -- per-case per-dimension scores
  results/analysis_summary.md    -- written summary of findings
  results/analysis_figures.html  -- visual charts

Usage:
  python src/score_results.py
  python src/score_results.py --ratings clinician_ratings.json
  python src/score_results.py --ratings clinician_ratings.json --key results/review_key.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

DIMENSIONS = ["accuracy", "completeness", "reasoning", "explainability", "utility"]
DIM_LABELS = {
    "accuracy": "Diagnostic Accuracy",
    "completeness": "Differential Completeness", 
    "reasoning": "Reasoning Quality",
    "explainability": "Explainability / Traceability",
    "utility": "Clinical Utility",
}

PREFERENCE_MAP = {
    "A": +1.0,
    "A_slight": +0.5,
    "equal": 0.0,
    "B_slight": -0.5,
    "B": -1.0,
}


def load_ratings(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_key(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score(ratings: dict, key: dict) -> pd.DataFrame:
    assignments = key.get("assignments", {})
    rows = []

    for case_data in ratings.get("cases", []):
        case_id = case_data["case_id"]
        assign = assignments.get(case_id, {})

        for letter in ["A", "B"]:
            condition = assign.get(letter, f"unknown_{letter}")
            is_rag = condition == "rag"
            resp = case_data.get("responses", {}).get(letter, {})
            ratings_d = resp.get("ratings", {})
            comment = resp.get("comment", "")

            row = {
                "case_id": case_id,
                "letter": letter,
                "condition": condition,
                "is_rag": is_rag,
                "comment": comment,
            }
            for dim in DIMENSIONS:
                row[dim] = ratings_d.get(dim)
            rows.append(row)

        # Preference: positive = preferred A
        pref_raw = case_data.get("preference", "equal") or "equal"
        pref_val = PREFERENCE_MAP.get(pref_raw, 0.0)

        # Convert to RAG-preference direction
        # If A = rag, then positive pref_val means RAG preferred
        # If A = no_rag, then negative pref_val means RAG preferred
        a_is_rag = assignments.get(case_id, {}).get("A") == "rag"
        rag_pref = pref_val if a_is_rag else -pref_val

        rows[-2]["preference_raw"] = pref_raw
        rows[-2]["rag_preference_score"] = rag_pref
        rows[-1]["preference_raw"] = pref_raw
        rows[-1]["rag_preference_score"] = rag_pref

    return pd.DataFrame(rows)


def _wilcoxon_or_ttest(rag_scores: list, no_rag_scores: list) -> dict:
    """Run paired test. Use Wilcoxon signed-rank if available, fallback to t-test."""
    a = np.array(rag_scores, dtype=float)
    b = np.array(no_rag_scores, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    diff = a - b
    mean_delta = float(np.mean(diff)) if n else 0.0

    result = {
        "n_pairs": n,
        "mean_rag": float(np.nanmean(a)) if n else None,
        "mean_no_rag": float(np.nanmean(b)) if n else None,
        "mean_delta": mean_delta,
    }

    if n < 2:
        result.update({"test": "insufficient_data", "p_value": None, "statistic": None})
        return result

    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="zsplit")
        result.update({"test": "wilcoxon_signed_rank", "statistic": float(stat), "p_value": float(p)})
    except Exception:
        try:
            from scipy.stats import ttest_rel
            stat, p = ttest_rel(a, b)
            result.update({"test": "paired_t_test", "statistic": float(stat), "p_value": float(p)})
        except Exception:
            std = float(np.std(diff, ddof=1)) if n > 1 else 0
            se = std / np.sqrt(n) if std > 0 else None
            t = float(mean_delta / se) if se else None
            result.update({"test": "manual_t_fallback", "statistic": t, "p_value": None})

    return result


def analyse(scored_df: pd.DataFrame) -> dict:
    rag_df = scored_df[scored_df["is_rag"] == True].set_index("case_id")
    no_rag_df = scored_df[scored_df["is_rag"] == False].set_index("case_id")

    common = rag_df.index.intersection(no_rag_df.index)
    rag_df = rag_df.loc[common]
    no_rag_df = no_rag_df.loc[common]

    results = {}

    for dim in DIMENSIONS:
        rag_scores = rag_df[dim].tolist()
        no_rag_scores = no_rag_df[dim].tolist()
        stat = _wilcoxon_or_ttest(rag_scores, no_rag_scores)
        results[dim] = stat

    # Overall preference
    pref_scores = scored_df.drop_duplicates("case_id")["rag_preference_score"].dropna()
    n_prefer_rag = int((pref_scores > 0).sum())
    n_prefer_no_rag = int((pref_scores < 0).sum())
    n_equal = int((pref_scores == 0).sum())
    results["preference"] = {
        "n_prefer_rag": n_prefer_rag,
        "n_prefer_no_rag": n_prefer_no_rag,
        "n_equal": n_equal,
        "mean_rag_preference": float(pref_scores.mean()) if len(pref_scores) else 0.0,
    }

    return results


def _sig(p: float | None) -> str:
    if p is None:
        return "p=N/A"
    if p < 0.001:
        return "p<0.001 ***"
    if p < 0.01:
        return f"p={p:.3f} **"
    if p < 0.05:
        return f"p={p:.3f} *"
    return f"p={p:.3f} (ns)"


def render_summary(scored_df: pd.DataFrame, analysis: dict, key: dict) -> str:
    model_key = key.get("model_key", "unknown")
    n_cases = scored_df["case_id"].nunique()

    lines = [
        "# RAG vs No-RAG: Clinician Evaluation Results",
        "",
        f"**Embedding model:** {model_key}  ",
        f"**Cases evaluated:** {n_cases}  ",
        "",
        "## Dimension Scores (1–5 scale)",
        "",
        "| Dimension | RAG Mean | No-RAG Mean | Delta | Test | Result |",
        "|---|---:|---:|---:|---|---|",
    ]

    for dim in DIMENSIONS:
        r = analysis[dim]
        m_rag = f"{r['mean_rag']:.2f}" if r["mean_rag"] is not None else "—"
        m_no = f"{r['mean_no_rag']:.2f}" if r["mean_no_rag"] is not None else "—"
        delta = f"{r['mean_delta']:+.2f}" if r["mean_delta"] is not None else "—"
        test = r.get("test", "—")
        sig = _sig(r.get("p_value"))
        lines.append(f"| {DIM_LABELS[dim]} | {m_rag} | {m_no} | {delta} | {test} | {sig} |")

    pref = analysis.get("preference", {})
    lines += [
        "",
        "## Overall Response Preference",
        "",
        f"- Prefer RAG: **{pref.get('n_prefer_rag', 0)}** cases",
        f"- Prefer No-RAG: **{pref.get('n_prefer_no_rag', 0)}** cases",
        f"- Equal: **{pref.get('n_equal', 0)}** cases",
        f"- Mean RAG preference score: **{pref.get('mean_rag_preference', 0):.2f}** (positive = RAG preferred)",
        "",
        "## Interpretation",
        "",
        "Dimensions where RAG delta > 0 indicate the ontology-grounded response "
        "was rated higher by the clinician. Statistical significance at p<0.05 is marked with *.",
        "",
        "**Note:** With small N, effect sizes are more meaningful than p-values. "
        "Consider the clinical magnitude of any delta alongside statistical tests.",
        "",
        "## Comments from Reviewer",
        "",
    ]

    # Add any per-case comments
    for _, row in scored_df.iterrows():
        if row.get("comment", "").strip():
            lines.append(f"- Case {row['case_id']} (Response {row['letter']}, {row['condition']}): {row['comment']}")

    return "\n".join(lines)

def auto_explainability_score(responses_path: Path) -> pd.DataFrame:
    """
    Count how many ontology concepts are explicitly cited per response.
    Proxy for automated explainability measurement.
    """
    import re
    df = pd.read_csv(responses_path)
    
    ontology_markers = [
        r"per ontology",
        r"ontology (class|concept|grounding|knowledge)",
        r"ontology:",
        r"inflammatory polyarthritis",
        r"examiner.s diagnosis",
        r"fulfills? a diagnostic criterion",
        r"type (i|ii|iii) synovial fluid",
        r"(marginal|non-marginal) syndesmophyte",
        r"(positively|negatively) birefringent",
        r"diagnostic criterion for",
    ]
    pattern = re.compile("|".join(ontology_markers), re.IGNORECASE)
    
    rows = []
    for _, row in df.iterrows():
        rag_hits = len(pattern.findall(str(row["response_rag"])))
        no_rag_hits = len(pattern.findall(str(row["response_no_rag"])))
        rows.append({
            "case_id": row["case_id"],
            "model_key": row.get("model_key", ""),
            "rag_ontology_citations": rag_hits,
            "no_rag_ontology_citations": no_rag_hits,
            "delta_citations": rag_hits - no_rag_hits,
        })
    
    result = pd.DataFrame(rows)
    print("\nAuto Explainability (ontology citation count):")
    print(f"  Mean RAG citations:    {result['rag_ontology_citations'].mean():.2f}")
    print(f"  Mean No-RAG citations: {result['no_rag_ontology_citations'].mean():.2f}")
    print(f"  Mean delta:            {result['delta_citations'].mean():+.2f}")
    return result


def render_html_charts(scored_df: pd.DataFrame, analysis: dict) -> str:
    """Generate a self-contained HTML page with bar charts."""
    dims = DIMENSIONS
    rag_means = [analysis[d].get("mean_rag") or 0 for d in dims]
    no_rag_means = [analysis[d].get("mean_no_rag") or 0 for d in dims]
    dim_labels = [DIM_LABELS[d] for d in dims]
    pref = analysis.get("preference", {})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Rheumatology RAG Evaluation Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background:#fafafa; }}
h1 {{ color: #1a3a5c; }}
h2 {{ color: #2c5f8a; margin-top: 30px; }}
.chart-container {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
.summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
.metric-card {{ background: white; border-radius: 8px; padding: 16px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
.metric-card .value {{ font-size: 2em; font-weight: bold; color: #1a3a5c; }}
.metric-card .label {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
.delta-pos {{ color: #27ae60; }}
.delta-neg {{ color: #e74c3c; }}
</style>
</head>
<body>
<h1>🩺 RAG vs No-RAG — Clinician Evaluation Results</h1>

<h2>Dimension Comparison</h2>
<div class="chart-container">
  <canvas id="dimChart" height="80"></canvas>
</div>

<h2>Per-Dimension Deltas (RAG − No-RAG)</h2>
<div class="chart-container">
  <canvas id="deltaChart" height="60"></canvas>
</div>

<h2>Overall Preference</h2>
<div class="summary-grid">
  <div class="metric-card"><div class="value delta-pos">{pref.get('n_prefer_rag',0)}</div><div class="label">Prefer RAG</div></div>
  <div class="metric-card"><div class="value">{pref.get('n_equal',0)}</div><div class="label">Equal</div></div>
  <div class="metric-card"><div class="value delta-neg">{pref.get('n_prefer_no_rag',0)}</div><div class="label">Prefer No-RAG</div></div>
  <div class="metric-card"><div class="value">{pref.get('mean_rag_preference',0):.2f}</div><div class="label">Mean RAG Pref Score</div></div>
</div>

<script>
const dims = {json.dumps(dim_labels)};
const ragMeans = {json.dumps(rag_means)};
const noRagMeans = {json.dumps(no_rag_means)};
const deltas = ragMeans.map((v, i) => +(v - noRagMeans[i]).toFixed(3));

new Chart(document.getElementById('dimChart'), {{
  type: 'bar',
  data: {{
    labels: dims,
    datasets: [
      {{ label: 'RAG', data: ragMeans, backgroundColor: '#2c5f8a', borderRadius: 4 }},
      {{ label: 'No-RAG', data: noRagMeans, backgroundColor: '#a8c5df', borderRadius: 4 }},
    ]
  }},
  options: {{
    plugins: {{ legend: {{ position: 'top' }} }},
    scales: {{
      y: {{ min: 0, max: 5, title: {{ display: true, text: 'Mean Rating (1–5)' }} }},
    }}
  }}
}});

new Chart(document.getElementById('deltaChart'), {{
  type: 'bar',
  data: {{
    labels: dims,
    datasets: [{{
      label: 'Delta (RAG − No-RAG)',
      data: deltas,
      backgroundColor: deltas.map(d => d >= 0 ? '#27ae60' : '#e74c3c'),
      borderRadius: 4,
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ position: 'top' }} }},
    scales: {{
      y: {{ title: {{ display: true, text: 'Score Delta' }} }},
    }}
  }}
}});
</script>
</body>
</html>"""


def main() -> None:
    p = argparse.ArgumentParser(description="Score clinician ratings against review key")
    p.add_argument("--ratings", default="results/clinician_ratings.json")
    p.add_argument("--key", default="results/review_key.json")
    p.add_argument("--output-dir", default="results")
    args = p.parse_args()

    ratings_path = Path(args.ratings)
    key_path = Path(args.key)

    if not ratings_path.exists():
        print(f"[Score] Ratings file not found: {ratings_path}")
        print(f"[Score] Have the clinician complete review_sheet.html and download ratings first.")
        sys.exit(1)

    if not key_path.exists():
        print(f"[Score] Key file not found: {key_path}")
        sys.exit(1)

    ratings = load_ratings(ratings_path)
    key = load_key(key_path)

    scored_df = score(ratings, key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scored_csv = output_dir / "scored_results.csv"
    scored_df.to_csv(scored_csv, index=False)
    print(f"[Score] Saved: {scored_csv}")

    analysis = analyse(scored_df)

    summary_md = render_summary(scored_df, analysis, key)
    summary_path = output_dir / "analysis_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    print(f"[Score] Saved: {summary_path}")

    charts_html = render_html_charts(scored_df, analysis)
    charts_path = output_dir / "analysis_figures.html"
    charts_path.write_text(charts_html, encoding="utf-8")
    print(f"[Score] Saved: {charts_path}")

    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    for dim in DIMENSIONS:
        r = analysis[dim]
        delta = r.get("mean_delta", 0) or 0
        p_val = r.get("p_value")
        sig = _sig(p_val)
        direction = "↑ RAG better" if delta > 0.05 else ("↓ No-RAG better" if delta < -0.05 else "→ similar")
        print(f"  {DIM_LABELS[dim]:35s}: delta={delta:+.2f} | {sig} | {direction}")

    pref = analysis.get("preference", {})
    print(f"\n  Overall preference: RAG={pref.get('n_prefer_rag',0)}, "
          f"Equal={pref.get('n_equal',0)}, No-RAG={pref.get('n_prefer_no_rag',0)}")
    
    auto_df = auto_explainability_score(Path("results/all_responses.csv"))
    auto_df.to_csv(output_dir / "explainability_auto.csv", index=False)


if __name__ == "__main__":
    main()
