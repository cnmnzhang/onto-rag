#!/usr/bin/env python3
"""src/generate_review_sheet.py

Generates a blinded clinician review sheet from experiment results.

For each case, the clinician sees:
  - The patient vignette
  - Response A and Response B (randomly assigned, labels hidden)
  - Rating form for each response

Outputs:
  results/review_sheet.html     -- formatted HTML for browser-based review
  results/review_sheet.csv      -- raw blinded data (for manual rating entry)
  results/review_key.json       -- mapping of A/B to actual conditions (KEEP HIDDEN until scoring)

Rating dimensions (1-5 scale):
  1. Diagnostic accuracy       -- Is the primary diagnosis correct/reasonable?
  2. Differential completeness -- Are relevant alternatives covered?
  3. Reasoning quality         -- Is the clinical reasoning explicit and sound?
  4. Clinical utility          -- Would this response change/guide management?
  + Free text comment

Usage:
  python src/generate_review_sheet.py
  python src/generate_review_sheet.py --input results/responses_minilm.csv --model minilm
  python src/generate_review_sheet.py --input results/all_responses.csv
"""

from __future__ import annotations

import argparse
import html as _html
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


RATING_CRITERIA = [
    ("accuracy", "Diagnostic Accuracy", "Is the primary diagnosis correct or clinically reasonable?"),
    ("completeness", "Differential Completeness", "Does the differential cover the key alternatives appropriately?"),
    ("reasoning", "Reasoning Quality", "Is the clinical reasoning explicit, evidence-based, and sound?"),
    ("reasoning_transparency", "Reasoning Transparency", "Can you follow the logical chain from finding to diagnosis? Are specific lab values, exam findings, and timecourses cited by name — or are conclusions just asserted?"),
    ("calibration", "Calibration", "Does the expressed confidence match the actual strength of the evidence? Does the response acknowledge uncertainty where the case is genuinely ambiguous?"),
    ("utility", "Clinical Utility", "Would this response meaningfully guide clinical management?"),
]

CSS = """
body { font-family: Georgia, serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; background: #fafafa; color: #222; }
h1 { color: #1a3a5c; border-bottom: 3px solid #1a3a5c; padding-bottom: 10px; }
h2 { color: #1a3a5c; margin-top: 40px; }
h3 { color: #2c5f8a; }
.case-block { background: white; border: 1px solid #ccd; border-radius: 8px; margin: 30px 0; padding: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.07); }
.case-header { background: #1a3a5c; color: white; padding: 12px 18px; border-radius: 6px 6px 0 0; margin: -24px -24px 20px -24px; }
.case-header h2 { color: white; margin: 0; font-size: 1.1em; }
.vignette { background: #f0f4f8; border-left: 4px solid #2c5f8a; padding: 14px 18px; margin: 16px 0; border-radius: 0 6px 6px 0; font-size: 0.97em; line-height: 1.6; }
.vignette p, .response-text p { margin: 0.4em 0; }
.vignette strong, .response-text strong { font-weight: 700; }
.vignette h1, .vignette h2, .vignette h3,
.response-text h1, .response-text h2, .response-text h3 { margin: 0.6em 0 0.3em; font-size: 1em; color: #1a3a5c; }
.response-text ul, .response-text ol, .vignette ul, .vignette ol { padding-left: 1.4em; margin: 0.4em 0; }
.response-text hr, .vignette hr { border: none; border-top: 1px solid #ddd; margin: 10px 0; }
.response-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
.response-box { border: 1px solid #bbb; border-radius: 6px; padding: 16px; background: #fff; }
.response-box h4 { margin-top: 0; color: #1a3a5c; font-size: 1.05em; border-bottom: 1px solid #ddd; padding-bottom: 8px; }
.response-text { font-size: 0.88em; line-height: 1.65; color: #333; border: 1px solid #e8e8e8; padding: 10px; border-radius: 4px; background: #fdfdfd; }
.rating-section { margin-top: 14px; }
.rating-row { margin: 10px 0; }
.rating-row label { display: block; font-size: 0.88em; font-weight: bold; color: #333; margin-bottom: 4px; }
.rating-row .description { font-size: 0.80em; color: #666; margin-bottom: 6px; font-style: italic; }
.star-row { display: flex; gap: 8px; align-items: center; }
.star-row input[type=radio] { display: none; }
.star-row label.star { font-size: 1.6em; cursor: pointer; color: #ccc; transition: color 0.15s; }
.star-row input[type=radio]:checked ~ label.star,
.star-row label.star:hover,
.star-row label.star:hover ~ label.star { color: #f5a623; }
.star-row { flex-direction: row-reverse; justify-content: flex-end; }
.star-row label.star:hover,
.star-row label.star:hover ~ label.star { color: #f5a623; }
.comment-box { width: 100%; box-sizing: border-box; font-size: 0.85em; padding: 8px; border: 1px solid #ccc; border-radius: 4px; resize: vertical; min-height: 60px; margin-top: 8px; font-family: inherit; }
.instructions { background: #fff8e1; border: 1px solid #f5d060; border-radius: 8px; padding: 18px; margin: 20px 0; font-size: 0.92em; line-height: 1.6; }
.instructions h3 { margin-top: 0; color: #7a5c00; }
.progress-note { color: #888; font-size: 0.85em; text-align: right; margin-bottom: 6px; }
.submit-section { background: #1a3a5c; color: white; padding: 24px; border-radius: 8px; margin-top: 40px; text-align: center; }
.submit-section h2 { color: white; margin-top: 0; }
button.submit-btn { background: #f5a623; color: #1a3a5c; border: none; padding: 14px 32px; font-size: 1.1em; font-weight: bold; border-radius: 6px; cursor: pointer; margin-top: 12px; }
button.submit-btn:hover { background: #e09510; }
.overall-section { background: #f5f5f5; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; }
@media print { .response-text { max-height: none; overflow: visible; } }
"""

JS = """
function collectRatings() {
    const data = { timestamp: new Date().toISOString(), cases: [] };
    const caseBlocks = document.querySelectorAll('.case-block');
    caseBlocks.forEach(block => {
        const caseId = block.dataset.caseid;
        const caseData = { case_id: caseId, responses: {} };
        ['A', 'B'].forEach(resp => {
            caseData.responses[resp] = { ratings: {}, comment: '' };
            ['accuracy','completeness','reasoning','reasoning_transparency','calibration','utility'].forEach(dim => {
                const name = `${caseId}_${resp}_${dim}`;
                const checked = block.querySelector(`input[name="${name}"]:checked`);
                caseData.responses[resp].ratings[dim] = checked ? parseInt(checked.value) : null;
            });
            const comment = block.querySelector(`textarea[name="${caseId}_${resp}_comment"]`);
            if (comment) caseData.responses[resp].comment = comment.value;
        });
        // Overall preference
        const pref = block.querySelector(`input[name="${caseId}_preference"]:checked`);
        caseData.preference = pref ? pref.value : null;
        data.cases.push(caseData);
    });
    // Overall comments
    const overall = document.getElementById('overall_comments');
    if (overall) data.overall_comments = overall.value;
    return data;
}

function downloadResults() {
    const data = collectRatings();
    const missing = [];
    data.cases.forEach(c => {
        ['A','B'].forEach(r => {
            Object.entries(c.responses[r].ratings).forEach(([dim, val]) => {
                if (val === null) missing.push(`${c.case_id}-${r}-${dim}`);
            });
        });
    });
    if (missing.length > 0 && !confirm(`${missing.length} rating(s) not filled in. Download anyway?`)) return;
    const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'clinician_ratings.json';
    a.click();
    URL.revokeObjectURL(url);
}

// Star rating interaction fix
document.addEventListener('DOMContentLoaded', function() {
    // Render markdown in all response and vignette divs
    if (typeof marked !== 'undefined') {
        marked.setOptions({ breaks: true });
        document.querySelectorAll('[data-markdown]').forEach(el => {
            el.innerHTML = marked.parse(el.getAttribute('data-markdown'));
        });
    }

    document.querySelectorAll('.star-row').forEach(row => {
        const labels = row.querySelectorAll('label.star');
        labels.forEach((label, i) => {
            label.addEventListener('mouseenter', () => {
                // highlight this and all lower
                labels.forEach((l, j) => {
                    l.style.color = j >= (labels.length - 1 - i) ? '#f5a623' : '#ccc';
                });
            });
            label.addEventListener('mouseleave', () => {
                const checked = row.querySelector('input:checked');
                labels.forEach((l, j) => {
                    if (checked) {
                        const checkedVal = parseInt(checked.value);
                        l.style.color = (j >= labels.length - checkedVal) ? '#f5a623' : '#ccc';
                    } else {
                        l.style.color = '#ccc';
                    }
                });
            });
        });
    });
});
"""


def _star_rating_html(name: str, label: str, description: str) -> str:
    stars = ""
    for val in range(5, 0, -1):
        stars += f'<input type="radio" id="{name}_{val}" name="{name}" value="{val}">'
        stars += f'<label class="star" for="{name}_{val}" title="{val} star">★</label>'
    return f"""
    <div class="rating-row">
      <label>{label}</label>
      <div class="description">{description}</div>
      <div class="star-row">{stars}</div>
    </div>"""


def _response_block_html(case_id: str, letter: str, response_text: str) -> str:
    ratings_html = ""
    for key, label, desc in RATING_CRITERIA:
        name = f"{case_id}_{letter}_{key}"
        ratings_html += _star_rating_html(name, label, desc)

    return f"""
    <div class="response-box">
      <h4>Response {letter}</h4>
      <div class="response-text" data-markdown="{_html.escape(response_text)}"></div>
      <div class="rating-section">
        <strong style="font-size:0.9em; color:#555;">Rate Response {letter}:</strong>
        {ratings_html}
        <div class="rating-row">
          <label>Comments (optional)</label>
          <textarea class="comment-box" name="{case_id}_{letter}_comment" 
                    placeholder="Any observations about this response..."></textarea>
        </div>
      </div>
    </div>"""


def _preference_html(case_id: str) -> str:
    return f"""
    <div style="margin-top:16px; padding:12px; background:#f0f4f8; border-radius:6px;">
      <strong>Overall preference for this case:</strong>
      <div style="margin-top:8px; display:flex; gap:20px; font-size:0.92em;">
        <label><input type="radio" name="{case_id}_preference" value="A"> Response A is clearly better</label>
        <label><input type="radio" name="{case_id}_preference" value="A_slight"> Response A is slightly better</label>
        <label><input type="radio" name="{case_id}_preference" value="equal"> About equal</label>
        <label><input type="radio" name="{case_id}_preference" value="B_slight"> Response B is slightly better</label>
        <label><input type="radio" name="{case_id}_preference" value="B"> Response B is clearly better</label>
      </div>
    </div>"""


def generate_review_sheet(
    responses_path: Path,
    output_dir: Path,
    model_key: str = "bge",
    seed: int = 42,
    task: str = "explainability",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    df = pd.read_csv(responses_path, quoting=0, on_bad_lines="warn")

    # If combined file, filter to one model
    if "model_key" in df.columns:
        avail = df["model_key"].unique().tolist()
        if model_key not in avail:
            model_key = avail[0]
            print(f"[Review] model_key not found, using: {model_key}")
        df = df[df["model_key"] == model_key].copy()

    # Select task-specific response columns
    no_rag_col = f"{task}_no_rag"
    rag_col = f"{task}_rag"
    # Fall back to legacy column names if task-specific ones don't exist
    if no_rag_col not in df.columns:
        if "response_no_rag" in df.columns:
            no_rag_col, rag_col = "response_no_rag", "response_rag"
            print(f"[Review] Falling back to legacy columns: {no_rag_col} / {rag_col}")
        else:
            raise ValueError(f"Cannot find columns '{no_rag_col}' or 'response_no_rag' in {responses_path}")

    task_label = "Explainability" if task == "explainability" else "Teaching"
    print(f"[Review] Task: {task_label} | Columns: {no_rag_col} / {rag_col}")
    print(f"[Review] Generating review sheet for {len(df)} cases (model: {model_key})")

    # Build blinded assignment: randomly assign no_rag -> A or B per case
    key_map: dict[str, dict] = {}
    blinded_rows = []

    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        if random.random() < 0.5:
            resp_a_text = str(row[no_rag_col])
            resp_b_text = str(row[rag_col])
            resp_a_condition = "no_rag"
            resp_b_condition = "rag"
        else:
            resp_a_text = str(row[rag_col])
            resp_b_text = str(row[no_rag_col])
            resp_a_condition = "rag"
            resp_b_condition = "no_rag"

        key_map[case_id] = {
            "A": resp_a_condition,
            "B": resp_b_condition,
            "model_key": model_key,
            "task": task,
        }

        blinded_rows.append({
            "case_id": case_id,
            "chart_text": str(row["chart_text"]),
            "gold_label": str(row.get("gold_label", "")),
            "notes": str(row.get("notes", "")),
            "response_A": resp_a_text,
            "response_B": resp_b_text,
        })

    # Save key (keep hidden from reviewer)
    key_path = output_dir / f"review_key_{task}.json"
    key_path.write_text(json.dumps({"seed": seed, "model_key": model_key, "task": task, "assignments": key_map}, indent=2))
    print(f"[Review] Saved key (DO NOT SHARE WITH REVIEWER): {key_path}")

    blinded_df = pd.DataFrame(blinded_rows)
    blinded_csv_path = output_dir / f"review_sheet_{task}.csv"
    blinded_df[["case_id", "chart_text", "response_A", "response_B"]].to_csv(blinded_csv_path, index=False)
    print(f"[Review] Saved blinded CSV: {blinded_csv_path}")

    # Generate HTML
    case_blocks_html = ""
    for i, brow in enumerate(blinded_rows, 1):
        case_id = brow["case_id"]
        chart_text = brow["chart_text"]
        notes = brow["notes"]

        case_blocks_html += f"""
    <div class="case-block" data-caseid="{case_id}">
      <div class="case-header">
        <h2>Case {i} of {len(blinded_rows)} &nbsp;|&nbsp; ID: {case_id}</h2>
      </div>
      <p class="progress-note">Case {i}/{len(blinded_rows)}</p>
      <h3>Patient Vignette</h3>
      <div class="vignette" data-markdown="{_html.escape(chart_text)}"></div>
      <h3>Evaluate the Two Responses</h3>
      <p style="font-size:0.88em; color:#555;">Read both responses, then rate each independently using the 1–5 star scales below.</p>
      <div class="response-grid">
        {_response_block_html(case_id, 'A', brow['response_A'])}
        {_response_block_html(case_id, 'B', brow['response_B'])}
      </div>
      {_preference_html(case_id)}
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rheumatology AI — Blinded Clinician Review</title>
<style>{CSS}</style>
</head>
<body>
<h1>🩺 Rheumatology AI Diagnostic Response Review</h1>

<div class="instructions">
  <h3>Instructions for the Reviewer</h3>
  <p><strong>What you are evaluating:</strong> For each patient case, you will see two AI-generated diagnostic responses (Response A and Response B). 
  The responses were generated under different conditions, but you do not know which is which. Please evaluate them independently and blindly.</p>
  <p><strong>How to rate:</strong> Use the 1–5 star scales for each dimension. Rate each response on its own merits before comparing. <em>Reasoning Transparency</em> asks whether you can trace the logic step by step. <em>Calibration</em> asks whether the stated confidence matches how strong the evidence actually is.</p>
  <ul>
    <li><strong>1 star</strong> = Poor / unacceptable</li>
    <li><strong>2 stars</strong> = Below average</li>
    <li><strong>3 stars</strong> = Acceptable / average</li>
    <li><strong>4 stars</strong> = Good</li>
    <li><strong>5 stars</strong> = Excellent</li>
  </ul>
  <p><strong>When finished:</strong> Click the <em>Download Ratings</em> button at the bottom. Send the downloaded <code>clinician_ratings.json</code> file to the study team.</p>
  <p style="color:#c0392b;"><strong>Important:</strong> Do not discuss with colleagues or look up cases during review. This is a blinded evaluation.</p>
</div>

{case_blocks_html}

<div class="submit-section">
  <h2>Submit Your Ratings</h2>
  <p>When you have rated all cases, click below to download your ratings file.</p>
  <button class="submit-btn" onclick="downloadResults()">⬇ Download Ratings (clinician_ratings.json)</button>

</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
<script>{JS}</script>
</body>
</html>"""

    html_path = output_dir / f"review_sheet_{task}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[Review] Saved HTML review sheet: {html_path}")
    print(f"\nNext step:")
    print(f"  1. Open {html_path} in a browser")
    print(f"  2. Send to clinician reviewer")
    print(f"  3. Receive clinician_ratings.json back")
    print(f"  4. Run: python src/score_results.py --ratings clinician_ratings.json")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate blinded clinician review sheet")
    p.add_argument("--input", default="results/all_responses.csv", help="Responses CSV")
    p.add_argument("--output-dir", default="results", help="Output directory")
    p.add_argument("--model", default="minilm", help="Which model's responses to use (minilm or bge)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for A/B assignment")
    p.add_argument("--task", default="explainability", choices=["explainability", "teaching"],
                   help="Which task to generate review sheet for")
    args = p.parse_args()

    generate_review_sheet(
        responses_path=Path(args.input),
        output_dir=Path(args.output_dir),
        model_key=args.model,
        seed=args.seed,
        task=args.task,
    )


if __name__ == "__main__":
    main()