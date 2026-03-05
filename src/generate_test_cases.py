#!/usr/bin/env python3
"""scripts/generate_test_cases_v2.py

Extends v1 with three additions:
  1. AMBIGUOUS_CASES  — differential-diagnosis vignettes for the teaching task
  2. Dual prompts     — EXPLAINABILITY_PROMPT and TEACHING_PROMPT, both applied
                        to every case at inference time (not at generation time)
  3. ontology_support_level — 'rich' / 'partial' / 'none' flag written to CSV
                              so analysis can stratify results without case filtering

USAGE
-----
Generate cases (same CLI as v1):
    python generate_test_cases_v2.py
    python generate_test_cases_v2.py --output data/test_cases_v2.csv --delay 0.5

Run inference (separate script, see eval_runner.py):
    The prompts below are imported by the eval runner — not used at generation time.

OUTPUT FORMAT (CSV columns)
---------------------------
case_id | chart_text | gold_label | category | notes | ontology_support_level
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

AIR_BASE = "http://purl.bioontology.org/ontology/AIR/"

# ---------------------------------------------------------------------------
# 1. CASES  (unchanged from v1 — import and extend)
# ---------------------------------------------------------------------------

# Original 28 cases — paste your existing CASES list here or import from v1.
# Shown abbreviated; replace with full list.
CASES = [
    ("Rheumatoid Arthritis",            "classic",   AIR_BASE + "DXRA",  "Classic seropositive RA"),
    ("Rheumatoid Arthritis",            "atypical",  AIR_BASE + "DXRA",  "Seronegative RA, atypical presentation"),
    ("Rheumatoid Arthritis",            "early",     AIR_BASE + "DXRA",  "Early undifferentiated — may be RA"),
    ("Systemic Lupus Erythematosus",    "classic",   AIR_BASE + "DXSLE", "Classic multi-system SLE"),
    ("Systemic Lupus Erythematosus",    "atypical",  AIR_BASE + "DXSLE", "SLE without typical rash"),
    ("Systemic Lupus Erythematosus",    "early",     AIR_BASE + "DXSLE", "Early incomplete SLE criteria"),
    ("Gout",                            "classic",   AIR_BASE + "DXGT",  "Classic acute podagra"),
    ("Gout",                            "atypical",  AIR_BASE + "DXGT",  "Gout in atypical joint"),
    ("Gout",                            "early",     AIR_BASE + "DXGT",  "First episode, uric acid borderline"),
    ("Psoriatic Arthritis",             "classic",   AIR_BASE + "DXPSO", "Classic PsA with skin disease"),
    ("Psoriatic Arthritis",             "atypical",  AIR_BASE + "DXPSO", "PsA without visible psoriasis"),
    ("Psoriatic Arthritis",             "early",     AIR_BASE + "DXPSO", "Inflammatory arthritis, psoriasis family history only"),
    ("Ankylosing Spondylitis",          "classic",   AIR_BASE + "DXANK", "Classic AS with sacroiliitis"),
    ("Ankylosing Spondylitis",          "atypical",  AIR_BASE + "DXANK", "AS presenting with peripheral arthritis"),
    ("Ankylosing Spondylitis",          "early",     AIR_BASE + "DXANK", "Early axial SpA, MRI not yet diagnostic"),
    ("Polymyalgia Rheumatica",          "classic",   AIR_BASE + "DXPMR", "Classic PMR with steroid response"),
    ("Polymyalgia Rheumatica",          "atypical",  AIR_BASE + "DXPMR", "PMR with normal ESR"),
    ("Polymyalgia Rheumatica",          "early",     AIR_BASE + "DXPMR", "Proximal stiffness, workup incomplete"),
    ("CPPD mimicking gout",             "mimicker",  "NONE",             "Pseudogout — not gout despite acute monoarthritis"),
    ("Reactive arthritis following GI infection mimicking ankylosing spondylitis",
                                        "mimicker",  "NONE",             "Reactive arthritis — not in 6-label set"),
    ("Drug-induced lupus from hydralazine or procainamide mimicking SLE",
                                        "mimicker",  "NONE",             "Drug-induced lupus — not true SLE"),
    ("Osteoarthritis mimicking rheumatoid arthritis",
                                        "mimicker",  "NONE",             "OA — mechanical not inflammatory"),
    ("Adult-onset Still's disease mimicking systemic lupus",
                                        "mimicker",  "NONE",             "Still's disease — not in label set"),
    ("Fibromyalgia",                    "none",      "NONE",             "Fibromyalgia — no inflammatory arthritis"),
    ("Sarcoid arthritis",               "none",      "NONE",             "Sarcoidosis — not in label set"),
    ("Septic arthritis",                "none",      "NONE",             "Bacterial arthritis — not in label set"),
    ("Early undifferentiated inflammatory arthritis that does not yet meet criteria for any specific diagnosis",
                                        "none",      "NONE",             "Genuinely ambiguous early presentation"),
]

# ---------------------------------------------------------------------------
# 2. AMBIGUOUS CASES  (new — differential pairs for the teaching task)
#
# gold_label is a pipe-separated pair of the two plausible diagnoses.
# The eval runner treats these as "correct" if the model names either or both.
# They stress-test the teaching task specifically: the ontology WHY sections
# for each diagnosis should provide contrasting reasoning the model can surface.
#
# Chosen pairs based on ontology rich/partial support:
#   RA vs PSA  — both have rich structured WHY; share joint involvement
#   PMR vs RA  — both affect older patients; stiffness is the shared feature
#   PMR vs FIB — critical clinical distinction; PMR has ESR/steroid response
# ---------------------------------------------------------------------------

AMBIGUOUS_CASES = [
    (
        "Early inflammatory polyarthritis that could be either Rheumatoid Arthritis "
        "or Psoriatic Arthritis — patient has symmetric small joint involvement but "
        "also occasional DIP swelling and a family history of psoriasis with no "
        "visible skin lesions currently",
        "ambiguous",
        AIR_BASE + "DXRA|" + AIR_BASE + "DXPSO",
        "RA vs PSA — shared features, DIP and family Hx favour PSA but serology pending",
    ),
    (
        "Bilateral proximal stiffness in a 68-year-old that could be either "
        "Polymyalgia Rheumatica or Rheumatoid Arthritis — shoulder and hip girdle "
        "aching, 90-minute morning stiffness, ESR 44, RF weakly positive at 1:40, "
        "no obvious synovitis on exam",
        "ambiguous",
        AIR_BASE + "DXPMR|" + AIR_BASE + "DXRA",
        "PMR vs RA — proximal stiffness and weak RF; key distinguisher is synovitis pattern",
    ),
    (
        "65-year-old woman with widespread aching, fatigue, and bilateral shoulder "
        "and hip pain for 4 months. ESR 18, CRP 4. No synovitis. Sleep disturbed. "
        "Could be Polymyalgia Rheumatica with normal inflammatory markers or "
        "fibromyalgia",
        "ambiguous",
        AIR_BASE + "DXPMR|NONE",
        "PMR vs fibromyalgia — normal ESR makes PMR uncertain; critical teaching case",
    ),
]

ALL_CASES = CASES + AMBIGUOUS_CASES


# ---------------------------------------------------------------------------
# 3. PROMPTS  (imported by eval_runner at inference time, not used here)
#
# Both prompts accept two template variables:
#   {vignette}  — the chart_text from the CSV
#   {context}   — RAG-retrieved ontology chunks (empty string for no-RAG arm)
#
# The RAG arm passes retrieved text; the no-RAG arm passes "".
# ---------------------------------------------------------------------------

# ── Explainability prompt ────────────────────────────────────────────────────
# Goal: grounded, traceable reasoning. Score: finding citation accuracy,
# correct dx, specificity of justification.
EXPLAINABILITY_PROMPT = """\
You are a rheumatology clinical decision support system.

{context_block}
Patient vignette:
{vignette}

Task: Identify the single most likely rheumatological diagnosis. Then explain \
your reasoning by listing the specific clinical findings from the vignette that \
support this diagnosis, and for each finding state why it is diagnostically \
significant.

If none of the following diagnoses fits, output NONE:
Rheumatoid Arthritis | Systemic Lupus Erythematosus | Gout | \
Psoriatic Arthritis | Ankylosing Spondylitis | Polymyalgia Rheumatica

Format your response as:
DIAGNOSIS: <name>
REASONING:
- <finding from vignette>: <why this finding supports the diagnosis>
- ...
"""

# ── Teaching prompt ──────────────────────────────────────────────────────────
# Goal: pedagogical completeness. Score: covers WHAT each finding is,
# WHY it matters, HOW to assess it. Should mention differential.
TEACHING_PROMPT = """\
You are a rheumatology attending teaching a medical student.

{context_block}
Patient vignette:
{vignette}

Task: Identify the most likely diagnosis and teach the student about this case. \
For each key clinical finding in the vignette:
  1. Explain what the finding is (define it clearly for a student)
  2. Explain why it matters diagnostically (what diseases does it point toward \
or away from, and why)
  3. Explain how a clinician assesses it in practice

Also briefly note one or two findings that could suggest an alternative diagnosis \
and explain why you favour your primary diagnosis over that alternative.

If none of the following diagnoses fits, state that and explain what the \
presentation does suggest instead:
Rheumatoid Arthritis | Systemic Lupus Erythematosus | Gout | \
Psoriatic Arthritis | Ankylosing Spondylitis | Polymyalgia Rheumatica
"""

# Helper: formats the context block consistently for both prompts
def format_context_block(retrieved_chunks: list[str]) -> str:
    """
    retrieved_chunks: list of .text strings from corpus records.
    Returns a formatted block to splice into either prompt,
    or empty string if the list is empty (no-RAG arm).
    """
    if not retrieved_chunks:
        return ""
    joined = "\n\n---\n\n".join(retrieved_chunks)
    return (
        f"The following ontology excerpts may be relevant. "
        f"Use them to ground your reasoning where applicable:\n\n"
        f"{joined}\n\n"
    )


# ---------------------------------------------------------------------------
# 4. ONTOLOGY SUPPORT LEVEL
#
# Assigned at CSV-generation time (not inference time) based on the corpus.
# 'rich'    — at least one linked finding with both WHY and HOW sections
# 'partial' — linked findings exist but none has full WHY+HOW
# 'none'    — gold label is NONE, or no corpus findings link to this dx
# ---------------------------------------------------------------------------

def _load_support_index(corpus_path: str | Path) -> dict[str, str]:
    """
    Pre-build a dx_code → support_level dict from the corpus.
    Called once at startup; returns a lookup used by get_ontology_support_level().
    """
    if not Path(corpus_path).exists():
        return {}

    dx_has_rich: dict[str, bool] = {}
    dx_has_partial: dict[str, bool] = {}

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["chunk_type"] != "finding_chunk":
                continue
            has_rich = bool(r.get("definition_why")) and bool(r.get("definition_how"))
            for dx in r.get("linked_dx_codes", []):
                if has_rich:
                    dx_has_rich[dx] = True
                else:
                    dx_has_partial.setdefault(dx, True)

    index: dict[str, str] = {}
    all_dx = set(dx_has_rich) | set(dx_has_partial)
    for dx in all_dx:
        index[dx] = "rich" if dx_has_rich.get(dx) else "partial"
    return index


def get_ontology_support_level(
    gold_label: str,
    support_index: dict[str, str],
) -> str:
    """
    Returns 'rich', 'partial', or 'none'.

    gold_label may be:
      - a single AIR URI         → look up single dx code
      - a pipe-separated pair    → take the higher of the two
      - 'NONE'                   → always 'none'
    """
    if gold_label == "NONE":
        return "none"

    # Extract dx codes from URIs (handles both single and pipe-separated)
    codes = [uri.split("/")[-1] for uri in gold_label.split("|")]

    levels = [support_index.get(code, "none") for code in codes]
    rank = {"rich": 0, "partial": 1, "none": 2}
    return min(levels, key=lambda l: rank[l])


# ---------------------------------------------------------------------------
# Vignette generation  (unchanged from v1 — copied for self-containment)
# ---------------------------------------------------------------------------

CASE_WRITER_PROMPT = """\
You are an experienced rheumatologist writing patient vignettes for a clinical \
reasoning exam.

Write a realistic patient vignette for the following:
Diagnosis / scenario: {diagnosis}
Presentation type: {category}

Presentation type guidance:
- classic   : All expected features present. Diagnosis is clear.
- atypical  : Correct diagnosis but one or two expected features absent or unusual.
- early     : Too early to be definitive. Genuine diagnostic uncertainty.
- mimicker  : Write the case as the mimicking condition, not the disease it resembles.
- none      : Clearly outside the common inflammatory arthritides.
- ambiguous : The presentation genuinely fits two diagnoses. Include features of \
both. A definitive diagnosis cannot be made without further investigation. \
Do NOT favour one over the other.

Rules:
- Write as a clinician would document a real patient encounter
- Include specific values: exact lab numbers, joint counts, duration, age, sex
- Include at least one finding that could fit an alternative diagnosis
- Do not name any diagnosis anywhere in the vignette
- Output only the vignette text, no preamble or explanation
"""


def _detect_backend() -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    return "dry_run"


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip()


def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    m = genai.GenerativeModel("gemini-1.5-flash")
    return m.generate_content(prompt).text.strip()


def _call_dry_run(diagnosis: str, category: str) -> str:
    return (
        f"[DRY RUN] Patient presenting with symptoms consistent with "
        f"{diagnosis} ({category} presentation). Set an API key for real vignettes."
    )


def generate_vignette(
    diagnosis: str,
    category: str,
    backend: str,
    retry_delay: float = 2.0,
    max_retries: int = 3,
) -> str:
    if backend == "dry_run":
        return _call_dry_run(diagnosis, category)

    prompt = CASE_WRITER_PROMPT.format(diagnosis=diagnosis, category=category)
    for attempt in range(max_retries):
        try:
            if backend == "anthropic":
                return _call_anthropic(prompt)
            elif backend == "openai":
                return _call_openai(prompt)
            elif backend == "gemini":
                return _call_gemini(prompt)
        except Exception as e:
            print(f"  [LLM] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                return f"[ERROR: {e}]"
    return "[ERROR: max retries exceeded]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output",  default="data/test_cases_v2.csv")
    p.add_argument("--corpus",  default="data/ai_rheum_corpus_v3.jsonl",
                   help="Path to corpus JSONL — used only for ontology_support_level")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--delay",   type=float, default=0.5)
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    backend = "dry_run" if args.dry_run else _detect_backend()
    print(f"[Generator] Backend : {backend}")
    print(f"[Generator] Cases   : {len(ALL_CASES)} ({len(CASES)} standard + {len(AMBIGUOUS_CASES)} ambiguous)")

    support_index = _load_support_index(args.corpus)
    if support_index:
        print(f"[Generator] Corpus loaded for support-level flagging ({len(support_index)} dx codes)")
    else:
        print(f"[Generator] WARNING: corpus not found at {args.corpus} — support_level will be 'unknown'")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (diagnosis, category, gold_label, notes) in enumerate(ALL_CASES, 1):
        print(f"  [{i:02d}/{len(ALL_CASES)}] {category:10s} | {diagnosis[:55]}")

        vignette = (
            _call_dry_run(diagnosis, category)
            if args.dry_run
            else generate_vignette(diagnosis, category, backend)
        )
        if not args.dry_run:
            time.sleep(args.delay)

        support_level = (
            get_ontology_support_level(gold_label, support_index)
            if support_index
            else "unknown"
        )

        rows.append({
            "case_id":                f"gen-{i:03d}",
            "chart_text":             vignette,
            "gold_label":             gold_label,
            "category":               category,
            "notes":                  notes,
            "ontology_support_level": support_level,   # NEW
        })

    fieldnames = ["case_id", "chart_text", "gold_label", "category", "notes",
                  "ontology_support_level"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    support_counts = Counter(r["ontology_support_level"] for r in rows)
    cat_counts = Counter(r["category"] for r in rows)

    print(f"\n[Generator] Done — {output_path}")
    print(f"  Category breakdown : {dict(cat_counts)}")
    print(f"  Support levels     : {dict(support_counts)}")
    print(f"\n[Generator] Prompts exported:")
    print(f"  EXPLAINABILITY_PROMPT  — import from this module for the explainability eval arm")
    print(f"  TEACHING_PROMPT        — import from this module for the teaching eval arm")
    print(f"  format_context_block() — call with retrieved chunks list (empty list = no-RAG arm)")


if __name__ == "__main__":
    main()