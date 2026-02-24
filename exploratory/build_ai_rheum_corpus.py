#!/usr/bin/env python3
"""Build the AI-RHEUM ontology corpus JSONL from BioPortal.

This is intentionally config-agnostic: it does NOT require get_config(...).
It reads label IDs from data/ai_rheum_label_set.json and writes:
  data/ai_rheum_corpus.jsonl

Usage:
  python scripts/build_ai_rheum_corpus.py

Env:
  BIOPORTAL_API_KEY (recommended)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

# Ensure we can import from src/ when run from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from classes.corpus import ensure_corpus  # noqa: E402


LABEL_SET_PATH = Path("data/ai_rheum_label_set.json")
OUTPUT_PATH = Path("data/ai_rheum_corpus.jsonl")
ACRONYM = "AI-RHEUM"


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()

    if not LABEL_SET_PATH.exists():
        print(f"Missing label set: {LABEL_SET_PATH}", file=sys.stderr)
        return 2

    raw = json.loads(LABEL_SET_PATH.read_text(encoding="utf-8"))
    label_ids = list(raw.get("labels") or [])

    if not label_ids:
        print(f"No labels found in {LABEL_SET_PATH}", file=sys.stderr)
        return 3

    records = ensure_corpus(
        acronym=ACRONYM,
        label_ids=label_ids,
        output_path=OUTPUT_PATH,
        prefer_bioportal=True,
    )

    print(f"Wrote/verified: {OUTPUT_PATH} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
