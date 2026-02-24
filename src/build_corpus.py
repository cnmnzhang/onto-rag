#!/usr/bin/env python3
"""Build the AI-RHEUM ontology corpus JSONL from BioPortal.

This is intentionally config-agnostic: it does NOT require get_config(...).
It reads label IDs from data/ai_rheum_label_set.json and writes:
  data/ai_rheum_corpus.jsonl

Usage:
  python3 exploratory/build_corpus.py

Env:
  BIOPORTAL_API_KEY (recommended)
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

# Script-mode import bootstrapping.
if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.bootstrap import ensure_repo_on_sys_path  # type: ignore
else:  # pragma: no cover
    from .bootstrap import ensure_repo_on_sys_path

ensure_repo_on_sys_path()

from classes.corpus import ensure_corpus  # noqa: E402
from config.constants import DEFAULT_ONTOLOGY  # noqa: E402
from config.paths import AI_RHEUM_CORPUS_PATH, AI_RHEUM_LABEL_SET_PATH  # noqa: E402
from schemas import LabelSet  # noqa: E402


LABEL_SET_PATH = AI_RHEUM_LABEL_SET_PATH
OUTPUT_PATH = AI_RHEUM_CORPUS_PATH
ACRONYM = DEFAULT_ONTOLOGY


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()

    if not LABEL_SET_PATH.exists():
        print(f"Missing label set: {LABEL_SET_PATH}", file=sys.stderr)
        return 2

    label_set = LabelSet.from_path(LABEL_SET_PATH)
    label_ids = list(label_set.labels)

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
