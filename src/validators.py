"""Validators for synthetic datasets and label sets.

These are lightweight checks intended to keep evaluation mechanical:
- Ensure labels are from the constrained label set ∪ NONE
- Ensure seed cases meet size + formatting constraints

This module is intentionally dependency-light (pandas optional but recommended).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence


Register = Literal["chart", "clinical_note", "colloquial", "public_health"]


@dataclass(frozen=True)
class LabelSet:
    labels: tuple[str, ...]
    none_label: str = "NONE"

    @property
    def allowed(self) -> set[str]:
        return set(self.labels) | {self.none_label}


ALLOWED_REGISTERS: tuple[Register, ...] = (
    "chart",
    "clinical_note",
    "colloquial",
    "public_health",
)


def load_label_set(path: str | Path) -> LabelSet:
    path = Path(path)
    data = json.loads(path.read_text())
    labels = tuple(data.get("labels") or [])
    none_label = data.get("none_label") or "NONE"
    return LabelSet(labels=labels, none_label=none_label)


def validate_label_set(label_set: LabelSet, *, min_labels: int = 5, max_labels: int = 8) -> None:
    if not label_set.none_label or not isinstance(label_set.none_label, str):
        raise ValueError("label_set.none_label must be a non-empty string")

    if label_set.none_label != "NONE":
        raise ValueError(f"label_set.none_label must be the exact string 'NONE'; got {label_set.none_label!r}")

    unique = list(dict.fromkeys(label_set.labels))
    if len(unique) != len(label_set.labels):
        raise ValueError("label_set.labels contains duplicates")

    if not (min_labels <= len(label_set.labels) <= max_labels):
        raise ValueError(f"label_set.labels must have {min_labels}–{max_labels} entries; got {len(label_set.labels)}")

    for label in label_set.labels:
        if not isinstance(label, str) or not label:
            raise ValueError("label_set.labels must be non-empty strings")
        if label == label_set.none_label:
            raise ValueError("label_set.labels must not contain the none_label")


def _line_count(text: str) -> int:
    return len(str(text).splitlines())


def _sentence_count(text: str) -> int:
    """Approximate line count for single-line charts.

    Some existing datasets store chart_text as one long line with sentence
    boundaries instead of explicit newline separators.
    """

    import re

    s = str(text).strip()
    if not s:
        return 0

    # Split on common sentence terminators.
    # Use a permissive split so we still count units when punctuation is
    # followed by commas or end-of-string (common in CSV-embedded text).
    parts = [p for p in re.split(r"[.!?]+", s) if p.strip()]
    return len(parts)


def _chart_text_unit_count(text: str) -> int:
    """Count 'units' in chart_text.

    Preferred: explicit newline-separated lines.
    Fallback: sentence count for single-line text.
    """

    lines = _line_count(text)
    if lines >= 2:
        return lines
    return _sentence_count(text)


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    lowered = str(text).lower()
    return any(k in lowered for k in keywords)


def validate_seed_cases_csv(
    csv_path: str | Path,
    *,
    label_set_path: str | Path = "data/label_set.json",
    min_rows: int = 15,
    max_rows: int = 20,
    min_lines: int = 5,
    max_lines: int = 12,
    require_overlap_symptoms_for_none: bool = True,
) -> None:
    """Validate the seeded dataset contract.

    Enforces:
    - row count 15–20
    - required columns: case_id, register, chart_text, gold_label
    - register in ALLOWED_REGISTERS
    - chart_text has 5–12 lines
    - gold_label in label_set ∪ NONE
    - NONE cases contain overlapping symptom keywords (heuristic)
    """

    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required to validate CSV files") from e

    label_set = load_label_set(label_set_path)
    validate_label_set(label_set)

    df = pd.read_csv(csv_path)

    if not (min_rows <= len(df) <= max_rows):
        raise ValueError(f"Seed dataset must have {min_rows}–{max_rows} rows; got {len(df)}")

    required = {"case_id", "register", "chart_text", "gold_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # register checks
    bad_register = df[~df["register"].isin(ALLOWED_REGISTERS)]
    if not bad_register.empty:
        raise ValueError(f"Invalid register values: {sorted(bad_register['register'].unique().tolist())}")

    # label checks
    bad_labels = df[~df["gold_label"].isin(label_set.allowed)]
    if not bad_labels.empty:
        rows = bad_labels[["case_id", "gold_label"]].to_dict(orient="records")
        raise ValueError(f"Found gold_label values not in label_set.json: {rows}")

    # line count checks
    line_counts = df["chart_text"].astype(str).apply(_line_count)
    bad_lines = df[(line_counts < min_lines) | (line_counts > max_lines)]
    if not bad_lines.empty:
        rows = bad_lines[["case_id"]].assign(line_count=line_counts).to_dict(orient="records")
        raise ValueError(f"Found chart_text outside {min_lines}–{max_lines} lines: {rows}")

    # heuristic overlap symptom check for NONE
    if require_overlap_symptoms_for_none:
        overlap_keywords = (
            "hoarse",
            "hoarseness",
            "dysphagia",
            "neck",
            "thyroid",
            "cervical",
            "lymph",
            "node",
            "globus",
        )
        none_df = df[df["gold_label"] == label_set.none_label]
        if not none_df.empty:
            ok = none_df["chart_text"].astype(str).apply(lambda t: _contains_any(t, overlap_keywords))
            if not bool(ok.all()):
                bad = none_df.loc[~ok, ["case_id"]].to_dict(orient="records")
                raise ValueError(f"NONE cases missing overlap symptom keywords (heuristic): {bad}")


def validate_rows_against_label_set(
    gold_labels: Iterable[str],
    *,
    label_set_path: str | Path = "data/label_set.json",
) -> None:
    """Convenience validator for label columns in any dataset."""

    label_set = load_label_set(label_set_path)
    validate_label_set(label_set)

    bad = sorted({l for l in gold_labels if l not in label_set.allowed})
    if bad:
        raise ValueError(f"Labels not in allowed set: {bad}")


def validate_synthetic_charts_csv(
    csv_path: str | Path,
    *,
    label_set_path: str | Path = "data/label_set.json",
    min_rows: int = 60,
    min_units: int = 5,
    max_units: int = 12,
    require_overlap_symptoms_for_none: bool = True,
) -> None:
    """Validate data/synthetic_charts.csv.

    This validator is intentionally compatible with the current repository data:
    - Accepts either `case_id` or `chart_id` as an identifier column.
    - Interprets the 5–12 'lines' requirement as either:
      (a) newline-separated lines, or
      (b) sentence-separated units if the text is a single line.
    """

    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required to validate CSV files") from e

    label_set = load_label_set(label_set_path)
    validate_label_set(label_set)

    df = pd.read_csv(csv_path)

    if len(df) < min_rows:
        raise ValueError(f"Synthetic charts dataset must have ≥ {min_rows} rows; got {len(df)}")

    required = {"chart_text", "gold_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "case_id" not in df.columns and "chart_id" not in df.columns:
        raise ValueError("Expected an identifier column: case_id or chart_id")

    # label checks
    bad_labels = df[~df["gold_label"].isin(label_set.allowed)]
    if not bad_labels.empty:
        cols = [c for c in ("case_id", "chart_id", "gold_label") if c in bad_labels.columns]
        rows = bad_labels[cols].head(10).to_dict(orient="records")
        raise ValueError(f"Found gold_label values not in label_set.json (showing up to 10): {rows}")

    # units (lines or sentences) checks
    unit_counts = df["chart_text"].astype(str).apply(_chart_text_unit_count)
    bad_units = df[(unit_counts < min_units) | (unit_counts > max_units)]
    if not bad_units.empty:
        id_col = "case_id" if "case_id" in bad_units.columns else "chart_id"
        rows = bad_units[[id_col]].assign(unit_count=unit_counts).head(10).to_dict(orient="records")
        raise ValueError(f"Found chart_text outside {min_units}–{max_units} units (showing up to 10): {rows}")

    if require_overlap_symptoms_for_none:
        overlap_keywords = (
            "hoarse",
            "hoarseness",
            "dysphagia",
            "neck",
            "thyroid",
            "cervical",
            "lymph",
            "node",
            "globus",
        )
        none_df = df[df["gold_label"] == label_set.none_label]
        if not none_df.empty:
            ok = none_df["chart_text"].astype(str).apply(lambda t: _contains_any(t, overlap_keywords))
            if not bool(ok.all()):
                id_col = "case_id" if "case_id" in none_df.columns else "chart_id"
                bad = none_df.loc[~ok, [id_col]].head(10).to_dict(orient="records")
                raise ValueError(f"NONE cases missing overlap symptom keywords (heuristic; showing up to 10): {bad}")
