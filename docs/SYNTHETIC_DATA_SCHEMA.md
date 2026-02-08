# Synthetic Data Schema

This project evaluates exact-match agreement on a **fully synthetic** dataset of short, chart-like texts. The dataset is stored as a CSV, and labels are constrained to a small, explicit set derived from an ontology plus a sentinel `NONE`.

## Files covered

- Dataset: [data/synthetic_charts.csv](../data/synthetic_charts.csv)
- Label set: [data/label_set.json](../data/label_set.json)

## Label space

### Allowed labels

The allowed label space is **exactly**:

- All strings in `data/label_set.json["labels"]` (5–8 entries recommended), plus
- The sentinel label `data/label_set.json["none_label"]` (must be the exact string `NONE`)

**Important:** Labels must match by **exact string equality**. If a row’s `gold_label` is not in the allowed set ∪ `{NONE}`, it must be treated as invalid for scoring.

### Definition of `NONE`

`NONE` means: the chart intentionally contains overlapping symptoms ("distractors") that resemble the in-domain condition, but the correct label is **not** one of the ontology labels in `labels`.

Guardrail: `NONE` cases must still contain distractor symptoms (e.g., hoarseness, dysphagia, neck swelling, fatigue), but with an alternative, non-domain explanation supported by the text.

## Dataset requirements

### Minimum size and balance targets

- **Minimum size:** ≥ 60 rows recommended for meaningful agreement comparisons
- **Balance target:** approximately 50% in-domain labels vs 50% `NONE`

### Text format: “5–12 lines”

The preferred format is 5–12 explicit newline-separated lines (i.e., `\n`).

For compatibility with the current checked-in dataset, a single-line `chart_text` is also allowed if it contains 5–12 **sentence-like units** (e.g., split on `.`, `!`, `?`).

### Privacy / identifiers

- No patient identifiers: no names, addresses, phone numbers, MRNs, or dates of birth.
- All content must be fully synthetic.

## CSV schema: data/synthetic_charts.csv

### Required columns

- `chart_text` (string)
  - Synthetic chart-like text meeting the 5–12 line/unit constraint
- `gold_label` (string)
  - Must be in `data/label_set.json["labels"]` or exactly `NONE`

### Optional columns (allowed)

- `case_id` (string) — recommended stable identifier
- `chart_id` (integer) — compatibility identifier (present in the current dataset)
- `age` (integer), `sex` (string), or other analysis fields

## Reproducibility policy

- **Random seed:** use a fixed seed (default `42`) for any synthetic generation intended for evaluation.
- **Determinism:** datasets used for scoring should be checked into `data/` and treated as immutable during an experiment run.
