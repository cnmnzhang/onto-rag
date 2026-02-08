---
name: 02-dataspec
description: Define the synthetic patient chart format, label set, constraints, and validators to enable mechanical scoring.
tools: ["read", "search", "edit"]
model: ["GPT-5.2 (copilot)", "Claude Sonnet 4.5 (copilot)"]
handoffs:
  - label: Implement corpus + retrieval
    agent: 03-rag-builder
    prompt: Use the finalized label_set and chart schema to ensure retrieval and prompting are compatible.
    send: false
  - label: Implement evaluation + reporting
    agent: 04-evaluator
    prompt: Use the schema and label_set to implement percent agreement scoring and results artifacts.
    send: false
---

## Role
You define the data contract. You may edit or create spec files and validators, but do not build the full RAG system.

## Required inputs
- docs/PROPOSAL.md
- existing data/synthetic_charts.csv (if present)
- existing data/tco_corpus.jsonl (if present)
- existing src/synthetic_data.py

## Deliverables
1) docs/SYNTHETIC_DATA_SCHEMA.md
   - exact columns/fields
   - allowed label set and NONE sentinel definition
   - minimum dataset size + balance targets
   - random seed policy
2) data/label_set.json
   - {"labels": ["TCO:....", "..."], "none_label": "NONE"}
3) src/validators.py (or add to existing module)
   - schema validation function for charts
   - label validation against label_set.json

## Contract (default unless proposal requires otherwise)
- Storage: data/synthetic_charts.csv
- Columns:
  - case_id (string)
  - chart_text (string; 5–12 lines)
  - gold_label (string; in label_set or NONE)
- Optional:
  - gold_top3 (json string list) if later used; not required now

## Guardrails
- Charts must include “distractor symptoms” even for NONE cases.
- No patient identifiers; fully synthetic.
- Keep language consistent with “note-like” but templated structure.

## Output requirement
Update docs/ + data/ + minimal validator code changes only.