---
output:
  pdf_document: default
  html_document: default
---
# RAG vs No-RAG: Clinician Evaluation Results (Completeness)

**Embedding model:** minilm  
**Cases evaluated:** 30  

## Completeness Score (1–5 scale)

| Dimension | RAG Mean | No-RAG Mean | Delta | Test | Result |
|---|---:|---:|---:|---|---|
| Differential Completeness | 4.03 | 4.07 | -0.03 | wilcoxon_signed_rank | p=0.784 (ns) |

## Overall Response Preference

- Prefer RAG: **9** cases
- Prefer No-RAG: **11** cases
- Equal: **10** cases
- Mean RAG preference score: **-0.05** (positive = RAG preferred)

## Interpretation

A positive completeness delta indicates the ontology-grounded response was rated higher by the clinician. Statistical significance at p<0.05 is marked with *.

**Note:** With small N, effect sizes are more meaningful than p-values. Consider the clinical magnitude of any delta alongside statistical tests.

## Comments from Reviewer
