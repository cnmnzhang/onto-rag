# Evaluation Summary

## Metrics

| Condition | Exact agreement | N | k | Backend |
|---|---:|---:|---:|---|
| No-RAG | 27.8% | 18 | 3 | dry_run |
| RAG(ai_rheum) | 44.4% | 18 | 3 | dry_run |

## Inferential (Paired t-test)

- Mean agreement delta (RAG - No-RAG): 16.67 points
- Improved / worse / unchanged: 6 / 3 / 9
- t-statistic: 0.9999999999999999
- p-value: 0.3313
- df: 17

## Parameters

- Ontology: ai_rheum
- Dataset: data/seed_cases_ai_rheum.csv
- Seed: 42
- Model: dry_run
- Embedding model: all-MiniLM-L6-v2
- Retrieval mode: tfidf
- Git commit: 24ce4277323e2db425a5345347f8b1d1cad36ecc

## Example Cases (5–10)

- seed-air-010: gold=NONE | no-rag=NONE | rag=NONE
- seed-air-012: gold=NONE | no-rag=http://purl.bioontology.org/ontology/AIR/DXPSO | rag=NONE
- seed-air-013: gold=NONE | no-rag=http://purl.bioontology.org/ontology/AIR/DXPSO | rag=NONE
- seed-air-014: gold=NONE | no-rag=http://purl.bioontology.org/ontology/AIR/DXPSO | rag=NONE
- seed-air-001: gold=http://purl.bioontology.org/ontology/AIR/DXRA | no-rag=http://purl.bioontology.org/ontology/AIR/DXPSO | rag=http://purl.bioontology.org/ontology/AIR/DXANK
- seed-air-002: gold=http://purl.bioontology.org/ontology/AIR/DXRA | no-rag=http://purl.bioontology.org/ontology/AIR/DXPSO | rag=http://purl.bioontology.org/ontology/AIR/DXANK
- seed-air-003: gold=http://purl.bioontology.org/ontology/AIR/DXSLE | no-rag=http://purl.bioontology.org/ontology/AIR/DXPSO | rag=http://purl.bioontology.org/ontology/AIR/DXRA
- seed-air-004: gold=http://purl.bioontology.org/ontology/AIR/DXSLE | no-rag=http://purl.bioontology.org/ontology/AIR/DXSLE | rag=http://purl.bioontology.org/ontology/AIR/DXRA

## Differential Diagnosis Snippets (RAG)

- seed-air-001: [{"label": "http://purl.bioontology.org/ontology/AIR/DXANK", "rationale": "Heuristic candidate based on rheumatologic diagnosis keywords"}, {"label": "http://purl.bioontology.org/ontology/AIR/DXRA", "rationale": "Heuristic candidate based on rheumatologic diagnosis keywords"}, {"label": "http://purl.bioontology.org/ontology/AIR/DXPSO", "rationale": "Heuristic candidate based on rheumatologic diagnosis keywords"}]
- seed-air-002: [{"label": "http://purl.bioontology.org/ontology/AIR/DXANK", "rationale": "Ankylosing Spondylitis has progressive joint involvement, especially in the spine and lower extremities."}, {"label": "http://purl.bioontology.org/ontology/AIR/DXPMR", "rationale": "Polymyalgia Rheumatica presents with myalgia and can involve multiple joints, including the wrists and forefeet."}]
- seed-air-003: [{"label": "http://purl.bioontology.org/ontology/AIR/DXANK", "rationale": "Heuristic candidate based on possible rheumatologic diagnosis indicators"}, {"label": "http://purl.bioontology.org/ontology/AIR/DXRA", "rationale": "Heuristic candidate based on possible rheumatologic diagnosis indicators"}, {"label": "http://purl.bioontology.org/ontology/AIR/DXPSO", "rationale": "Heuristic candidate based on possible rheumatologic diagnosis indicators"}]
- seed-air-004: [{"label": "http://purl.bioontology.org/ontology/AIR/DXANK", "rationale": "Heuristic candidate based on possible rheumatologic diagnosis indicators"}, {"label": "http://purl.bioontology.org/ontology/AIR/DXRA", "rationale": "Heuristic candidate based on possible rheumatologic diagnosis indicators"}, {"label": "http://purl.bioontology.org/ontology/AIR/DXPSO", "rationale": "Heuristic candidate based on possible rheumatologic diagnosis indicators"}]
- seed-air-005: [{"label": "NONE", "rationale": "Heuristic: no clear rheumatologic diagnosis indicators"}]
