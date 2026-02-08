# Evaluation Summary

## Metrics

| Condition | Exact agreement | N | k | Backend |
|---|---:|---:|---:|---|
| No-RAG | 72.2% | 18 | 3 | huggingface |
| RAG(ai_rheum) | 61.1% | 18 | 3 | huggingface |

## Parameters

- Ontology: ai_rheum
- Dataset: data/seed_cases_ai_rheum.csv
- Seed: 42
- Model: Qwen/Qwen2.5-1.5B-Instruct
- Git commit: fa547115bf844ae663dd3e73c7d910fa3f4f9299

## Example Cases (5–10)

- seed-air-001: gold=http://purl.bioontology.org/ontology/AIR/RA | no-rag=http://purl.bioontology.org/ontology/AIR/RA | rag=http://purl.bioontology.org/ontology/AIR/RA
- seed-air-004: gold=http://purl.bioontology.org/ontology/AIR/SLE | no-rag=http://purl.bioontology.org/ontology/AIR/SLE | rag=http://purl.bioontology.org/ontology/AIR/SLE
- seed-air-010: gold=NONE | no-rag=NONE | rag=NONE
- seed-air-011: gold=NONE | no-rag=NONE | rag=NONE
- seed-air-002: gold=http://purl.bioontology.org/ontology/AIR/RA | no-rag=NONE | rag=NONE
- seed-air-003: gold=http://purl.bioontology.org/ontology/AIR/SLE | no-rag=http://purl.bioontology.org/ontology/AIR/SLE | rag=NONE
- seed-air-005: gold=http://purl.bioontology.org/ontology/AIR/GT | no-rag=http://purl.bioontology.org/ontology/AIR/RA | rag=http://purl.bioontology.org/ontology/AIR/DXANK
- seed-air-006: gold=http://purl.bioontology.org/ontology/AIR/GT | no-rag=http://purl.bioontology.org/ontology/AIR/RA | rag=http://purl.bioontology.org/ontology/AIR/SLE
