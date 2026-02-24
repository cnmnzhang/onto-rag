# Evaluation Summary

## Metrics

| Condition | Exact agreement | N | k | Backend |
|---|---:|---:|---:|---|
| No-RAG | 27.8% | 18 | 3 | huggingface |
| RAG(ai_rheum) | 44.4% | 18 | 3 | huggingface |

## Inferential (Paired t-test)

- Mean agreement delta (RAG - No-RAG): 16.67 points
- Improved / worse / unchanged: 4 / 1 / 13
- t-statistic: 1.3743685418725535
- p-value: 0.1872
- df: 17

## Parameters

- Ontology: ai_rheum
- Dataset: data/seed_cases_ai_rheum.csv
- Seed: 42
- Model: Qwen/Qwen2.5-1.5B-Instruct
- Embedding model: all-MiniLM-L6-v2
- Retrieval mode: embeddings
- Git commit: b3e706ecdf2f326a5be6fd7ab96799bb47b7507a

## Example Cases (5–10)

- seed-air-001: gold=http://purl.bioontology.org/ontology/AIR/DXRA | no-rag=http://purl.bioontology.org/ontology/AIR/DXPSO | rag=http://purl.bioontology.org/ontology/AIR/DXRA
- seed-air-007: gold=http://purl.bioontology.org/ontology/AIR/DXPSO | no-rag=http://purl.bioontology.org/ontology/AIR/DXPSO | rag=http://purl.bioontology.org/ontology/AIR/DXPSO
- seed-air-008: gold=http://purl.bioontology.org/ontology/AIR/DXANK | no-rag=http://purl.bioontology.org/ontology/AIR/DXPSO | rag=http://purl.bioontology.org/ontology/AIR/DXANK
- seed-air-010: gold=NONE | no-rag=NONE | rag=NONE
- seed-air-002: gold=http://purl.bioontology.org/ontology/AIR/DXRA | no-rag=http://purl.bioontology.org/ontology/AIR/DXPSO | rag=http://purl.bioontology.org/ontology/AIR/DXANK
- seed-air-003: gold=http://purl.bioontology.org/ontology/AIR/DXSLE | no-rag=http://purl.bioontology.org/ontology/AIR/DXPSO | rag=http://purl.bioontology.org/ontology/AIR/DXPSO
- seed-air-004: gold=http://purl.bioontology.org/ontology/AIR/DXSLE | no-rag=http://purl.bioontology.org/ontology/AIR/DXSLE | rag=http://purl.bioontology.org/ontology/AIR/DXPSO
- seed-air-005: gold=http://purl.bioontology.org/ontology/AIR/DXGT | no-rag=http://purl.bioontology.org/ontology/AIR/DXPSO | rag=http://purl.bioontology.org/ontology/AIR/DXPSO

## Differential Diagnosis Snippets (RAG)

- seed-air-001: [{"label": "http://purl.bioontology.org/ontology/AIR/DXRA", "rationale": "Symmetric MCP/PIP inflammatory arthritis, RF+ indicates Rheumatoid Arthritis"}, {"label": "http://purl.bioontology.org/ontology/AIR/DXPSO", "rationale": "Inflammatory polyarthritis, elevated ESR suggests systemic inflammation consistent with other forms of polyarthritis"}]
- seed-air-002: [{"label": "http://purl.bioontology.org/ontology/AIR/DXANK", "rationale": "Ankylosing Spondylitis has progressive joint involvement, especially in the spine and lower extremities."}, {"label": "http://purl.bioontology.org/ontology/AIR/DXPMR", "rationale": "Polymyalgia Rheumatica presents with myalgia and can involve multiple joints, including the wrists and forefeet."}]
- seed-air-003: [{"label": "http://purl.bioontology.org/ontology/AIR/DXPSO", "rationale": "Systemic lupus erythematosus (SLE) is characterized by photosensitivity, oral ulcers, anti-dsDNA positivity, and low complement levels."}, {"label": "http://purl.bioontology.org/ontology/AIR/DXANK", "rationale": "The presence of anti-dsDNA antibodies suggests an autoimmune process, which could indicate systemic lupus erythematosus."}]
- seed-air-004: [{"label": "http://purl.bioontology.org/ontology/AIR/DXPSO", "rationale": "Symptoms such as pleuritic chest pain, malar rash, cytopenia, nephritis, and low complement suggest systemic lupus erythematosus."}, {"label": "http://purl.bioontology.org/ontology/AIR/DXANK", "rationale": "The combination of pleuritic chest pain and malar rash could also point towards an autoimmune condition like ANCA-associated vasculitis."}]
- seed-air-005: [{"label": "http://purl.bioontology.org/ontology/AIR/DXPMR", "rationale": "First MTP involvement suggests polyarticular onset which aligns with inflammatory polyarthritis classification."}, {"label": "http://purl.bioontology.org/ontology/AIR/DXRA", "rationale": "Redness and swelling indicate an inflammatory process, consistent with rheumatoid arthritis classification."}]
