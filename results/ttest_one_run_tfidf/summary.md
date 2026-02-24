# Evaluation Summary

## Metrics

| Condition | Exact agreement | N | k | Backend |
|---|---:|---:|---:|---|
| No-RAG | 43.3% | 104 | 3 | dry_run |
| RAG(tco) | 42.3% | 104 | 3 | dry_run |

## Inferential (Paired t-test)

- Mean agreement delta (RAG - No-RAG): -0.96 points
- Improved / worse / unchanged: 0 / 1 / 103
- t-statistic: -1.0
- p-value: 0.3197
- df: 103

## Parameters

- Ontology: tco
- Dataset: data/synthetic_charts.csv
- Seed: 42
- Model: dry_run
- Embedding model: all-MiniLM-L6-v2
- Retrieval mode: tfidf
- Git commit: ecbab2312fe2e976add793c771085fca99131d6a
- Excluded rows (gold label outside allowed set): 16

## Example Cases (5–10)

- chart-2: gold=NONE | no-rag=NONE | rag=NONE
- chart-3: gold=NONE | no-rag=NONE | rag=NONE
- chart-8: gold=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma | no-rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma | rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma
- chart-13: gold=NONE | no-rag=NONE | rag=NONE
- chart-4: gold=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_I_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8 | no-rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma | rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma
- chart-5: gold=NONE | no-rag=NONE | rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma
- chart-6: gold=NONE | no-rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma | rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma
- chart-7: gold=NONE | no-rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma | rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma

## Differential Diagnosis Snippets (RAG)

- chart-2: [{"label": "NONE", "rationale": "Heuristic: normal/benign findings"}]
- chart-3: [{"label": "NONE", "rationale": "Heuristic: normal/benign findings"}]
- chart-4: [{"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Tall_Cell_Variant_Thyroid_Gland_Papillary_Carcinoma", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}, {"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}, {"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_III_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}]
- chart-5: [{"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Tall_Cell_Variant_Thyroid_Gland_Papillary_Carcinoma", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}, {"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}, {"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_III_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}]
- chart-6: [{"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Tall_Cell_Variant_Thyroid_Gland_Papillary_Carcinoma", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}, {"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}, {"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_III_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}]
