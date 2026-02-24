# Evaluation Summary

## Metrics

| Condition | Exact agreement | N | k | Backend |
|---|---:|---:|---:|---|
| No-RAG | 43.3% | 104 | 3 | dry_run |
| RAG(tco) | 42.3% | 104 | 3 | dry_run |

## Inferential (Paired t-test)

- Mean agreement delta (RAG - No-RAG): -0.96 points
- Improved / worse / unchanged: 6 / 7 / 91
- t-statistic: -0.2761155959988715
- p-value: 0.783
- df: 103

## Parameters

- Ontology: tco
- Dataset: data/synthetic_charts.csv
- Seed: 42
- Model: dry_run
- Embedding model: all-MiniLM-L6-v2
- Git commit: a9b3ab8db968d10da214c063531b39e546d94409
- Excluded rows (gold label outside allowed set): 16

## Example Cases (5–10)

- chart-2: gold=NONE | no-rag=NONE | rag=NONE
- chart-3: gold=NONE | no-rag=NONE | rag=NONE
- chart-10: gold=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_I_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8 | no-rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma | rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_I_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8
- chart-12: gold=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_I_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8 | no-rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma | rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_I_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8
- chart-4: gold=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_I_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8 | no-rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma | rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma
- chart-5: gold=NONE | no-rag=NONE | rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_I_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8
- chart-6: gold=NONE | no-rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma | rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_I_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8
- chart-7: gold=NONE | no-rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma | rag=http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_I_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8

## Differential Diagnosis Snippets (RAG)

- chart-2: [{"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma", "rationale": "The patient has no symptoms suggestive of papillary thyroid carcinoma, such as neck swelling or difficulty swallowing."}, {"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_I_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8", "rationale": "The patient's age and other factors suggest they may have stage I disease, which typically involves localized disease without metastasis."}]
- chart-3: [{"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma", "rationale": "The patient has symptoms consistent with papillary thyroid carcinoma, which can cause neck pain and laryngeal inflammation."}, {"label": "NONE", "rationale": "This classification aligns with the patient's age and presentation, as classical papillary thyroid carcinoma is the most common type among younger patients."}]
- chart-4: [{"label": "NONE", "rationale": "The patient has a 2-month history of an anterior neck mass, which is consistent with a papillary carcinoma."}, {"label": "NONE", "rationale": "The presence of tall malignant follicular cells suggests a variant of papillary carcinoma."}]
- chart-5: [{"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Tall_Cell_Variant_Thyroid_Gland_Papillary_Carcinoma", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}, {"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_I_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}, {"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}]
- chart-6: [{"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Tall_Cell_Variant_Thyroid_Gland_Papillary_Carcinoma", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}, {"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Stage_I_Thyroid_Gland_Papillary_Carcinoma_AJCC_v8", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}, {"label": "http://www.semanticweb.org/hx-jta/ontologies/thyroid_cancer_ontology#Thyroid_Gland_Papillary_Carcinoma", "rationale": "Heuristic candidate based on possible thyroid cancer indicators"}]
