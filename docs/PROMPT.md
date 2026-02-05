You are Claude (coding mode). Create a SINGLE Jupyter notebook that implements the project described in PROPOSAL.md in the current repo.

Goal
- Compare No-RAG vs RAG(TCO) on synthetic patient charts for thyroid-cancer differential labeling.
- Primary metric: percent agreement (exact match). Optional: agreement@3 if you implement top-3.
- Include negative/distractor charts where the gold label is NONE (no thyroid condition).

Hard requirements
1) The notebook must read PROPOSAL.md at the top and include a short “Plan extracted from proposal” markdown cell summarizing what it will do (2–6 bullets).
2) The notebook must be runnable end-to-end with minimal manual steps.
3) The notebook must include:
   - Programmatic ontology querying (BioPortal REST API preferred) for TCO:
     - list/search classes
     - fetch class label(s), synonyms, and hierarchical parents/children (as available)
   - Build an ontology “document corpus” from TCO classes for retrieval (text chunks).
   - Synthetic chart generator producing a dataset with:
     - thyroid cancer cases mapped to TCO class IDs (a small, curated label set is fine)
     - non-thyroid distractor cases labeled NONE
   - Two inference conditions:
     A) baseline LLM (no retrieval)
     B) RAG(TCO) where retrieved ontology chunks are injected into the prompt
   - Evaluation:
     - exact percent agreement between predicted label and gold label for both conditions
     - a small results table + a confusion matrix-like display (counts; plotting optional)
   - Save artifacts to disk:
     - synthetic_charts.csv
     - tco_corpus.jsonl (or parquet)
     - results.json (metrics + run metadata)
     - optional examples.md with a handful of representative successes/failures

Scope control (do not overbuild)
- Keep the TCO subset small and explicit (e.g., 5–10 thyroid cancer TCO classes).
- Retrieval can be simple vector search; fall back to lexical search if embeddings unavailable.
- Use a small LLM call budget: implement caching of model responses to disk keyed by hash(prompt).

Environment assumptions
- Use Python and source .venv/bin/activate
- Prefer these libraries if available: requests, pandas, numpy, scikit-learn.
- If you use embeddings, use sentence-transformers if installed; otherwise implement TF-IDF retrieval.
- Do NOT require external databases.
- For LLM calls: implement a thin abstraction:
  - If OPENAI_API_KEY exists, use OpenAI chat completions (or a placeholder mock if not).
  - If no key exists, run in “dry-run mode” that skips LLM calls and still demonstrates retrieval + pipeline wiring with deterministic baseline heuristics.

BioPortal details
- Read BIOPORTAL_API_KEY from environment.
- Base URL: http(s)://data.bioontology.org
- Include robust error handling, rate-limit backoff, and a “smoke test” cell that confirms successful API access.
- thyroid_cancer_ontology_bioportal.ipynb contains connection example

Notebook structure (use these top-level section headers as markdown cells)
1. Setup and Configuration
2. Load Proposal (PROPOSAL.md)
3. Connect to TCO (BioPortal) and Inspect Ontology
4. Build TCO Retrieval Corpus
5. Generate Synthetic Patient Charts (with gold labels)
6. Retrieval Function (RAG context builder)
7. Prediction: No-RAG baseline
8. Prediction: RAG(TCO)
9. Evaluation (Percent Agreement) + Tables
10. Save Artifacts + Reproducibility Notes

Prompting specs
- Constrain model outputs to the allowed label set:
  - Allowed = {selected TCO_IDs} ∪ {NONE}
- Require model to output strict JSON:
  {"predicted_label": "<TCO_ID_or_NONE>", "top3_labels": ["...optional..."], "rationale": "short"}
- Implement a validator that:
  - coerces invalid outputs to NONE (and logs the error)
  - records raw response for auditing

Synthetic chart generator specs
- Use a fixed random seed.
- Each chart should be 5–12 lines and include: age, sex, presenting symptoms, at least one imaging or pathology hint, and at least one distractor feature for realism.
- Generate at least 60 charts total with an approximate 50/50 split between thyroid-cancer and NONE.
- Ensure the gold label distribution is not single-class dominated; balance across the selected cancer classes.

Deliverable quality
- Include concise comments and markdown explanations.
- Keep it readable; avoid long helper code in cells—define utilities in a few well-named functions.
- At the end, print a compact summary:
  - N charts, label distribution
  - agreement_no_rag, agreement_rag
  - path to saved artifacts

Now create the notebook content as a .ipynb JSON (not a python script), with correct Jupyter notebook format (cells, metadata). Ensure it runs top-to-bottom.