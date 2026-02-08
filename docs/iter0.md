What Worked

Single-command run works: python3 evaluate.py generates all required artifacts: results.json, predictions.csv, summary.md.
Mechanical scoring + constraints are enforced: exact percent agreement only; invalid predicted labels are coerced to NONE and counted in results.json (coercions).
Caching behavior is in place: LLM responses are cached via llm_cache.json, and retrieval embeddings/index are cached under retriever_cache (so reruns are fast and stable when prompts/seed don’t change).
What Failed / Fell Short

Label-set mismatch caused data loss: 16/120 rows in synthetic_charts.csv were excluded because their gold labels are not in label_set.json (Stage_IV and Stage_IVA variants). The official evaluation therefore ran on n_cases=104, not 120 (recorded in results.json).
No measurable RAG lift under current backend: both No-RAG and RAG(TCO) agreement are identical in the latest run (dry-run heuristic backend), so this run doesn’t yet demonstrate the core hypothesis (RAG improves agreement).
Minimal Scope Refinements (Final Deliverables)

Resolve the gold-label mismatch (pick one):
Expand label_set.json to include the two stage-IV IRIs present in synthetic_charts.csv (keeps dataset intact at 120), or
Regenerate/normalize synthetic_charts.csv gold labels to the existing 6-label set (keeps label space small, but requires dataset edit).
Make the evaluation “real-model ready” while staying minimal:
Run once with a real backend (Gemini/OpenAI/HF) to test the RAG-vs-noRAG claim; keep caching on so it’s repeatable.
Optional but high-value for determinism: set temperature to 0 for evaluation runs (or expose it as an env var) so cached/non-cached runs align better.
Tighten reporting without adding new metrics:
In summary.md, list the excluded gold label counts (already present in JSON) so reviewers immediately see why n_cases < 120.
Keep the example cases split as “RAG correct” vs “RAG incorrect” (already effectively done) to support quick qualitative inspection.