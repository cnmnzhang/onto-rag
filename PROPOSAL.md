Ontology-Grounded RAG for Thyroid Cancer Differentials Using Synthetic Patient Charts

Problem and motivation

Large language models can produce plausible clinical differentials, but their outputs are often inconsistent, overly confident, and weakly grounded in domain knowledge. In thyroid cancer and related presentations (e.g., neck mass, dysphagia, hoarseness), small distinctions in terminology and subtype definitions materially affect the differential. A compact domain ontology such as the Thyroid Cancer Ontology (TCO) offers a structured, controlled vocabulary and hierarchical organization that can be used to constrain and ground model outputs. This project evaluates whether retrieval-augmented generation (RAG) using TCO improves diagnostic labeling consistency and correctness compared to a no-RAG baseline, using synthetic patient charts to control ground truth and avoid data-access delays.

Knowledge resources
	•	Primary ontology: Thyroid Cancer Ontology (TCO) (OWL/RDF form; accessed via BioPortal API and/or local OWL file)
	•	Knowledge access layer: BioPortal REST API for programmatic ontology querying (labels, synonyms, hierarchy, definitions/annotations if available)
	•	Optional supporting terminology (non-required): none (kept minimal to match scope)

Project goals (mini-aims)

Aim 1 — Ontology-grounded RAG pipeline.
Build a RAG system where retrieval is derived from TCO content (class labels, synonyms, definitions/annotations, and hierarchical neighborhood). The generation model produces a structured differential that references only ontology-backed entities.

Aim 2 — Controlled-terminology output for differential diagnosis.
Force the system to output diagnoses as TCO class identifiers and labels, preventing free-text drift and enabling objective evaluation.

Aim 3 — Comparative evaluation (RAG vs no-RAG).
Using a curated set of synthetic patient charts that include thyroid cancer cases, benign thyroid disease cases, and non-thyroid “distractor” cases with overlapping symptoms, compare percent agreement with ground-truth chart labels between:
	•	Baseline LLM (no RAG)
	•	LLM + RAG(TCO)

Data: synthetic patient charts with distractors

We will generate a dataset of synthetic charts (target: 60–120), each labeled with a ground-truth outcome. Charts will be compact and standardized for scoring.

Chart types
	1.	Thyroid cancer (TCO-labeled)
	2.	Non-cancer thyroid conditions (may map to a “non-cancer thyroid” bucket; if TCO lacks coverage, these will be explicitly labeled as non-TCO thyroid condition)
	3.	Non-thyroid cases with overlapping symptoms (distractors)
	•	Examples of overlapping presentations: hoarseness, cervical lymphadenopathy, dysphagia, fatigue, weight loss, anxiety-like symptoms, incidental imaging findings
	•	Ground-truth label: No thyroid condition (a controlled “NONE” label outside TCO, used only as a sentinel class)

Why distractors matter
A realistic differential task must include negative cases; otherwise, agreement can be artificially inflated. Including distractors tests whether RAG reduces inappropriate thyroid-cancer labeling when symptoms are non-specific.

Methods (technical)

Ontology processing and indexing
	1.	Programmatic querying (BioPortal):
	•	Retrieve class list (or scoped subset)
	•	For each class: label(s), synonym(s), definition/annotation text if available
	•	Retrieve hierarchical relations (parents/children) to support neighborhood expansion
	2.	Textualization (“ontology documents”):
Each class becomes an indexable document chunk, e.g.
	•	TCO_ID, preferred label, synonyms
	•	short definition/notes (if available)
	•	parent label(s) and immediate children labels
	3.	Retriever:
	•	Embedding-based retrieval over ontology documents
	•	Query formed from chart text (and optionally extracted candidate terms)

Two inference conditions

Condition A: No-RAG baseline
	•	Input: chart text
	•	Output: structured differential with top-1 (or top-3) predictions

Condition B: RAG(TCO)
	•	Input: chart text
	•	Retrieval: top-k ontology chunks
	•	Generation: model instructed to use retrieved content and to output only from the allowed label set

Output schema (enables objective scoring)
	•	predicted_label: either a TCO_ID (thyroid cancer classes) or sentinel NONE
	•	Optional: top3_labels for differential-style evaluation
	•	rationale: brief explanation (not used for primary scoring; retained for qualitative examples)

Evaluation: percent agreement

Primary metric(s), reported separately for each condition:
	•	Exact agreement: % charts where predicted_label == gold_label
Optionally (if using a top-3 differential output):
	•	Agreement@3: % charts where gold_label ∈ top3_labels

No stratified analysis is required; the evaluation remains a straightforward head-to-head comparison.

Expected results

We expect RAG(TCO) to improve agreement by:
	•	reducing synonym/terminology drift (controlled terminology effect)
	•	improving subtype selection when TCO provides structured naming and proximity context
	•	reducing inappropriate thyroid-cancer labeling in distractor cases by anchoring the model to what is and is not in the ontology-derived label set

Feasibility and risks
	•	Ontology coverage risk: If TCO does not include benign thyroid conditions, we treat them as either (a) NONE if outside scope, or (b) an explicitly defined non-cancer category outside TCO, documented as a limitation.
	•	Ontology sparsity risk: If definitional text is limited, the RAG benefit may be smaller; in that case, the project still demonstrates controlled terminology + hierarchy-aware retrieval, and evaluation remains valid.
	•	Synthetic realism risk: We will use structured templates and clinician-facing style constraints to keep charts plausible, while clearly labeling them synthetic.

Timeline aligned to checkpoints and finals

Checkpoint 1 (plan): finalize ontology choice (TCO), define label space (TCO classes + sentinel NONE), draft chart template, outline evaluation metric.
Checkpoint 2 (programmatic demo): show working BioPortal queries that (1) list classes, (2) retrieve labels/synonyms, (3) retrieve parent/child relations; show a “hello world” retrieval for a sample chart and the retrieved ontology chunks.
Final (March 16): full RAG pipeline, completed synthetic dataset, agreement results, qualitative error analysis examples, slides + essay + ontology pointer/artifacts.

LLM attestation plan

We will document how LLMs were used in:
	•	generating synthetic charts (if used)
	•	summarizing ontology chunks (if used)
	•	running the baseline and RAG conditions
We will explicitly state that no real patient data were used, and that outputs are for evaluation of ontology-grounded NLP methods rather than clinical decision-making.