
Implementation Plan: RAG vs No-RAG for Thyroid Cancer Differential Diagnosis
Overview
Create a comprehensive Jupyter notebook comparing No-RAG vs RAG(TCO) approaches for thyroid cancer differential diagnosis using synthetic patient charts.

Goal: Demonstrate that ontology-grounded RAG improves diagnostic consistency by retrieving relevant Thyroid Cancer Ontology (TCO) classifications to constrain LLM outputs.

Primary Metric: Percent agreement (exact match between predicted and gold labels)

Critical Files
thyroid_cancer_ontology_bioportal.ipynb - Working BioPortal API patterns (rate-limit handling, class fetching, normalization utilities)
PROPOSAL.md - Complete project requirements (must be read in notebook Section 2)
PROMPT.md - Notebook structure specification and detailed requirements
.env - Contains BIOPORTAL_API_KEY; check for OPENAI_API_KEY to determine LLM mode
requirements.txt - Current dependencies (pandas, requests) - needs scikit-learn added
Notebook Structure (10 Sections)
Section 1: Setup and Configuration
Purpose: Import libraries, set constants, configure API access, set random seed

Implementation:

Import: pandas, numpy, requests, json, hashlib, random, time
Constants: BASE_URL="https://data.bioontology.org", TCO_ACRONYM="TCO", RANDOM_SEED=42
Load environment: BIOPORTAL_API_KEY, OPENAI_API_KEY (optional)
Set random seeds for reproducibility
Detect retrieval method: sentence-transformers if available, else TF-IDF
Detect LLM mode: OpenAI if key exists, else dry-run
Section 2: Load Proposal (PROPOSAL.md)
Purpose: Read and display project context

Implementation:

Read PROPOSAL.md file content
Display markdown cell with 2-6 bullet summary:
Compare No-RAG vs RAG(TCO) for thyroid cancer diagnosis
Generate 60-120 synthetic patient charts (50/50 thyroid/NONE split)
Query TCO via BioPortal API to build retrieval corpus
Evaluate using exact percent agreement metric
Include negative/distractor cases labeled NONE
Section 3: Connect to TCO (BioPortal) and Inspect Ontology
Purpose: Establish BioPortal connection and validate API access

Implementation:

Reuse API wrapper patterns from thyroid_cancer_ontology_bioportal.ipynb:
api_get() - unauthenticated requests with rate-limit backoff
api_get_auth() - authenticated requests with API key
Session-based requests handling 429 rate limit errors
Smoke test: Query /ontologies/TCO to get metadata
Display: ontology name, version, number of classes
Validate API key works correctly
Key endpoints:

/ontologies/TCO - metadata
/ontologies/TCO/classes - paginated class listing
/ontologies/TCO/classes/{encoded_IRI} - detailed class info
Section 4: Build TCO Retrieval Corpus
Purpose: Create searchable ontology documents from TCO classes

Implementation Steps:

4.1 Select 8 Representative TCO Classes

Query /ontologies/TCO/classes?pagesize=100 with pagination
Filter for thyroid cancer classes using keywords: "carcinoma", "cancer", "tumor", "neoplasm"
Select diverse subtypes: papillary, follicular, medullary, anaplastic, poorly differentiated, hurthle cell, clear cell
Ensure balanced coverage of major diagnostic categories
4.2 Fetch Detailed Class Information

For each selected class IRI:
GET /ontologies/TCO/classes/{encoded_IRI}
Extract: prefLabel, synonyms, definition, parent/child links
Handle 404 errors for cross-ontology lookups
Normalize text fields (may be list or string)
4.3 Build Ontology Documents

Each document contains:
tco_id: Class IRI (unique identifier)
label: Preferred label
synonyms: List of alternative names
definition: Textual definition
parent_labels: Parent class labels (hierarchical context)
document_text: Concatenated searchable text combining all fields
Save to tco_corpus.jsonl (JSON Lines format)
Corpus Structure Example:


{
  "tco_id": "http://purl.obolibrary.org/obo/TCO_0000123",
  "label": "Papillary thyroid carcinoma",
  "synonyms": ["PTC", "papillary thyroid cancer"],
  "definition": "A differentiated thyroid carcinoma...",
  "parent_labels": ["Thyroid carcinoma"],
  "document_text": "Label: Papillary thyroid carcinoma\nSynonyms: PTC, papillary thyroid cancer\n..."
}
Section 5: Generate Synthetic Patient Charts
Purpose: Create balanced dataset with gold labels

Implementation:

5.1 Chart Templates

Thyroid Cancer Templates (60 charts, balanced across 8 TCO classes):

Structure: age, sex, duration, symptoms, exam findings, imaging, pathology, distractor
Imaging hints: "ultrasound shows microcalcifications", "CT shows local invasion"
Pathology hints: "FNA cytology shows suspicious cells", "malignant cells detected"
Distractors: "mild fatigue", "history of hyperlipidemia", "family history of diabetes"
NONE Templates (60 charts, non-thyroid conditions):

Overlapping symptoms: hoarseness, dysphagia, cervical lymphadenopathy, neck pain
Diagnoses: viral laryngitis, reactive lymphadenopathy, GERD, vocal cord strain
Key: "Ultrasound shows normal thyroid", "no thyroid masses", "TSH normal"
5.2 Generation Function

Use fixed random seed (42) for reproducibility
Round-robin assignment of thyroid charts across TCO classes (ensures no single-class domination)
Each chart 5-12 lines with realistic clinical features
Shuffle final dataset to mix thyroid/NONE cases
Save to synthetic_charts.csv with columns: chart_id, chart_text, gold_label, age, sex
Expected Output: 120 charts total (60 thyroid cancer + 60 NONE)

Section 6: Retrieval Function (RAG Context Builder)
Purpose: Build RAG context from retrieved ontology documents

Implementation:

6.1 Retrieval Strategy Decision


try:
    from sentence_transformers import SentenceTransformer
    USE_EMBEDDINGS = True
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    USE_EMBEDDINGS = False
6.2 TF-IDF Retriever (Fallback)

Vectorize corpus using TfidfVectorizer:
max_features=500
stop_words='english'
ngram_range=(1, 2) for phrase matching
Compute cosine similarity between query and documents
Return top-3 most similar documents
6.3 Embedding Retriever (Preferred)

Model: all-MiniLM-L6-v2 (small, fast, effective)
Encode corpus documents once (cache embeddings)
Encode query and compute cosine similarity
Return top-3 most similar documents
6.4 RAG Context Builder

Format retrieved documents as context string:

Relevant thyroid cancer classifications from TCO:

1. Papillary thyroid carcinoma
   Synonyms: PTC, papillary thyroid cancer
   Definition: A differentiated thyroid carcinoma...

2. Follicular thyroid carcinoma
   ...
Section 7: Prediction - No-RAG Baseline
Purpose: Generate predictions without ontology retrieval

Implementation:

7.1 LLM Abstraction Layer

OpenAI Mode (if OPENAI_API_KEY exists):

Model: gpt-3.5-turbo
Temperature: 0.3 (relatively deterministic)
Max tokens: 200
System prompt: Instructs model to output JSON with allowed labels
User prompt: Patient chart text only
Dry-Run Mode (if no API key):

Deterministic keyword heuristics:
"malignant" or "carcinoma" → first thyroid cancer label
"thyroid nodule" or "FNA" → second thyroid cancer label
Otherwise → NONE
Demonstrates pipeline wiring without API costs
7.2 LLM Response Caching

Hash prompt using MD5 to create cache key
Check cache before making API call
Save responses to llm_cache.json after each call
Cache persists across notebook runs (reduces API budget)
7.3 Output Schema


{
  "predicted_label": "<TCO_ID_or_NONE>",
  "top3_labels": ["<label1>", "<label2>", "<label3>"],
  "rationale": "<brief explanation>"
}
7.4 Output Validator

Check predicted_label is in allowed set (8 TCO_IDs + "NONE")
If invalid, coerce to "NONE" and log error
Validate top3_labels entries
Record raw response for auditing
7.5 Run Predictions

Process all 120 charts
Store results: chart_id, gold_label, predicted_label, top3_labels, rationale
Progress indicator every 20 charts
Section 8: Prediction - RAG(TCO)
Purpose: Generate predictions with ontology-grounded retrieval

Implementation:

Use same LLM interface as No-RAG
For each chart:
Build RAG context using retrieval function (Section 6)
Inject context into prompt before chart text
Generate prediction
Validate and cache response
Store results including rag_context field for inspection
Separate cache entries (different prompts due to RAG context)
Expected: RAG context constrains model to ontology-consistent labels, improving agreement

Section 9: Evaluation (Percent Agreement) + Tables
Purpose: Compare No-RAG vs RAG performance

Implementation:

9.1 Metrics Calculation

Exact Agreement: (predicted == gold).sum() / total * 100
Agreement@3: Gold label in top-3 predictions
Calculate for both No-RAG and RAG conditions
9.2 Confusion Matrix

Use pandas crosstab or manual matrix construction
Rows: gold labels (8 TCO classes + NONE)
Columns: predicted labels
Display human-readable labels (not IRIs)
Show counts for each gold/predicted pair
9.3 Per-Class Agreement

Break down agreement by individual TCO class
Identify which cancer types are hardest to classify
9.4 Error Analysis

Extract 3 representative error examples from each condition
Include: chart text, gold label, predicted label, rationale
Compare error patterns between No-RAG and RAG
9.5 Display Results


Evaluation Results:
  No-RAG Exact Agreement: X.X%
  RAG(TCO) Exact Agreement: Y.Y%
  Improvement: Z.Z percentage points

Confusion Matrix - No-RAG:
[pandas DataFrame with readable labels]

Confusion Matrix - RAG(TCO):
[pandas DataFrame with readable labels]
Section 10: Save Artifacts + Reproducibility Notes
Purpose: Export results and document reproducibility

Implementation:

10.1 Save Artifacts

synthetic_charts.csv - All charts with gold labels (saved in Section 5)
tco_corpus.jsonl - Ontology retrieval corpus (saved in Section 4)
results.json - Metrics + metadata:

{
  "metadata": {
    "timestamp": "...",
    "random_seed": 42,
    "n_charts": 120,
    "n_thyroid": 60,
    "n_none": 60,
    "n_tco_classes": 8,
    "retrieval_method": "embeddings|tfidf",
    "llm_mode": "openai|dry_run"
  },
  "metrics": {
    "no_rag": {"exact_agreement": X.X, "agreement_at_3": Y.Y},
    "rag_tco": {"exact_agreement": X.X, "agreement_at_3": Y.Y}
  },
  "label_distribution": {...},
  "tco_classes": {"IRI": "label", ...}
}
examples.md - Representative error examples formatted as markdown
llm_cache.json - Cached LLM responses
10.2 Final Summary Print


==============================================================
FINAL SUMMARY
==============================================================
Total Charts: 120
  Thyroid Cancer: 60
  NONE (distractors): 60

TCO Classes Used: 8
  - Papillary thyroid carcinoma
  - Follicular thyroid carcinoma
  ...

Evaluation Results:
  No-RAG Exact Agreement: X.X%
  RAG(TCO) Exact Agreement: Y.Y%
  Improvement: Z.Z percentage points

Artifacts Saved:
  - synthetic_charts.csv
  - tco_corpus.jsonl
  - results.json
  - examples.md
  - llm_cache.json
==============================================================
10.3 Reproducibility Notes (Markdown Cell)

Fixed random seed (42)
LLM response caching
BioPortal API versioning considerations
Installation instructions
Environment variable requirements
Note about dry-run mode
Key Technical Decisions
Component	Decision	Rationale
Retrieval	TF-IDF fallback from sentence-transformers	Runs without optional dependencies
LLM	OpenAI gpt-3.5-turbo with dry-run mode	Cost-effective with graceful degradation
TCO Selection	8 diverse cancer subtypes	Balances coverage with manageability
Charts	120 total (60 thyroid, 60 NONE)	Balanced dataset, meets 60-120 requirement
Caching	MD5 hash keyed by full prompt	Reduces API calls, disk-persisted
Validation	Coerce invalid to NONE with logging	Robust error handling
Corpus Format	JSON Lines with combined document_text	Easy inspection, optimized for retrieval
Evaluation	Exact agreement + agreement@3 + confusion matrix	Comprehensive performance analysis
Dependencies
Required (must install):


pip install scikit-learn
Optional (improves quality):


pip install sentence-transformers  # Better retrieval
pip install openai  # Actual LLM calls
Updated requirements.txt:


pandas
requests
numpy
scikit-learn
sentence-transformers  # Optional
openai  # Optional
Error Handling Strategy
BioPortal API:

Rate limit (429): Exponential backoff with retry
Not found (404): Log and skip
Timeout: Retry up to 3 times
LLM Responses:

Invalid JSON: Coerce to NONE, log error, record raw response
Invalid labels: Coerce to NONE
API errors: Fall back to NONE with error message in rationale
Missing Dependencies:

No sentence-transformers: Fall back to TF-IDF
No OpenAI key: Use dry-run mode with deterministic heuristics
Verification Steps
After implementation, verify:

Notebook runs top-to-bottom without errors
PROPOSAL.md is read and summarized in Section 2
BioPortal smoke test passes in Section 3
Corpus contains 8 TCO classes with complete metadata
120 charts generated with balanced distribution
Both No-RAG and RAG pipelines complete successfully
Metrics show meaningful comparison (RAG should improve agreement)
All 5 artifacts saved correctly
Results are reproducible with same random seed
Notebook works in both dry-run and OpenAI modes
Expected Outcomes
Corpus: 8 thyroid cancer TCO classes with labels, synonyms, definitions
Charts: 120 synthetic patient charts (60 thyroid, 60 NONE)
Agreement (No-RAG): Baseline performance (likely 40-60% depending on LLM quality)
Agreement (RAG): Improved performance (target: +10-20 percentage points)
Demonstration: Clear evidence that ontology-grounded retrieval improves diagnostic consistency



1. tco_rag_comparison.ipynb
A complete, runnable Jupyter notebook with all 10 required sections:

Section 1: Setup and Configuration

Imports, constants, API configuration
Auto-detects retrieval method (embeddings vs TF-IDF)
Auto-detects LLM mode (OpenAI vs dry-run)
Section 2: Load Proposal

Reads PROPOSAL.md
Displays 6-bullet plan summary
Section 3: Connect to TCO (BioPortal)

API wrapper functions with rate-limit handling (reused from existing notebook)
Smoke test to validate connection
Displays ontology metadata
Section 4: Build TCO Retrieval Corpus

Fetches all TCO classes with pagination
Selects 8 diverse thyroid cancer classes (papillary, follicular, medullary, etc.)
Fetches detailed information (labels, synonyms, definitions, hierarchy)
Builds searchable ontology documents
Saves to tco_corpus.jsonl
Section 5: Generate Synthetic Patient Charts

Template-based generation with realistic clinical features
60 thyroid cancer charts (balanced across 8 TCO classes)
60 NONE distractor charts (non-thyroid conditions)
Saves to synthetic_charts.csv
Section 6: Retrieval Function

Dual implementation: embeddings (sentence-transformers) or TF-IDF fallback
Top-3 retrieval with similarity scores
RAG context builder formatting ontology documents
Section 7: Prediction - No-RAG

LLM abstraction layer (OpenAI or dry-run mode)
Hash-based prompt caching (llm_cache.json)
Output validator (coerces invalid labels to NONE)
Processes all 120 charts
Section 8: Prediction - RAG(TCO)

Same LLM interface with ontology context injection
Retrieves relevant TCO classes for each chart
Separate cache entries due to different prompts
Section 9: Evaluation

Exact percent agreement calculation
Agreement@3 metric
Confusion matrices (readable labels)
Error analysis with 3 examples per condition
Section 10: Save Artifacts

results.json (metrics + metadata)
examples.md (error examples)
Final summary display
Reproducibility notes
2. Updated requirements.txt
Added necessary dependencies:

numpy
scikit-learn (required for TF-IDF fallback)
sentence-transformers (optional, for better retrieval)
openai (optional, for actual LLM calls)
Key Features
✓ End-to-end runnable - No manual intervention required
✓ Graceful fallbacks - Works without sentence-transformers or OpenAI API key
✓ Reproducible - Fixed random seed (42) and LLM response caching
✓ Robust error handling - Rate limits, 404s, invalid JSON, coercion to NONE
✓ Comprehensive evaluation - Agreement metrics, confusion matrices, error examples
✓ Complete artifacts - All 5 required files saved with metadata

To Run

# Install dependencies
pip install -r requirements.txt

# Set environment variables (BioPortal key already in .env)
export OPENAI_API_KEY=<your_key>  # Optional - uses dry-run mode if not set

# Run notebook
jupyter notebook tco_rag_comparison.ipynb
The notebook will work in dry-run mode using deterministic keyword heuristics if no OpenAI API key is provided, allowing you to demonstrate the pipeline wiring and retrieval functionality without API costs.