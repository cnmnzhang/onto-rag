#!/usr/bin/env python3
"""
Ontology-Grounded RAG for Thyroid Cancer Differential Diagnosis
Comparing No-RAG vs RAG(TCO) on Synthetic Patient Charts
"""

import os
import time
import json
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from urllib.parse import quote

import pandas as pd
import numpy as np
import requests

from llm_interface import LLMInterface
from retrievers import create_retriever
from synthetic_data import DOMAIN_SPECS, generate_synthetic_dataset

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

BASE_URL = "https://data.bioontology.org"
TCO_ACRONYM = "TCO"
TIMEOUT = 30
CACHE_FILE = "llm_cache.json"
RANDOM_SEED = 42

# Load environment variables
BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("Ontology-Grounded RAG for Thyroid Cancer Diagnosis")
print("=" * 60)
print(f"Random seed: {RANDOM_SEED}")
print(f"BioPortal API key: {'Found' if BIOPORTAL_API_KEY else 'Not found'}")
print(f"OpenAI API key: {'Found' if OPENAI_API_KEY else 'Not found'}")
print(f"Use Hugging Face: {os.getenv('USE_HUGGINGFACE', 'false')}")
print()


# ============================================================================
# BioPortal API Functions
# ============================================================================

print("\n" + "=" * 60)
print("Connecting to BioPortal")
print("=" * 60)

session = requests.Session()


def api_get(endpoint: str, params: Optional[Dict] = None) -> Dict:
    """Make unauthenticated API request with rate-limit handling."""
    url = f"{BASE_URL}{endpoint}"
    response = session.get(url, params=params or {}, timeout=TIMEOUT)

    if response.status_code == 429:
        print("Rate limited, waiting 1 second...")
        time.sleep(1.0)
        response = session.get(url, params=params or {}, timeout=TIMEOUT)

    response.raise_for_status()
    return response.json()


def api_get_auth(endpoint: str, params: Optional[Dict] = None) -> Dict:
    """Make authenticated API request."""
    params = dict(params or {})
    if BIOPORTAL_API_KEY:
        params["apikey"] = BIOPORTAL_API_KEY
    return api_get(endpoint, params)


# Smoke test
try:
    ontology_meta = api_get_auth(f"/ontologies/{TCO_ACRONYM}")
    print(f"✓ Successfully connected to BioPortal")
    print(f"  Ontology: {ontology_meta.get('name')}")
    print(f"  Acronym: {ontology_meta.get('acronym')}")

    submission = api_get_auth(f"/ontologies/{TCO_ACRONYM}/latest_submission")
    print(f"  Version: {submission.get('version', 'N/A')}")
    print(f"  Released: {submission.get('released', 'N/A')}")
except Exception as e:
    print(f"✗ Error connecting to BioPortal: {e}")
    raise


# ============================================================================
# Build TCO Retrieval Corpus
# ============================================================================

print("\n" + "=" * 60)
print("Building TCO Retrieval Corpus")
print("=" * 60)


def list_all_tco_classes(max_pages: int = 10) -> List[Dict]:
    """Retrieve all TCO classes with pagination."""
    classes = []
    for page in range(1, max_pages + 1):
        try:
            data = api_get_auth(
                f"/ontologies/{TCO_ACRONYM}/classes",
                params={"page": page, "pagesize": 100}
            )
            collection = data if isinstance(data, list) else data.get("collection", [])
            if not collection:
                break
            classes.extend(collection)
            print(f"  Page {page}: {len(collection)} classes")
        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break
    return classes


def select_thyroid_cancer_classes(classes: List[Dict], n: int = 8) -> List[Dict]:
    """Select representative thyroid cancer class IRIs."""
    cancer_keywords = ["carcinoma", "cancer", "tumor", "neoplasm", "adenoma"]
    candidates = []

    for cls in classes:
        label = (cls.get("prefLabel") or cls.get("label") or "").lower()
        if any(kw in label for kw in cancer_keywords) and "thyroid" in label:
            candidates.append({
                "iri": cls.get("@id"),
                "label": cls.get("prefLabel") or cls.get("label")
            })

    # Select diverse subtypes
    diversity_keywords = [
        "papillary", "follicular", "medullary", "anaplastic",
        "poorly differentiated", "hurthle cell", "clear cell", "insular"
    ]

    selected = []
    for keyword in diversity_keywords:
        for candidate in candidates:
            if keyword in candidate["label"].lower() and candidate not in selected:
                selected.append(candidate)
                if len(selected) >= n:
                    return selected

    # Fill remaining slots
    for candidate in candidates:
        if candidate not in selected:
            selected.append(candidate)
            if len(selected) >= n:
                break

    return selected


def normalize_list_field(value: Any) -> List[str]:
    """Normalize various field formats to list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def get_class_details(iri: str) -> Dict:
    """Fetch full class details including synonyms and hierarchy."""
    encoded = quote(iri, safe="")
    try:
        return api_get_auth(f"/ontologies/{TCO_ACRONYM}/classes/{encoded}")
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  Class not found: {iri}")
            return {"@id": iri, "error": "not_found"}
        raise


def build_ontology_document(class_details: Dict) -> Dict:
    """Create searchable document from class details."""
    iri = class_details.get("@id")
    label = class_details.get("prefLabel") or class_details.get("label") or ""

    # Extract synonyms
    synonyms = normalize_list_field(class_details.get("synonym"))

    # Extract definition
    definition = normalize_list_field(class_details.get("definition"))

    # Extract parent labels
    parent_labels = []
    parents_link = class_details.get("links", {}).get("parents")
    if parents_link:
        try:
            parents_data = api_get_auth(parents_link.replace(BASE_URL, ""))
            parent_collection = parents_data if isinstance(parents_data, list) else parents_data.get("collection", [])
            parent_labels = [p.get("prefLabel") or p.get("label") for p in parent_collection if p.get("prefLabel") or p.get("label")]
        except:
            pass

    # Create document text
    text_parts = [
        f"Label: {label}",
        f"Synonyms: {', '.join(synonyms)}" if synonyms else "",
        f"Definition: {' '.join(definition)}" if definition else "",
        f"Parent classes: {', '.join(parent_labels)}" if parent_labels else "",
    ]

    document_text = "\n".join([p for p in text_parts if p])

    return {
        "tco_id": iri,
        "label": label,
        "synonyms": synonyms,
        "definition": " ".join(definition),
        "parent_labels": parent_labels,
        "document_text": document_text
    }


# Fetch all classes
print("Fetching all TCO classes...")
all_classes = list_all_tco_classes(max_pages=10)
print(f"✓ Retrieved {len(all_classes)} total classes")

# Select representative classes
print("\nSelecting representative thyroid cancer classes...")
selected_classes = select_thyroid_cancer_classes(all_classes, n=8)
print(f"✓ Selected {len(selected_classes)} thyroid cancer classes:")
for cls in selected_classes:
    print(f"  - {cls['label']}")

# Build corpus
print("\nFetching detailed information for selected classes...")
corpus = []
tco_label_map = {}  # IRI -> label mapping

for cls in selected_classes:
    iri = cls["iri"]
    print(f"  Fetching: {cls['label']}")
    details = get_class_details(iri)
    if "error" not in details:
        doc = build_ontology_document(details)
        corpus.append(doc)
        tco_label_map[iri] = doc["label"]
        time.sleep(0.2)  # Rate limit courtesy

print(f"\n✓ Built corpus with {len(corpus)} TCO classes")

# Save corpus
corpus_df = pd.DataFrame(corpus)
corpus_df.to_json("tco_corpus.jsonl", orient="records", lines=True)
print("✓ Saved corpus to tco_corpus.jsonl")


# ============================================================================
# Generate Synthetic Patient Charts
# ============================================================================

print("\n" + "=" * 60)
print("Generating Synthetic Patient Charts")
print("=" * 60)


# Generate dataset
domain_spec = DOMAIN_SPECS["tco_thyroid"]
charts_df = generate_synthetic_dataset(
    corpus,
    tco_label_map,
    n_domain=60,
    n_none=60,
    domain=domain_spec,
)
charts_df.to_csv("synthetic_charts.csv", index=False)

print(f"✓ Generated {len(charts_df)} charts")
print("✓ Saved charts to synthetic_charts.csv")

# Display distribution
print("\nLabel distribution:")
label_counts = charts_df["gold_label"].value_counts()
for label, count in label_counts.items():
    display_label = tco_label_map.get(label, label)
    print(f"  {display_label}: {count}")


# ============================================================================
# Build Retrieval Function
# ============================================================================

print("\n" + "=" * 60)
print("Building Retrieval Function")
print("=" * 60)

retriever = create_retriever(corpus, top_k=3, prefer_embeddings=True)

print("✓ Retriever initialized")


def build_rag_context(chart_text: str, retriever) -> str:
    """Build RAG context from retrieved ontology documents."""
    retrieved = retriever.retrieve(chart_text)

    context_parts = ["Relevant thyroid cancer classifications from TCO:"]
    for i, doc in enumerate(retrieved, 1):
        context_parts.append(f"\n{i}. {doc['label']}")
        if doc.get('synonyms'):
            context_parts.append(f"   Synonyms: {', '.join(doc['synonyms'][:3])}")
        if doc.get('definition'):
            def_text = doc['definition'][:200]
            context_parts.append(f"   Definition: {def_text}..." if len(doc['definition']) > 200 else f"   Definition: {def_text}")

    return "\n".join(context_parts)


# ============================================================================
# Initialize LLM Interface
# ============================================================================

print("\n" + "=" * 60)
print("Initializing LLM Interface")
print("=" * 60)

allowed_labels = [doc["tco_id"] for doc in corpus]
llm = LLMInterface(allowed_labels, cache_file="llm_cache.json")

backend_info = llm.get_backend_info()
print(f"\nBackend: {backend_info['backend']}")
print(f"Cache size: {backend_info['cache_size']} entries")
print(f"Cache file: {backend_info['cache_file']}")


# ============================================================================
# Run No-RAG Predictions
# ============================================================================

print("\n" + "=" * 60)
print("Running No-RAG Predictions")
print("=" * 60)

no_rag_results = []

for idx, row in charts_df.iterrows():
    prediction = llm.predict(row["chart_text"], rag_context=None)
    no_rag_results.append({
        "chart_id": row["chart_id"],
        "gold_label": row["gold_label"],
        "predicted_label": prediction["predicted_label"],
        "top3_labels": prediction["top3_labels"],
        "rationale": prediction["rationale"]
    })

    if (idx + 1) % 20 == 0:
        print(f"  Processed {idx + 1}/{len(charts_df)} charts")

no_rag_df = pd.DataFrame(no_rag_results)
print(f"✓ Completed No-RAG predictions for {len(no_rag_df)} charts")


# ============================================================================
# Run RAG(TCO) Predictions
# ============================================================================

print("\n" + "=" * 60)
print("Running RAG(TCO) Predictions")
print("=" * 60)

rag_results = []

for idx, row in charts_df.iterrows():
    # Build RAG context
    rag_context = build_rag_context(row["chart_text"], retriever)

    # Get prediction with RAG context
    prediction = llm.predict(row["chart_text"], rag_context=rag_context)
    rag_results.append({
        "chart_id": row["chart_id"],
        "gold_label": row["gold_label"],
        "predicted_label": prediction["predicted_label"],
        "top3_labels": prediction["top3_labels"],
        "rationale": prediction["rationale"],
        "rag_context": rag_context
    })

    if (idx + 1) % 20 == 0:
        print(f"  Processed {idx + 1}/{len(charts_df)} charts")

rag_df = pd.DataFrame(rag_results)
print(f"✓ Completed RAG(TCO) predictions for {len(rag_df)} charts")


# ============================================================================
# Evaluation
# ============================================================================

print("\n" + "=" * 60)
print("Evaluation")
print("=" * 60)


def calculate_agreement(results_df: pd.DataFrame) -> float:
    """Calculate exact percent agreement."""
    correct = (results_df["predicted_label"] == results_df["gold_label"]).sum()
    total = len(results_df)
    return (correct / total) * 100 if total > 0 else 0.0


def calculate_agreement_at_k(results_df: pd.DataFrame, k: int = 3) -> float:
    """Calculate agreement@k (gold label in top-k predictions)."""
    correct = 0
    for _, row in results_df.iterrows():
        if row["gold_label"] in row["top3_labels"][:k]:
            correct += 1
    return (correct / len(results_df)) * 100 if len(results_df) > 0 else 0.0


# Calculate metrics
agreement_no_rag = calculate_agreement(no_rag_df)
agreement_rag = calculate_agreement(rag_df)

agreement_at3_no_rag = calculate_agreement_at_k(no_rag_df, k=3)
agreement_at3_rag = calculate_agreement_at_k(rag_df, k=3)

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"No-RAG Exact Agreement:    {agreement_no_rag:.1f}%")
print(f"RAG(TCO) Exact Agreement:  {agreement_rag:.1f}%")
print(f"Improvement:               {(agreement_rag - agreement_no_rag):+.1f} percentage points")
print()
print(f"No-RAG Agreement@3:        {agreement_at3_no_rag:.1f}%")
print(f"RAG(TCO) Agreement@3:      {agreement_at3_rag:.1f}%")
print("=" * 60)


def create_confusion_matrix(results_df: pd.DataFrame, label_map: Dict) -> pd.DataFrame:
    """Create confusion matrix as DataFrame."""
    # Get unique labels (sorted)
    all_labels = sorted(set(results_df["gold_label"]) | set(results_df["predicted_label"]))

    # Initialize matrix
    matrix = defaultdict(lambda: defaultdict(int))

    for _, row in results_df.iterrows():
        gold = row["gold_label"]
        pred = row["predicted_label"]
        matrix[gold][pred] += 1

    # Convert to DataFrame
    df = pd.DataFrame(matrix).T.fillna(0).astype(int)

    # Rename rows/columns to readable labels
    label_names = {iri: label_map.get(iri, "NONE") for iri in all_labels}
    label_names["NONE"] = "NONE"

    # Shorten labels for display
    def shorten_label(label):
        if label == "NONE":
            return "NONE"
        return label[:30] + "..." if len(label) > 30 else label

    display_names = {k: shorten_label(v) for k, v in label_names.items()}

    df = df.rename(index=display_names, columns=display_names)

    return df


# Create confusion matrices
print("\nConfusion Matrix - No-RAG:")
print("(Rows: Gold Label, Columns: Predicted Label)")
cm_no_rag = create_confusion_matrix(no_rag_df, tco_label_map)
print(cm_no_rag)

print("\n" + "=" * 60)
print("\nConfusion Matrix - RAG(TCO):")
print("(Rows: Gold Label, Columns: Predicted Label)")
cm_rag = create_confusion_matrix(rag_df, tco_label_map)
print(cm_rag)


# ============================================================================
# Save Artifacts
# ============================================================================

print("\n" + "=" * 60)
print("Saving Artifacts")
print("=" * 60)

# Save results.json
results = {
    "metadata": {
        "timestamp": pd.Timestamp.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "n_charts": len(charts_df),
        "n_thyroid": int((charts_df["gold_label"] != "NONE").sum()),
        "n_none": int((charts_df["gold_label"] == "NONE").sum()),
        "n_tco_classes": len(corpus),
        "retrieval_method": "embeddings" if retriever.__class__.__name__ == "EmbeddingRetriever" else "tfidf",
        "llm_backend": backend_info["backend"]
    },
    "metrics": {
        "no_rag": {
            "exact_agreement": float(agreement_no_rag),
            "agreement_at_3": float(agreement_at3_no_rag)
        },
        "rag_tco": {
            "exact_agreement": float(agreement_rag),
            "agreement_at_3": float(agreement_at3_rag)
        },
        "improvement": {
            "exact_agreement_delta": float(agreement_rag - agreement_no_rag),
            "agreement_at_3_delta": float(agreement_at3_rag - agreement_at3_no_rag)
        }
    },
    "label_distribution": charts_df["gold_label"].value_counts().to_dict(),
    "tco_classes": {doc["tco_id"]: doc["label"] for doc in corpus}
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✓ Saved results.json")


# Extract error examples
def extract_error_examples(
    results_df: pd.DataFrame,
    charts_df: pd.DataFrame,
    label_map: Dict,
    n_examples: int = 3
) -> List[Dict]:
    """Extract example errors for qualitative analysis."""
    errors = results_df[results_df["predicted_label"] != results_df["gold_label"]]

    examples = []
    for _, row in errors.head(n_examples).iterrows():
        chart = charts_df[charts_df["chart_id"] == row["chart_id"]].iloc[0]
        examples.append({
            "chart_text": chart["chart_text"],
            "gold_label": label_map.get(row["gold_label"], row["gold_label"]),
            "predicted_label": label_map.get(row["predicted_label"], row["predicted_label"]),
            "rationale": row["rationale"]
        })

    return examples


error_examples_no_rag = extract_error_examples(no_rag_df, charts_df, tco_label_map, n_examples=3)
error_examples_rag = extract_error_examples(rag_df, charts_df, tco_label_map, n_examples=3)

# Save examples.md
with open("examples.md", "w") as f:
    f.write("# Representative Examples\n\n")
    f.write("This file contains representative error examples from both No-RAG and RAG(TCO) conditions.\n\n")

    f.write("## No-RAG Error Examples\n\n")
    for i, ex in enumerate(error_examples_no_rag, 1):
        f.write(f"### Example {i}\n\n")
        f.write(f"**Chart:** {ex['chart_text']}\n\n")
        f.write(f"**Gold Label:** {ex['gold_label']}\n\n")
        f.write(f"**Predicted:** {ex['predicted_label']}\n\n")
        f.write(f"**Rationale:** {ex['rationale']}\n\n")
        f.write("---\n\n")

    f.write("## RAG(TCO) Error Examples\n\n")
    for i, ex in enumerate(error_examples_rag, 1):
        f.write(f"### Example {i}\n\n")
        f.write(f"**Chart:** {ex['chart_text']}\n\n")
        f.write(f"**Gold Label:** {ex['gold_label']}\n\n")
        f.write(f"**Predicted:** {ex['predicted_label']}\n\n")
        f.write(f"**Rationale:** {ex['rationale']}\n\n")
        f.write("---\n\n")

print("✓ Saved examples.md")


# ============================================================================
# Final Summary
# ============================================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Total Charts: {len(charts_df)}")
print(f"  Thyroid Cancer: {(charts_df['gold_label'] != 'NONE').sum()}")
print(f"  NONE (distractors): {(charts_df['gold_label'] == 'NONE').sum()}")
print(f"\nTCO Classes Used: {len(corpus)}")
for doc in corpus:
    print(f"  - {doc['label']}")
print(f"\nEvaluation Results:")
print(f"  No-RAG Exact Agreement: {agreement_no_rag:.1f}%")
print(f"  RAG(TCO) Exact Agreement: {agreement_rag:.1f}%")
print(f"  Improvement: {(agreement_rag - agreement_no_rag):+.1f} percentage points")
print(f"\nArtifacts Saved:")
print(f"  - synthetic_charts.csv")
print(f"  - tco_corpus.jsonl")
print(f"  - results.json")
print(f"  - examples.md")
print(f"  - llm_cache.json")
print("=" * 60)
print("\n✓ Execution complete!")


if __name__ == "__main__":
    pass
