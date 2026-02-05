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
from onto_config import OntologyConfig, get_config, build_system_prompt, format_template

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# Select ontology configuration
# Options: "tco" (thyroid cancer), "diabetes", "lung_cancer"
# Or create custom: CONFIG = OntologyConfig(acronym="...", ...)
CONFIG = get_config("tco")

BASE_URL = "https://data.bioontology.org"
ONTOLOGY_ACRONYM = CONFIG.acronym
TIMEOUT = 30
CACHE_FILE = "llm_cache.json"
CORPUS_FILE = CONFIG.corpus_filename
RANDOM_SEED = 42

# Load environment variables
BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


from dotenv import load_dotenv
load_dotenv()
# Force Gemini (not Hugging Face)
os.environ['USE_HUGGINGFACE'] = 'true'


# ============================================================================
# BioPortal API Functions
# ============================================================================
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

def test_bioportal_connection():
    """Test connection to BioPortal and fetch ontology metadata."""
    # Smoke test
    try:
        ontology_meta = api_get_auth(f"/ontologies/{ONTOLOGY_ACRONYM}")
        print(f"✓ Successfully connected to BioPortal")
        print(f"  Ontology: {ontology_meta.get('name')}")
        print(f"  Acronym: {ontology_meta.get('acronym')}")

        submission = api_get_auth(f"/ontologies/{ONTOLOGY_ACRONYM}/latest_submission")
        print(f"  Version: {submission.get('version', 'N/A')}")
        print(f"  Released: {submission.get('released', 'N/A')}")
    except Exception as e:
        print(f"✗ Error connecting to BioPortal: {e}")
        raise


# ============================================================================
# Build TCO Retrieval Corpus
# ============================================================================
def list_all_ontology_classes(max_pages: int = 10) -> List[Dict]:
    """Retrieve all ontology classes with pagination."""
    classes = []
    for page in range(1, max_pages + 1):
        try:
            data = api_get_auth(
                f"/ontologies/{ONTOLOGY_ACRONYM}/classes",
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


def select_disease_classes(classes: List[Dict], config: OntologyConfig) -> List[Dict]:
    """Select representative disease classes using config keywords."""
    candidates = []

    # Filter classes by keywords
    for cls in classes:
        label = (cls.get("prefLabel") or cls.get("label") or "").lower()
        # Check if any class keyword is present
        if any(kw in label for kw in config.class_keywords):
            # For organ-specific diseases, also check organ is mentioned
            if config.organ.lower() != "n/a":
                if config.organ.lower() in label or config.disease_name.lower() in label:
                    candidates.append({
                        "iri": cls.get("@id"),
                        "label": cls.get("prefLabel") or cls.get("label")
                    })
            else:
                # For non-organ diseases (like diabetes), just use class keywords
                if config.disease_name.split()[0].lower() in label:  # First word of disease
                    candidates.append({
                        "iri": cls.get("@id"),
                        "label": cls.get("prefLabel") or cls.get("label")
                    })

    # Select diverse subtypes using subtype keywords
    selected = []
    for keyword in config.subtype_keywords:
        for candidate in candidates:
            if keyword in candidate["label"].lower() and candidate not in selected:
                selected.append(candidate)
                if len(selected) >= config.num_classes:
                    return selected

    # Fill remaining slots with any candidates
    for candidate in candidates:
        if candidate not in selected:
            selected.append(candidate)
            if len(selected) >= config.num_classes:
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
        return api_get_auth(f"/ontologies/{ONTOLOGY_ACRONYM}/classes/{encoded}")
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  Class not found: {iri}")
            return {"@id": iri, "error": "not_found"}
        raise


def build_ontology_document(class_details: Dict, config: OntologyConfig) -> Dict:
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
        config.doc_id_field: iri,
        "label": label,
        "synonyms": synonyms,
        "definition": " ".join(definition),
        "parent_labels": parent_labels,
        "document_text": document_text
    }


def load_corpus_cache(path: str, config: OntologyConfig) -> Optional[Tuple[List[Dict], Dict[str, str]]]:
    """Load cached corpus from disk if available and valid."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_json(path, lines=True)
    except ValueError:
        return None
    if df.empty:
        return None

    records = df.to_dict(orient="records")
    corpus: List[Dict] = []
    label_map: Dict[str, str] = {}
    doc_id_field = config.doc_id_field

    for rec in records:
        doc_id = rec.get(doc_id_field)
        label = rec.get("label")
        if not doc_id or not label:
            continue
        corpus.append({
            doc_id_field: doc_id,
            "label": label,
            "synonyms": rec.get("synonyms") or [],
            "definition": rec.get("definition") or "",
            "parent_labels": rec.get("parent_labels") or [],
            "document_text": rec.get("document_text") or "",
        })
        label_map[doc_id] = label

    if not corpus:
        return None

    return corpus, label_map


def save_corpus_cache(corpus: List[Dict], path: str) -> None:
    """Persist corpus to disk for reuse."""
    pd.DataFrame(corpus).to_json(path, orient="records", lines=True)

# Build corpus
def build_rag_context(chart_text: str, retriever, config: OntologyConfig) -> str:
    """Build RAG context from retrieved ontology documents."""
    retrieved = retriever.retrieve(chart_text)

    context_header = format_template(config.rag_context_header, config)
    context_parts = [context_header]

    for i, doc in enumerate(retrieved, 1):
        context_parts.append(f"\n{i}. {doc['label']}")
        if doc.get('synonyms'):
            context_parts.append(f"   Synonyms: {', '.join(doc['synonyms'][:3])}")
        if doc.get('definition'):
            def_text = doc['definition'][:200]
            context_parts.append(f"   Definition: {def_text}..." if len(doc['definition']) > 200 else f"   Definition: {def_text}")

    return "\n".join(context_parts)


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


def main() -> None:
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print(f"Ontology-Grounded RAG for {CONFIG.disease_name.title()} Diagnosis")
    print("=" * 60)
    print(f"Configuration: {CONFIG.name} ({CONFIG.acronym})")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"BioPortal API key: {'Found' if BIOPORTAL_API_KEY else 'Not found'}")
    print(f"OpenAI API key: {'Found' if OPENAI_API_KEY else 'Not found'}")
    print(f"Use Hugging Face: {os.getenv('USE_HUGGINGFACE', 'false')}")
    print()

    print("\n" + "=" * 60)
    print("Connecting to BioPortal")
    print("=" * 60)
    test_bioportal_connection()

    # Build Ontology Retrieval Corpus
    print("\n" + "=" * 60)
    print(f"Building {CONFIG.acronym} Retrieval Corpus")
    print("=" * 60)

    cached = load_corpus_cache(CORPUS_FILE, CONFIG)
    if cached:
        corpus, label_map = cached
        print(f"✓ Loaded cached corpus from {CORPUS_FILE} ({len(corpus)} classes)")
    else:
        print(f"Fetching all {CONFIG.acronym} classes...")
        all_classes = list_all_ontology_classes(max_pages=10)
        print(f"✓ Retrieved {len(all_classes)} total classes")

        print(f"\nSelecting representative {CONFIG.disease_name} classes...")
        selected_classes = select_disease_classes(all_classes, CONFIG)
        print(f"✓ Selected {len(selected_classes)} {CONFIG.disease_name} classes:")
        for cls in selected_classes:
            print(f"  - {cls['label']}")

        print("\nFetching detailed information for selected classes...")
        corpus = []
        label_map = {}  # IRI -> label mapping

        for cls in selected_classes:
            iri = cls["iri"]
            print(f"  Fetching: {cls['label']}")
            details = get_class_details(iri)
            if "error" not in details:
                doc = build_ontology_document(details, CONFIG)
                corpus.append(doc)
                label_map[iri] = doc["label"]
                time.sleep(0.2)  # Rate limit courtesy

        print(f"\n✓ Built corpus with {len(corpus)} {CONFIG.acronym} classes")

        # Save corpus
        save_corpus_cache(corpus, CORPUS_FILE)
        print(f"✓ Saved corpus to {CORPUS_FILE}")

    # Generate Synthetic Patient Charts
    print("\n" + "=" * 60)
    print("Generating Synthetic Patient Charts")
    print("=" * 60)

    from synthetic_data import generate_synthetic_dataset
    charts_df = generate_synthetic_dataset(
        corpus,
        label_map,
        config=CONFIG,
        n_domain=CONFIG.n_disease_charts,
        n_none=CONFIG.n_none_charts,
    )
    charts_df.to_csv("synthetic_charts.csv", index=False)

    print(f"✓ Generated {len(charts_df)} charts")
    print("✓ Saved charts to synthetic_charts.csv")

    print("\nLabel distribution:")
    label_counts = charts_df["gold_label"].value_counts()
    for label, count in label_counts.items():
        display_label = label_map.get(label, label)
        print(f"  {display_label}: {count}")

    # Build Retrieval Function
    print("\n" + "=" * 60)
    print("Building Retrieval Function")
    print("=" * 60)

    retriever = create_retriever(corpus, top_k=3, prefer_embeddings=True)
    print("✓ Retriever initialized")

    # Initialize LLM Interface
    print("\n" + "=" * 60)
    print("Initializing LLM Interface")
    print("=" * 60)

    allowed_labels = [doc[CONFIG.doc_id_field] for doc in corpus]
    llm = LLMInterface(allowed_labels, cache_file=CACHE_FILE, config=CONFIG)

    backend_info = llm.get_backend_info()
    print(f"\nBackend: {backend_info['backend']}")
    print(f"Cache size: {backend_info['cache_size']} entries")
    print(f"Cache file: {backend_info['cache_file']}")

    # Run No-RAG Predictions
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
            "rationale": prediction["rationale"],
        })

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(charts_df)} charts")

    no_rag_df = pd.DataFrame(no_rag_results)
    print(f"✓ Completed No-RAG predictions for {len(no_rag_df)} charts")

    # Run RAG Predictions
    print("\n" + "=" * 60)
    print(f"Running RAG({CONFIG.acronym}) Predictions")
    print("=" * 60)

    rag_results = []
    for idx, row in charts_df.iterrows():
        rag_context = build_rag_context(row["chart_text"], retriever, CONFIG)
        prediction = llm.predict(row["chart_text"], rag_context=rag_context)
        rag_results.append({
            "chart_id": row["chart_id"],
            "gold_label": row["gold_label"],
            "predicted_label": prediction["predicted_label"],
            "top3_labels": prediction["top3_labels"],
            "rationale": prediction["rationale"],
            "rag_context": rag_context,
        })

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(charts_df)} charts")

    rag_df = pd.DataFrame(rag_results)
    print(f"✓ Completed RAG({CONFIG.acronym}) predictions for {len(rag_df)} charts")

    # Evaluation
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    agreement_no_rag = calculate_agreement(no_rag_df)
    agreement_rag = calculate_agreement(rag_df)
    agreement_at3_no_rag = calculate_agreement_at_k(no_rag_df, k=3)
    agreement_at3_rag = calculate_agreement_at_k(rag_df, k=3)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"No-RAG Exact Agreement:    {agreement_no_rag:.1f}%")
    print(f"RAG({CONFIG.acronym}) Exact Agreement:  {agreement_rag:.1f}%")
    print(f"Improvement:               {(agreement_rag - agreement_no_rag):+.1f} percentage points")
    print()
    print(f"No-RAG Agreement@3:        {agreement_at3_no_rag:.1f}%")
    print(f"RAG({CONFIG.acronym}) Agreement@3:      {agreement_at3_rag:.1f}%")
    print("=" * 60)

    # Confusion matrices
    print("\nConfusion Matrix - No-RAG:")
    print("(Rows: Gold Label, Columns: Predicted Label)")
    cm_no_rag = create_confusion_matrix(no_rag_df, label_map)
    print(cm_no_rag)

    print("\n" + "=" * 60)
    print(f"\nConfusion Matrix - RAG({CONFIG.acronym}):")
    print("(Rows: Gold Label, Columns: Predicted Label)")
    cm_rag = create_confusion_matrix(rag_df, label_map)
    print(cm_rag)

    # Save artifacts
    print("\n" + "=" * 60)
    print("Saving Artifacts")
    print("=" * 60)

    results = {
        "metadata": {
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": {
                "ontology_acronym": CONFIG.acronym,
                "ontology_name": CONFIG.name,
                "disease_name": CONFIG.disease_name,
                "organ": CONFIG.organ,
            },
            "random_seed": RANDOM_SEED,
            "n_charts": len(charts_df),
            "n_disease": int((charts_df["gold_label"] != "NONE").sum()),
            "n_none": int((charts_df["gold_label"] == "NONE").sum()),
            "n_classes": len(corpus),
            "retrieval_method": "embeddings" if retriever.__class__.__name__ == "EmbeddingRetriever" else "tfidf",
            "llm_backend": backend_info["backend"],
        },
        "metrics": {
            "no_rag": {
                "exact_agreement": float(agreement_no_rag),
                "agreement_at_3": float(agreement_at3_no_rag),
            },
            f"rag_{CONFIG.acronym.lower()}": {
                "exact_agreement": float(agreement_rag),
                "agreement_at_3": float(agreement_at3_rag),
            },
            "improvement": {
                "exact_agreement_delta": float(agreement_rag - agreement_no_rag),
                "agreement_at_3_delta": float(agreement_at3_rag - agreement_at3_no_rag),
            },
        },
        "label_distribution": charts_df["gold_label"].value_counts().to_dict(),
        "ontology_classes": {doc[CONFIG.doc_id_field]: doc["label"] for doc in corpus},
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✓ Saved results.json")

    error_examples_no_rag = extract_error_examples(no_rag_df, charts_df, label_map, n_examples=3)
    error_examples_rag = extract_error_examples(rag_df, charts_df, label_map, n_examples=3)

    with open("examples.md", "w") as f:
        f.write("# Representative Examples\n\n")
        f.write(f"This file contains representative error examples from both No-RAG and RAG({CONFIG.acronym}) conditions.\n\n")

        f.write("## No-RAG Error Examples\n\n")
        for i, ex in enumerate(error_examples_no_rag, 1):
            f.write(f"### Example {i}\n\n")
            f.write(f"**Chart:** {ex['chart_text']}\n\n")
            f.write(f"**Gold Label:** {ex['gold_label']}\n\n")
            f.write(f"**Predicted:** {ex['predicted_label']}\n\n")
            f.write(f"**Rationale:** {ex['rationale']}\n\n")
            f.write("---\n\n")

        f.write(f"## RAG({CONFIG.acronym}) Error Examples\n\n")
        for i, ex in enumerate(error_examples_rag, 1):
            f.write(f"### Example {i}\n\n")
            f.write(f"**Chart:** {ex['chart_text']}\n\n")
            f.write(f"**Gold Label:** {ex['gold_label']}\n\n")
            f.write(f"**Predicted:** {ex['predicted_label']}\n\n")
            f.write(f"**Rationale:** {ex['rationale']}\n\n")
            f.write("---\n\n")

    print("✓ Saved examples.md")

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Configuration: {CONFIG.disease_name.title()} ({CONFIG.acronym})")
    print(f"Total Charts: {len(charts_df)}")
    print(f"  {CONFIG.disease_name.title()}: {(charts_df['gold_label'] != 'NONE').sum()}")
    print(f"  NONE (distractors): {(charts_df['gold_label'] == 'NONE').sum()}")
    print(f"\n{CONFIG.acronym} Classes Used: {len(corpus)}")
    for doc in corpus:
        print(f"  - {doc['label']}")
    print(f"\nEvaluation Results:")
    print(f"  No-RAG Exact Agreement: {agreement_no_rag:.1f}%")
    print(f"  RAG({CONFIG.acronym}) Exact Agreement: {agreement_rag:.1f}%")
    print(f"  Improvement: {(agreement_rag - agreement_no_rag):+.1f} percentage points")
    print(f"\nArtifacts Saved:")
    print("  - synthetic_charts.csv")
    print(f"  - {CORPUS_FILE}")
    print("  - results.json")
    print("  - examples.md")
    print("  - llm_cache.json")
    print("=" * 60)
    print("\n✓ Execution complete!")


if __name__ == "__main__":
    main()
