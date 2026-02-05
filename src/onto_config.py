"""
Ontology Configuration System for Disease-Agnostic RAG Framework

This module provides disease/ontology-specific configurations that can be
swapped to adapt the framework to different medical domains (thyroid cancer,
diabetes, lung cancer, etc.).

To use a different disease, simply change:
    CONFIG = get_config("diabetes")  # Instead of "tco"

Or create a custom configuration:
    custom_config = OntologyConfig(acronym="DOID", disease_name="diabetes", ...)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class OntologyConfig:
    """
    Configuration for a disease ontology and clinical domain.

    This dataclass defines all disease-specific parameters needed for:
    - Ontology selection and class filtering
    - Synthetic patient chart generation
    - RAG context formatting
    - LLM system prompts
    - Dry-run heuristics
    """

    # === Ontology Metadata ===
    acronym: str
    """Ontology acronym (e.g., 'TCO', 'DOID', 'ICD10')"""

    name: str
    """Full ontology name (e.g., 'Thyroid Cancer Ontology')"""

    disease_name: str
    """Disease name for prompts (e.g., 'thyroid cancer', 'diabetes mellitus')"""

    # === Anatomical/Domain Information ===
    organ: str
    """Primary organ (e.g., 'thyroid', 'lung', 'pancreas')"""

    mass_location: str
    """Anatomical location for mass/lesion (e.g., 'anterior neck', 'chest', 'abdomen')
    Use 'N/A' if not applicable (e.g., diabetes)"""

    # === Ontology Class Selection ===
    class_keywords: List[str]
    """Keywords to identify relevant ontology classes (e.g., ['carcinoma', 'cancer'])"""

    subtype_keywords: List[str]
    """Keywords for diverse subtype selection (e.g., ['papillary', 'follicular'])"""

    num_classes: int = 8
    """Target number of ontology classes to select"""

    # === Chart Generation - Disease Cases ===
    disease_age_range: Tuple[int, int] = (25, 75)
    """Age range for disease cases (min, max)"""

    disease_durations: List[str] = field(default_factory=lambda: [
        "2-month", "3-month", "6-week", "4-month"
    ])
    """Duration options for disease history"""

    disease_symptoms: List[str] = field(default_factory=list)
    """Clinical symptoms for disease cases"""

    exam_templates: List[str] = field(default_factory=list)
    """Physical exam findings templates (use {organ}, {mass_location} placeholders)"""

    imaging_templates: List[str] = field(default_factory=list)
    """Imaging findings templates (use {organ} placeholder)"""

    pathology_templates: List[str] = field(default_factory=list)
    """Pathology/biopsy findings templates"""

    distractor_templates: List[str] = field(default_factory=list)
    """Distractor findings (non-specific symptoms)"""

    # === Chart Generation - Negative Cases ===
    none_age_range: Tuple[int, int] = (20, 70)
    """Age range for NONE (negative) cases"""

    none_durations: List[str] = field(default_factory=lambda: [
        "2-week", "3-week", "1-month", "2-month"
    ])
    """Duration options for NONE cases (typically shorter)"""

    none_symptoms: List[str] = field(default_factory=list)
    """Primary symptoms for NONE cases (may overlap with disease)"""

    none_secondary_symptoms: List[str] = field(default_factory=list)
    """Secondary symptoms/context for NONE cases"""

    none_exam_templates: List[str] = field(default_factory=list)
    """Exam findings for NONE cases (use {organ} placeholder)"""

    none_workup_templates: List[str] = field(default_factory=list)
    """Workup/investigation findings for NONE cases"""

    none_diagnosis_templates: List[str] = field(default_factory=list)
    """Final diagnoses for NONE cases"""

    # === LLM System Prompt ===
    system_prompt_template: str = ""
    """System prompt template for LLM (use {disease_name}, {allowed_labels} placeholders)"""

    # === RAG Context ===
    rag_context_header: str = "Relevant classifications from {name}:"
    """Header for RAG context (use {name}, {acronym} placeholders)"""

    # === Dry-Run Heuristic Keywords ===
    positive_keywords: List[str] = field(default_factory=list)
    """Keywords that suggest disease is present (for dry-run mode)"""

    negative_keywords: List[str] = field(default_factory=list)
    """Keywords that suggest disease is absent (for dry-run mode)"""

    # === File Naming ===
    corpus_filename: str = "corpus.jsonl"
    """Filename for ontology corpus cache"""

    doc_id_field: str = "doc_id"
    """Field name for document ID in corpus (e.g., 'tco_id', 'doid_id')"""

    # === Dataset Parameters ===
    n_disease_charts: int = 60
    """Number of disease charts to generate"""

    n_none_charts: int = 60
    """Number of NONE charts to generate"""


# =============================================================================
# Pre-Built Configurations
# =============================================================================

TCO_CONFIG = OntologyConfig(
    # Ontology Metadata
    acronym="TCO",
    name="Thyroid Cancer Ontology",
    disease_name="thyroid cancer",

    # Anatomical Info
    organ="thyroid",
    mass_location="anterior neck",

    # Class Selection
    class_keywords=["carcinoma", "cancer", "tumor", "neoplasm", "adenoma"],
    subtype_keywords=[
        "papillary", "follicular", "medullary", "anaplastic",
        "poorly differentiated", "hurthle cell", "clear cell", "insular"
    ],
    num_classes=8,

    # Disease Chart Generation
    disease_age_range=(25, 75),
    disease_durations=["2-month", "3-month", "6-week", "4-month"],
    disease_symptoms=[
        "dysphagia and hoarseness",
        "mild dysphagia",
        "throat discomfort",
        "difficulty swallowing",
        "voice changes",
        "neck swelling",
    ],
    exam_templates=[
        "firm 2.5 cm {organ} nodule",
        "palpable right {organ} mass",
        "fixed 3 cm left {organ} nodule",
        "hard {mass_location} mass",
        "irregular {organ} enlargement",
    ],
    imaging_templates=[
        "Ultrasound shows hypoechoic {organ} nodule with microcalcifications",
        "CT shows heterogeneous {organ} mass with local invasion",
        "MRI demonstrates solid {organ} lesion with irregular borders",
        "PET-CT shows FDG-avid {organ} nodule",
        "Ultrasound reveals solid {organ} mass with increased vascularity",
    ],
    pathology_templates=[
        "FNA cytology shows suspicious cells",
        "FNA reveals malignant cells",
        "Biopsy demonstrates atypical cells",
        "FNA shows atypical follicular cells",
        "Cytology consistent with malignancy",
    ],
    distractor_templates=[
        "Patient also reports mild fatigue.",
        "Patient has history of hyperlipidemia.",
        "Family history of diabetes.",
        "Patient on levothyroxine for {organ} dysfunction.",
        "Occasional palpitations noted.",
    ],

    # NONE Chart Generation
    none_age_range=(20, 70),
    none_durations=["2-week", "3-week", "1-month", "2-month"],
    none_symptoms=[
        "hoarseness",
        "cervical lymphadenopathy",
        "dysphagia",
        "neck pain",
        "fatigue",
        "throat discomfort",
    ],
    none_secondary_symptoms=[
        "Reports recent upper respiratory infection",
        "Denies fever or weight loss",
        "Reports concurrent sore throat",
        "Reports voice strain from prolonged speaking",
        "History of seasonal allergies",
    ],
    none_exam_templates=[
        "shows no {organ} masses",
        "reveals reactive cervical lymph nodes",
        "shows normal {organ} examination",
        "reveals laryngeal inflammation",
        "demonstrates benign findings",
    ],
    none_workup_templates=[
        "Laryngoscopy shows vocal cord inflammation",
        "Ultrasound shows normal {organ}, reactive nodes",
        "CT shows no {organ} abnormalities",
        "TSH and {organ} ultrasound normal",
        "Labs show normal {organ} function",
    ],
    none_diagnosis_templates=[
        "Diagnosed with viral laryngitis.",
        "Diagnosed with reactive lymphadenopathy.",
        "Diagnosed with gastroesophageal reflux.",
        "Diagnosed with vocal cord strain.",
        "Diagnosed with benign {organ} nodule.",
    ],

    # System Prompt
    system_prompt_template="""You are a clinical diagnosis assistant. Given a patient chart,
predict the most likely {disease_name} diagnosis from the allowed label set.

ALLOWED LABELS: {allowed_labels}, NONE

Output valid JSON only:
{{"predicted_label": "<label>", "top3_labels": ["<label1>", "<label2>", "<label3>"], "rationale": "<brief explanation>"}}

If no {disease_name} is evident, return "NONE" as the predicted_label.
Use the full IRI (http://...) for {disease_name} labels, not just the short name.""",

    # RAG Context
    rag_context_header="Relevant thyroid cancer classifications from TCO:",

    # Dry-Run Heuristics
    positive_keywords=[
        "malignant", "fna reveals malignant", "carcinoma",
        "thyroid mass", "thyroid nodule", "fna cytology shows suspicious",
        "atypical cells", "suspicious cells"
    ],
    negative_keywords=[
        "normal thyroid", "no thyroid", "tsh", "benign",
        "reactive", "viral", "normal examination"
    ],

    # File Naming
    corpus_filename="tco_corpus.jsonl",
    doc_id_field="tco_id",

    # Dataset
    n_disease_charts=60,
    n_none_charts=60,
)


DIABETES_CONFIG = OntologyConfig(
    # Ontology Metadata
    acronym="DOID",
    name="Disease Ontology",
    disease_name="diabetes mellitus",

    # Anatomical Info
    organ="pancreas",
    mass_location="N/A",  # No anatomical mass for diabetes

    # Class Selection
    class_keywords=["diabetes", "hyperglycemia", "insulin resistance", "glucose"],
    subtype_keywords=[
        "type 1", "type 2", "gestational", "MODY", "LADA",
        "insulin-dependent", "non-insulin-dependent"
    ],
    num_classes=8,

    # Disease Chart Generation
    disease_age_range=(20, 75),
    disease_symptoms=[
        "polyuria and polydipsia",
        "polydipsia and weight loss",
        "fatigue and frequent urination",
        "unexplained weight loss",
        "increased thirst and hunger",
        "blurred vision",
    ],
    exam_templates=[
        "BMI {bmi}, acanthosis nigricans noted",
        "thin habitus, signs of recent weight loss",
        "BMI 32, central obesity present",
        "normal physical examination",
        "signs of peripheral neuropathy",
    ],
    imaging_templates=[
        "Pancreatic imaging shows atrophy",
        "No acute findings on abdominal imaging",
        "CT abdomen shows fatty liver",
        "Pancreas appears normal on imaging",
    ],
    pathology_templates=[
        "Fasting glucose 280 mg/dL",
        "HbA1c 9.5%",
        "Random glucose 320 mg/dL",
        "Glucose tolerance test positive",
        "C-peptide low, consistent with type 1",
    ],
    distractor_templates=[
        "Patient also reports mild headaches.",
        "Family history of hypertension.",
        "Patient on atorvastatin for hyperlipidemia.",
        "Occasional muscle cramps noted.",
    ],

    # NONE Chart Generation
    none_symptoms=[
        "mild fatigue",
        "occasional thirst",
        "recent weight changes",
        "intermittent nausea",
    ],
    none_secondary_symptoms=[
        "Reports recent stress at work",
        "Denies polyuria or polydipsia",
        "Reports adequate hydration",
        "Normal appetite and weight",
    ],
    none_exam_templates=[
        "shows normal BMI and examination",
        "reveals no signs of hyperglycemia",
        "demonstrates normal hydration",
    ],
    none_workup_templates=[
        "Fasting glucose 92 mg/dL (normal)",
        "HbA1c 5.4% (normal)",
        "Glucose tolerance test normal",
        "Random glucose 105 mg/dL (normal)",
    ],
    none_diagnosis_templates=[
        "Diagnosed with anxiety disorder.",
        "Diagnosed with chronic fatigue syndrome.",
        "Diagnosed with stress reaction.",
        "Diagnosed with adjustment disorder.",
    ],

    # System Prompt
    system_prompt_template="""You are a clinical diagnosis assistant. Given a patient chart,
predict the most likely {disease_name} diagnosis from the allowed label set.

ALLOWED LABELS: {allowed_labels}, NONE

Output valid JSON only:
{{"predicted_label": "<label>", "top3_labels": ["<label1>", "<label2>", "<label3>"], "rationale": "<brief explanation>"}}

If no {disease_name} is evident, return "NONE" as the predicted_label.""",

    # RAG Context
    rag_context_header="Relevant diabetes classifications from DOID:",

    # Dry-Run Heuristics
    positive_keywords=[
        "hyperglycemia", "elevated glucose", "polyuria", "polydipsia",
        "hba1c", "insulin", "diabetic"
    ],
    negative_keywords=[
        "normal glucose", "normal hba1c", "euglycemic", "no diabetes"
    ],

    # File Naming
    corpus_filename="doid_corpus.jsonl",
    doc_id_field="doid_id",
)


LUNG_CANCER_CONFIG = OntologyConfig(
    # Ontology Metadata
    acronym="NCIT",
    name="National Cancer Institute Thesaurus",
    disease_name="lung cancer",

    # Anatomical Info
    organ="lung",
    mass_location="chest",

    # Class Selection
    class_keywords=["carcinoma", "cancer", "neoplasm", "tumor"],
    subtype_keywords=[
        "adenocarcinoma", "squamous cell", "small cell",
        "large cell", "non-small cell", "bronchioloalveolar"
    ],
    num_classes=8,

    # Disease Chart Generation
    disease_age_range=(45, 85),
    disease_symptoms=[
        "persistent cough and hemoptysis",
        "chronic cough",
        "dyspnea and chest pain",
        "hemoptysis",
        "weight loss and fatigue",
        "chest discomfort",
    ],
    exam_templates=[
        "diminished breath sounds on right",
        "dullness to percussion over {mass_location}",
        "wheezing on left {organ} fields",
        "decreased air entry right base",
    ],
    imaging_templates=[
        "Chest X-ray shows right upper lobe mass",
        "CT {mass_location} demonstrates 3 cm spiculated nodule",
        "PET-CT shows FDG-avid {organ} mass with mediastinal involvement",
        "MRI reveals {organ} lesion with pleural invasion",
    ],
    pathology_templates=[
        "Bronchoscopy with biopsy shows malignant cells",
        "CT-guided biopsy positive for carcinoma",
        "Sputum cytology positive for malignancy",
        "Transbronchial biopsy demonstrates cancer cells",
    ],
    distractor_templates=[
        "Patient reports 40 pack-year smoking history.",
        "Patient has history of COPD.",
        "Family history of lung disease.",
        "Recent upper respiratory infection noted.",
    ],

    # NONE Chart Generation
    none_symptoms=[
        "chronic cough",
        "mild dyspnea",
        "chest tightness",
        "recent cold symptoms",
    ],
    none_secondary_symptoms=[
        "Reports recent bronchitis",
        "Denies hemoptysis or weight loss",
        "Reports improvement with bronchodilators",
        "History of seasonal allergies",
    ],
    none_exam_templates=[
        "shows clear {organ} fields",
        "reveals no masses on examination",
        "demonstrates normal respiratory exam",
    ],
    none_workup_templates=[
        "Chest X-ray shows no acute findings",
        "CT {mass_location} shows no masses",
        "Bronchoscopy reveals mild inflammation only",
        "Sputum culture positive for bacteria, no malignancy",
    ],
    none_diagnosis_templates=[
        "Diagnosed with community-acquired pneumonia.",
        "Diagnosed with chronic bronchitis.",
        "Diagnosed with asthma exacerbation.",
        "Diagnosed with upper respiratory infection.",
    ],

    # System Prompt
    system_prompt_template="""You are a clinical diagnosis assistant. Given a patient chart,
predict the most likely {disease_name} diagnosis from the allowed label set.

ALLOWED LABELS: {allowed_labels}, NONE

Output valid JSON only:
{{"predicted_label": "<label>", "top3_labels": ["<label1>", "<label2>", "<label3>"], "rationale": "<brief explanation>"}}

If no {disease_name} is evident, return "NONE" as the predicted_label.""",

    # RAG Context
    rag_context_header="Relevant lung cancer classifications from NCIT:",

    # Dry-Run Heuristics
    positive_keywords=[
        "malignant", "carcinoma", "hemoptysis", "lung mass",
        "spiculated nodule", "mediastinal", "metastatic"
    ],
    negative_keywords=[
        "normal chest", "no masses", "clear lungs", "pneumonia", "bronchitis"
    ],

    # File Naming
    corpus_filename="ncit_corpus.jsonl",
    doc_id_field="ncit_id",
)


# =============================================================================
# Registry and Helper Functions
# =============================================================================

ONTOLOGY_CONFIGS: Dict[str, OntologyConfig] = {
    "tco": TCO_CONFIG,
    "diabetes": DIABETES_CONFIG,
    "lung_cancer": LUNG_CANCER_CONFIG,
}


def get_config(key: str) -> OntologyConfig:
    """
    Get ontology configuration by key.

    Args:
        key: Configuration key ('tco', 'diabetes', 'lung_cancer', etc.)

    Returns:
        OntologyConfig instance

    Raises:
        ValueError: If key is not found in registry

    Example:
        >>> config = get_config("tco")
        >>> config.disease_name
        'thyroid cancer'
    """
    if key not in ONTOLOGY_CONFIGS:
        available = ", ".join(ONTOLOGY_CONFIGS.keys())
        raise ValueError(
            f"Unknown config key '{key}'. Available configurations: {available}"
        )
    return ONTOLOGY_CONFIGS[key]


def format_template(template: str, config: OntologyConfig, **kwargs) -> str:
    """
    Format a template string with config values and additional kwargs.

    Args:
        template: Template string with {placeholders}
        config: OntologyConfig instance
        **kwargs: Additional format arguments

    Returns:
        Formatted string

    Example:
        >>> config = TCO_CONFIG
        >>> template = "Exam shows {organ} mass in {mass_location}"
        >>> format_template(template, config)
        'Exam shows thyroid mass in anterior neck'
    """
    format_dict = {
        "organ": config.organ,
        "mass_location": config.mass_location,
        "disease_name": config.disease_name,
        "acronym": config.acronym,
        "name": config.name,
        **kwargs
    }
    return template.format(**format_dict)


def build_system_prompt(config: OntologyConfig, allowed_labels: List[str]) -> str:
    """
    Build LLM system prompt from config.

    Args:
        config: OntologyConfig instance
        allowed_labels: List of allowed label IRIs/IDs

    Returns:
        Formatted system prompt string

    Example:
        >>> config = TCO_CONFIG
        >>> labels = ["http://purl.../TCO_0000123", "http://purl.../TCO_0000456"]
        >>> prompt = build_system_prompt(config, labels)
        >>> "thyroid cancer" in prompt
        True
    """
    allowed_str = ", ".join(sorted([l for l in allowed_labels if l != "NONE"]))
    return config.system_prompt_template.format(
        disease_name=config.disease_name,
        allowed_labels=allowed_str
    )


def list_configs() -> None:
    """
    Print available configurations.

    Example:
        >>> list_configs()
        Available ontology configurations:
          - tco: Thyroid Cancer Ontology (TCO)
          - diabetes: Disease Ontology (DOID)
          - lung_cancer: National Cancer Institute Thesaurus (NCIT)
    """
    print("Available ontology configurations:")
    for key, config in ONTOLOGY_CONFIGS.items():
        print(f"  - {key}: {config.name} ({config.acronym})")


if __name__ == "__main__":
    # Demo usage
    print("Ontology Configuration Demo")
    print("=" * 60)

    list_configs()

    print("\nTCO Configuration:")
    print(f"  Disease: {TCO_CONFIG.disease_name}")
    print(f"  Organ: {TCO_CONFIG.organ}")
    print(f"  Classes to select: {TCO_CONFIG.num_classes}")
    print(f"  Corpus file: {TCO_CONFIG.corpus_filename}")

    print("\nExample template formatting:")
    template = "Patient has {organ} mass in {mass_location}"
    print(f"  Template: {template}")
    print(f"  Formatted: {format_template(template, TCO_CONFIG)}")
