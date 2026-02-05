from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import random

import pandas as pd


@dataclass(frozen=True)
class DiseaseDomainSpec:
    key: str
    name: str
    acronym: Optional[str]
    organ: str
    mass_location: str
    disease_name: str
    doc_id_field: str


TCO_THYROID_DOMAIN = DiseaseDomainSpec(
    key="tco_thyroid",
    name="Thyroid Cancer",
    acronym="TCO",
    organ="thyroid",
    mass_location="anterior neck",
    disease_name="thyroid cancer",
    doc_id_field="tco_id",
)

DOMAIN_SPECS: Dict[str, DiseaseDomainSpec] = {
    TCO_THYROID_DOMAIN.key: TCO_THYROID_DOMAIN,
}


def generate_disease_chart(
    domain_id: str,
    label: str,
    domain: DiseaseDomainSpec,
) -> Dict:
    """Generate a synthetic disease-domain patient chart."""
    age = random.randint(25, 75)
    sex = random.choice(["M", "F"])
    duration = random.choice(["2-month", "3-month", "6-week", "4-month"])

    symptoms = random.choice([
        "dysphagia and hoarseness",
        "mild dysphagia",
        "throat discomfort",
        "difficulty swallowing",
        "voice changes",
        "neck swelling",
    ])

    exam_findings = random.choice([
        f"firm 2.5 cm {domain.organ} nodule",
        f"palpable right {domain.organ} mass",
        f"fixed 3 cm left {domain.organ} nodule",
        f"hard {domain.mass_location} mass",
        f"irregular {domain.organ} enlargement",
    ])

    imaging = random.choice([
        f"Ultrasound shows hypoechoic {domain.organ} nodule with microcalcifications",
        f"CT shows heterogeneous {domain.organ} mass with local invasion",
        f"MRI demonstrates solid {domain.organ} lesion with irregular borders",
        f"PET-CT shows FDG-avid {domain.organ} nodule",
        f"Ultrasound reveals solid {domain.organ} mass with increased vascularity",
    ])

    pathology = random.choice([
        "FNA cytology shows suspicious cells",
        "FNA reveals malignant cells",
        "Biopsy demonstrates atypical cells",
        "FNA shows atypical follicular cells",
        "Cytology consistent with malignancy",
    ])

    distractor = random.choice([
        "Patient also reports mild fatigue.",
        "Patient has history of hyperlipidemia.",
        "Family history of diabetes.",
        f"Patient on levothyroxine for {domain.organ} dysfunction.",
        "Occasional palpitations noted.",
    ])

    chart_text = (
        f"{age}-year-old {sex} presents with {duration} history of {domain.mass_location} mass. "
        f"Reports {symptoms}. "
        f"Physical exam shows {exam_findings}. "
        f"{imaging}. "
        f"{pathology}. "
        f"{distractor}"
    )

    return {
        "chart_text": chart_text,
        "gold_label": domain_id,
        "age": age,
        "sex": sex,
    }


def generate_none_chart(domain: DiseaseDomainSpec) -> Dict:
    """Generate a NONE (non-domain) distractor chart."""
    age = random.randint(20, 70)
    sex = random.choice(["M", "F"])
    duration = random.choice(["2-week", "3-week", "1-month", "2-month"])

    primary_symptom = random.choice([
        "hoarseness",
        "cervical lymphadenopathy",
        "dysphagia",
        "neck pain",
        "fatigue",
        "throat discomfort",
    ])

    secondary_symptoms = random.choice([
        "Reports recent upper respiratory infection",
        "Denies fever or weight loss",
        "Reports concurrent sore throat",
        "Reports voice strain from prolonged speaking",
        "History of seasonal allergies",
    ])

    exam = random.choice([
        f"shows no {domain.organ} masses",
        "reveals reactive cervical lymph nodes",
        f"shows normal {domain.organ} examination",
        "reveals laryngeal inflammation",
        "demonstrates benign findings",
    ])

    workup = random.choice([
        "Laryngoscopy shows vocal cord inflammation",
        f"Ultrasound shows normal {domain.organ}, reactive nodes",
        f"CT shows no {domain.organ} abnormalities",
        f"TSH and {domain.organ} ultrasound normal",
        f"Labs show normal {domain.organ} function",
    ])

    diagnosis = random.choice([
        "Diagnosed with viral laryngitis.",
        "Diagnosed with reactive lymphadenopathy.",
        "Diagnosed with gastroesophageal reflux.",
        "Diagnosed with vocal cord strain.",
        f"Diagnosed with benign {domain.organ} nodule.",
    ])

    chart_text = (
        f"{age}-year-old {sex} presents with {duration} of {primary_symptom}. "
        f"{secondary_symptoms}. "
        f"Physical exam {exam}. "
        f"{workup}. "
        f"{diagnosis}"
    )

    return {
        "chart_text": chart_text,
        "gold_label": "NONE",
        "age": age,
        "sex": sex,
    }


def generate_synthetic_dataset(
    corpus: List[Dict],
    label_map: Dict[str, str],
    n_domain: int = 60,
    n_none: int = 60,
    domain: DiseaseDomainSpec = TCO_THYROID_DOMAIN,
) -> pd.DataFrame:
    """Generate balanced synthetic dataset."""
    charts = []

    # Generate disease-domain charts (balanced across ontology classes)
    domain_ids = [doc[domain.doc_id_field] for doc in corpus]
    for i in range(n_domain):
        domain_id = domain_ids[i % len(domain_ids)]
        label = label_map[domain_id]
        chart = generate_disease_chart(domain_id, label, domain)
        charts.append(chart)

    # Generate NONE charts
    for _ in range(n_none):
        chart = generate_none_chart(domain)
        charts.append(chart)

    # Shuffle
    random.shuffle(charts)

    df = pd.DataFrame(charts)
    df["chart_id"] = range(len(df))

    return df
