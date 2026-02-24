from __future__ import annotations

from typing import Dict, List
import random

import pandas as pd
from classes.onto_config import OntologyConfig, format_template


def generate_disease_chart(
    domain_id: str,
    label: str,
    config: OntologyConfig,
) -> Dict:
    """Generate a synthetic disease-domain patient chart."""
    age = random.randint(*config.disease_age_range)
    sex = random.choice(["M", "F"])
    duration = random.choice(config.disease_durations)

    symptoms = random.choice(config.disease_symptoms)

    # Format exam findings with config values
    exam_template = random.choice(config.exam_templates)
    exam_findings = format_template(exam_template, config)

    # Format imaging with config values
    imaging_template = random.choice(config.imaging_templates)
    imaging = format_template(imaging_template, config)

    # Pathology doesn't need formatting (no placeholders)
    pathology = random.choice(config.pathology_templates)

    # Format distractor with config values
    distractor_template = random.choice(config.distractor_templates)
    distractor = format_template(distractor_template, config)

    # Build chart text
    if config.mass_location.lower() != "n/a":
        chart_text = (
            f"{age}-year-old {sex} presents with {duration} history of {config.mass_location} mass. "
            f"Reports {symptoms}. "
            f"Physical exam shows {exam_findings}. "
            f"{imaging}. "
            f"{pathology}. "
            f"{distractor}"
        )
    else:
        # For diseases without anatomical mass (e.g., diabetes)
        chart_text = (
            f"{age}-year-old {sex} presents with {duration} history of {symptoms}. "
            f"Physical exam {exam_findings}. "
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


def generate_none_chart(config: OntologyConfig) -> Dict:
    """Generate a NONE (non-domain) distractor chart."""
    age = random.randint(*config.none_age_range)
    sex = random.choice(["M", "F"])
    duration = random.choice(config.none_durations)

    primary_symptom = random.choice(config.none_symptoms)
    secondary_symptoms = random.choice(config.none_secondary_symptoms)

    # Format exam with config values
    exam_template = random.choice(config.none_exam_templates)
    exam = format_template(exam_template, config)

    # Format workup with config values
    workup_template = random.choice(config.none_workup_templates)
    workup = format_template(workup_template, config)

    # Format diagnosis with config values
    diagnosis_template = random.choice(config.none_diagnosis_templates)
    diagnosis = format_template(diagnosis_template, config)

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
    config: OntologyConfig,
    n_domain: int = 60,
    n_none: int = 60,
) -> pd.DataFrame:
    """Generate balanced synthetic dataset."""
    charts = []

    # Generate disease-domain charts (balanced across ontology classes)
    domain_ids = [doc[config.doc_id_field] for doc in corpus]
    for i in range(n_domain):
        domain_id = domain_ids[i % len(domain_ids)]
        label = label_map[domain_id]
        chart = generate_disease_chart(domain_id, label, config)
        charts.append(chart)

    # Generate NONE charts
    for _ in range(n_none):
        chart = generate_none_chart(config)
        charts.append(chart)

    # Shuffle
    random.shuffle(charts)

    df = pd.DataFrame(charts)
    df["chart_id"] = range(len(df))

    return df
