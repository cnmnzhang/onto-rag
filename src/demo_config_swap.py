#!/usr/bin/env python3
"""
Demo script showing how to swap disease configurations

This demonstrates the power of the ontology-agnostic design.
Simply change one line to analyze a different disease!
"""

from onto_config import get_config, list_configs, TCO_CONFIG, DIABETES_CONFIG

print("=" * 70)
print("ONTOLOGY CONFIGURATION DEMO")
print("=" * 70)

# Show all available configurations
print("\nAvailable Configurations:")
list_configs()

# Demo 1: Thyroid Cancer (TCO)
print("\n" + "=" * 70)
print("DEMO 1: Thyroid Cancer Configuration")
print("=" * 70)

config = get_config("tco")
print(f"\nDisease: {config.disease_name}")
print(f"Ontology: {config.name} ({config.acronym})")
print(f"Organ: {config.organ}")
print(f"Mass Location: {config.mass_location}")
print(f"Number of classes: {config.num_classes}")
print(f"\nClass keywords: {', '.join(config.class_keywords)}")
print(f"Subtype keywords: {', '.join(config.subtype_keywords[:4])}...")
print(f"\nCorpus file: {config.corpus_filename}")
print(f"Document ID field: {config.doc_id_field}")

# Demo 2: Diabetes
print("\n" + "=" * 70)
print("DEMO 2: Diabetes Configuration")
print("=" * 70)

config = get_config("diabetes")
print(f"\nDisease: {config.disease_name}")
print(f"Ontology: {config.name} ({config.acronym})")
print(f"Organ: {config.organ}")
print(f"Mass Location: {config.mass_location}")
print(f"Number of classes: {config.num_classes}")
print(f"\nClass keywords: {', '.join(config.class_keywords)}")
print(f"Subtype keywords: {', '.join(config.subtype_keywords[:4])}...")
print(f"\nCorpus file: {config.corpus_filename}")
print(f"Document ID field: {config.doc_id_field}")

# Demo 3: Template Formatting
print("\n" + "=" * 70)
print("DEMO 3: Template Formatting")
print("=" * 70)

from onto_config import format_template

tco_template = "Patient has {organ} mass in {mass_location}"
print(f"\nTemplate: {tco_template}")
print(f"TCO: {format_template(tco_template, TCO_CONFIG)}")

diabetes_template = "Patient has elevated {organ} markers"
print(f"\nTemplate: {diabetes_template}")
print(f"Diabetes: {format_template(diabetes_template, DIABETES_CONFIG)}")

# Demo 4: Chart Generation Templates
print("\n" + "=" * 70)
print("DEMO 4: Chart Generation Templates")
print("=" * 70)

print("\nTCO Disease Symptoms:")
for i, symptom in enumerate(TCO_CONFIG.disease_symptoms[:3], 1):
    print(f"  {i}. {symptom}")

print("\nDiabetes Disease Symptoms:")
for i, symptom in enumerate(DIABETES_CONFIG.disease_symptoms[:3], 1):
    print(f"  {i}. {symptom}")

print("\nTCO Exam Templates (with formatting):")
for i, template in enumerate(TCO_CONFIG.exam_templates[:2], 1):
    formatted = format_template(template, TCO_CONFIG)
    print(f"  {i}. {formatted}")

print("\nDiabetes Exam Templates:")
for i, template in enumerate(DIABETES_CONFIG.exam_templates[:2], 1):
    print(f"  {i}. {template}")

# Demo 5: Switching is Easy!
print("\n" + "=" * 70)
print("DEMO 5: How to Switch Configurations in rag_exp.py")
print("=" * 70)

print("""
# NEW WAY (1 line change):
CONFIG = get_config("diabetes")  # Instead of get_config("tco")

# Or directly:
from onto_config import DIABETES_CONFIG
CONFIG = DIABETES_CONFIG

# That's it! Everything else automatically adapts:
# - Ontology queries use DOID instead of TCO
# - Chart generation uses diabetes symptoms
# - LLM prompts mention "diabetes mellitus"
# - Heuristics use diabetes keywords
# - Corpus saved to doid_corpus.jsonl
""")

print("\n" + "=" * 70)
print("✓ Configuration system successfully demonstrated!")
print("=" * 70)
