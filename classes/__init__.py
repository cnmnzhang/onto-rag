"""Convenience imports for the project's core Python classes.

This package exists to make it easy to discover and import the primary classes
without forcing a large refactor of the module layout.

Example:
    from classes import OntologyConfig, LLMInterface, OntoDoc
"""

from __future__ import annotations

from .corpus import OntoDoc
from .label_alias import LabelNormalizer
from .llm_interface import LLMInterface
from .onto_config import OntologyConfig
from .retrievers import EmbeddingRetriever, FaissRetriever, TFIDFRetriever

__all__ = [
    "EmbeddingRetriever",
    "FaissRetriever",
    "LabelNormalizer",
    "LLMInterface",
    "OntologyConfig",
    "OntoDoc",
    "TFIDFRetriever",
]
