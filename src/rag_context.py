"""RAG context builder.

Given chart-like input text and a retriever, formats a bounded prompt context.
"""

from __future__ import annotations

from typing import Dict, List

from classes.onto_config import OntologyConfig, format_template


def _clip(text: str, max_chars: int) -> str:
    text = str(text)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "…"


def _join_labels(values: object, *, max_items: int, max_chars: int) -> str:
    if not isinstance(values, list):
        return ""
    labels = [str(v).strip() for v in values if str(v).strip()]
    if not labels:
        return ""
    return _clip(", ".join(labels[:max_items]), max_chars)


def build_rag_context(
    chart_text: str,
    retriever,
    config: OntologyConfig,
    *,
    top_k: int = 3,
    max_chars: int = 2000,
    max_synonyms: int = 3,
    max_definition_chars: int = 320,
    max_hierarchy_items: int = 4,
) -> str:
    """Build bounded RAG context.

    The returned string is capped to `max_chars` to avoid giant prompts.
    """

    retrieved: List[Dict] = retriever.retrieve(chart_text)
    retrieved = list(retrieved)[:top_k]

    context_header = format_template(config.rag_context_header, config)
    parts: List[str] = [context_header]

    for i, doc in enumerate(retrieved, 1):
        label = doc.get("label") or ""
        parts.append(f"\n{i}. {label}")

        synonyms = doc.get("synonyms") or []
        if synonyms:
            syn = ", ".join([str(s) for s in synonyms[:max_synonyms] if s])
            if syn:
                parts.append(f"   Synonyms: {_clip(syn, 200)}")

        definition = doc.get("definition") or ""
        if definition:
            parts.append(f"   Definition: {_clip(str(definition), max_definition_chars)}")

        parent_labels = doc.get("parent_labels") or []
        if parent_labels:
            parents = ", ".join([str(p) for p in parent_labels[:4] if p])
            if parents:
                parts.append(f"   Parents: {_clip(parents, 220)}")

        ancestor_labels = _join_labels(
            doc.get("ancestor_labels"),
            max_items=max_hierarchy_items,
            max_chars=220,
        )
        if ancestor_labels:
            parts.append(f"   Broader classes: {ancestor_labels}")

        sibling_labels = _join_labels(
            doc.get("sibling_labels"),
            max_items=max_hierarchy_items,
            max_chars=220,
        )
        if sibling_labels:
            parts.append(f"   Sibling classes: {sibling_labels}")

        child_labels = _join_labels(
            doc.get("child_labels"),
            max_items=max_hierarchy_items,
            max_chars=220,
        )
        if child_labels:
            parts.append(f"   Child classes: {child_labels}")

    context = "\n".join(parts).strip()
    return _clip(context, max_chars)
