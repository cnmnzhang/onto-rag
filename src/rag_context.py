"""RAG context builder.

Given chart-like input text and a retriever, formats a bounded prompt context.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from onto_config import OntologyConfig, format_template


def _clip(text: str, max_chars: int) -> str:
    text = str(text)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "…"


def build_rag_context(
    chart_text: str,
    retriever,
    config: OntologyConfig,
    *,
    top_k: int = 3,
    max_chars: int = 2000,
    max_synonyms: int = 3,
    max_definition_chars: int = 320,
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

    context = "\n".join(parts).strip()
    return _clip(context, max_chars)
