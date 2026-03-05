#!/usr/bin/env python3
"""src/rag_context.py

Formats retrieved ontology chunks into a bounded context string for the LLM prompt.
"""

from __future__ import annotations

from typing import Any


def build_rag_context(
    retrieved_docs: list[dict[str, Any]],
    *,
    max_chars: int = 2500,
    max_definition_chars: int = 400,
) -> str:
    """
    Format a list of retrieved corpus records into a prompt context string.
    
    Prioritises:
      1. finding_chunks with WHY/HOW content
      2. diagnosis_chunks
      3. domain_chunks
    """
    if not retrieved_docs:
        return ""

    parts = ["RELEVANT ONTOLOGY KNOWLEDGE:"]

    for i, doc in enumerate(retrieved_docs, 1):
        chunk_type = doc.get("chunk_type", "")
        label = doc.get("label", "")
        score = doc.get("retrieval_score", 0.0)

        parts.append(f"\n[{i}] {label} (relevance: {score:.3f})")

        if doc.get("synonyms"):
            parts.append(f"    Synonyms: {', '.join(doc['synonyms'][:3])}")

        if doc.get("body_systems"):
            parts.append(f"    Body system: {', '.join(doc['body_systems'])}")

        if chunk_type == "finding_chunk":
            why = doc.get("definition_why", "")
            how = doc.get("definition_how", "")
            definition = doc.get("definition", "")

            if why:
                clipped = why[:max_definition_chars]
                if len(why) > max_definition_chars:
                    clipped += "…"
                parts.append(f"    Diagnostic relevance: {clipped}")
            else:
                # v3 corpus: fall back to definition_what (no top-level 'definition' field)
                what = doc.get("definition_what", "")
                if what:
                    clipped = what[:max_definition_chars]
                    if len(what) > max_definition_chars:
                        clipped += "…"
                    parts.append(f"    Description: {clipped}")

            if how:
                clipped = how[:200]
                if len(how) > 200:
                    clipped += "…"
                parts.append(f"    Assessment: {clipped}")

            if doc.get("linked_dx_codes"):
                parts.append(f"    Linked diagnoses: {', '.join(doc['linked_dx_codes'])}")

        elif chunk_type == "diagnosis_chunk":
            text = doc.get("text", "")
            # Show clinical keywords line if present
            for line in text.split("\n"):
                if line.strip().startswith("Clinical keywords:"):
                    parts.append(f"    {line.strip()}")
                    break
            # Show associated findings if any
            if "Associated clinical findings" in text:
                findings_section = text.split("Associated clinical findings")[1]
                clipped = findings_section[:max_definition_chars]
                if len(findings_section) > max_definition_chars:
                    clipped += "…"
                parts.append(f"    Associated findings by system:{clipped}")

        elif chunk_type == "domain_chunk":
            if doc.get("child_labels"):
                sample = doc["child_labels"][:8]
                parts.append(f"    Findings in this domain: {', '.join(sample)}")

    context = "\n".join(parts).strip()

    # Hard cap
    if len(context) > max_chars:
        context = context[:max_chars - 1] + "…"

    return context
