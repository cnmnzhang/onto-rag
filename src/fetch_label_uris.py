#!/usr/bin/env python3
"""Fetch and write label-set URIs from BioPortal.

Default mode targets AI-RHEUM and prefers diagnosis-class IDs (DX* tails),
which are richer than the bare concept IDs for this ontology.

Usage:
  python3 src/fetch_label_uris.py

Writes:
  data/ai_rheum_label_set.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None


BASE = "https://data.bioontology.org"
DEFAULT_ONTOLOGY = "AI-RHEUM"
DEFAULT_NONE_LABEL = "NONE"

# Curated disease concepts for AI-RHEUM label-set generation.
AI_RHEUM_CONCEPTS: list[str] = [
    "Rheumatoid arthritis",
    "Systemic lupus erythematosus",
    "Gout",
    "Psoriatic arthritis",
    "Ankylosing spondylitis",
    "Polymyalgia rheumatica",
]

# Prefer diagnosis branch IDs for AIR when available.
AI_RHEUM_PREFERRED_TAILS: dict[str, str] = {
    "Rheumatoid arthritis": "DXRA",
    "Systemic lupus erythematosus": "DXSLE",
    "Gout": "DXGT",
    "Psoriatic arthritis": "DXPSO",
    "Ankylosing spondylitis": "DXANK",
    "Polymyalgia rheumatica": "DXPMR",
}


def _uri_tail(uri: str) -> str:
    s = str(uri).strip()
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _norm(s: str) -> str:
    return " ".join(str(s).strip().split()).casefold()


def _bp_get(endpoint: str, *, params: dict[str, str]) -> dict[str, Any]:
    api_key = os.getenv("BIOPORTAL_API_KEY")
    p = dict(params)
    if api_key:
        p["apikey"] = api_key

    url = f"{BASE}{endpoint}"
    r = requests.get(url, params=p, timeout=30)

    if r.status_code == 401:
        raise requests.HTTPError(
            "BioPortal returned 401 Unauthorized. "
            "Set BIOPORTAL_API_KEY in your shell or .env.",
            response=r,
        )

    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected JSON from BioPortal: {type(data)}")
    return data


def fetch_search_hits(query: str, *, ontology: str, pagesize: int = 20) -> list[dict[str, Any]]:
    data = _bp_get(
        "/search",
        params={
            "q": query,
            "ontologies": ontology,
            "pagesize": str(pagesize),
        },
    )
    collection = data.get("collection")
    if not isinstance(collection, list):
        return []
    return [c for c in collection if isinstance(c, dict)]


def _pick_primary_ai_rheum(concept: str, hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not hits:
        return None

    concept_norm = _norm(concept)
    preferred_tail = AI_RHEUM_PREFERRED_TAILS.get(concept, "")
    exact_pref = [h for h in hits if _norm(str(h.get("prefLabel") or "")) == concept_norm]

    # 1) Exact preferred DX tail for the concept.
    if preferred_tail:
        for h in exact_pref:
            if _uri_tail(str(h.get("@id") or "")) == preferred_tail:
                return h

    # 2) Any exact-prefLabel DX* hit.
    for h in exact_pref:
        if _uri_tail(str(h.get("@id") or "")).startswith("DX"):
            return h

    # 3) Any exact-prefLabel hit.
    if exact_pref:
        return exact_pref[0]

    # 4) Any DX* hit in search results.
    for h in hits:
        if _uri_tail(str(h.get("@id") or "")).startswith("DX"):
            return h

    # 5) Fallback to first result.
    return hits[0]


def pick_primary(concept: str, hits: list[dict[str, Any]], *, ontology: str) -> dict[str, Any] | None:
    if ontology.upper() == "AI-RHEUM":
        return _pick_primary_ai_rheum(concept, hits)
    return hits[0] if hits else None


def _fmt(hit: dict[str, Any]) -> dict[str, Any]:
    uri = str(hit.get("@id") or "")
    return {
        "prefLabel": hit.get("prefLabel"),
        "@id": uri,
        "tail": _uri_tail(uri),
        "matchType": hit.get("matchType"),
    }


def _default_output_path(ontology: str) -> Path:
    if ontology.upper() == "AI-RHEUM":
        return Path("data/ai_rheum_label_set.json")
    return Path("data/label_set.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch canonical label URIs from BioPortal and write label_set JSON.")
    p.add_argument("--ontology", default=DEFAULT_ONTOLOGY, help="Ontology acronym, e.g., AI-RHEUM")
    p.add_argument("--output", default=None, help="Output label_set JSON path")
    p.add_argument("--none-label", default=DEFAULT_NONE_LABEL, help="none_label sentinel in output JSON")
    p.add_argument("--pagesize", type=int, default=20, help="BioPortal search pagesize")
    return p.parse_args()


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()

    args = _parse_args()
    ontology = str(args.ontology).strip() or DEFAULT_ONTOLOGY
    none_label = str(args.none_label).strip() or DEFAULT_NONE_LABEL
    output_path = Path(args.output) if args.output else _default_output_path(ontology)

    if ontology.upper() != "AI-RHEUM":
        print(
            f"Unsupported ontology for this curated script: {ontology}. "
            f"Supported: AI-RHEUM.",
            file=sys.stderr,
        )
        return 2

    concepts = list(AI_RHEUM_CONCEPTS)
    results: list[dict[str, Any]] = []
    label_ids: list[str] = []

    for concept in concepts:
        hits = fetch_search_hits(concept, ontology=ontology, pagesize=int(args.pagesize))
        primary = pick_primary(concept, hits, ontology=ontology)

        row: dict[str, Any] = {
            "concept": concept,
            "primary": _fmt(primary) if primary else None,
            "hits": [_fmt(h) for h in hits[:8]],
        }
        if not primary:
            row["error"] = "no_hits"
            results.append(row)
            continue

        uri = str(primary.get("@id") or "").strip()
        if not uri:
            row["error"] = "missing_primary_id"
            results.append(row)
            continue

        results.append(row)
        label_ids.append(uri)

    payload = {
        "labels": label_ids,
        "none_label": none_label,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"ontology": ontology, "output": str(output_path), "results": results}, indent=2, ensure_ascii=False))
    print(f"\nWrote label set: {output_path} ({len(label_ids)} labels)")

    # Non-zero on missing primary hits/IDs or duplicate label IDs.
    if any("error" in r for r in results):
        return 3
    if len(set(label_ids)) != len(label_ids):
        print("Duplicate label IDs detected in output.", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        raise
