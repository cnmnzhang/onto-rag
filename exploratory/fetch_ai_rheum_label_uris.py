#!/usr/bin/env python3
"""Fetch top BioPortal search-hit URIs for the approved AI-RHEUM diagnosis concepts.

Per project convention:
- Ontology acronym: AI-RHEUM
- We treat the first returned BioPortal search result as the "primary" class.

Usage:
  python scripts/fetch_ai_rheum_label_uris.py

Env:
  BIOPORTAL_API_KEY (recommended)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

BASE = "https://data.bioontology.org"
ONTOLOGY = "AI-RHEUM"

CONCEPTS: list[str] = [
    "Rheumatoid arthritis",
    "Systemic lupus erythematosus",
    "Gout",
    "Psoriatic arthritis",
    "Ankylosing spondylitis",
    "Polymyalgia rheumatica",
]


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
            "Set BIOPORTAL_API_KEY (export it in your shell or add it to .env).",
            response=r,
        )

    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected JSON from BioPortal: {type(data)}")
    return data


def fetch_search_hits(query: str, *, pagesize: int = 5) -> list[dict[str, Any]]:
    data = _bp_get(
        "/search",
        params={
            "q": query,
            "ontologies": ONTOLOGY,
            "pagesize": str(pagesize),
        },
    )
    collection = data.get("collection")
    if not isinstance(collection, list):
        return []
    return [c for c in collection if isinstance(c, dict)]


def pick_primary(hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    # Per instruction: "Use the primary class URI returned by BioPortal search".
    # Interpret "primary" as the first hit returned.
    return hits[0] if hits else None


def main() -> int:
    # Convenience: load BIOPORTAL_API_KEY from a local .env if present.
    # This keeps usage to `python scripts/fetch_ai_rheum_label_uris.py`.
    if load_dotenv is not None:
        load_dotenv()

    results: list[dict[str, Any]] = []

    for concept in CONCEPTS:
        hits = fetch_search_hits(concept, pagesize=5)
        primary = pick_primary(hits)

        if not primary:
            results.append(
                {
                    "concept": concept,
                    "error": "no_hits",
                    "hits": [],
                }
            )
            continue

        def _fmt(hit: dict[str, Any]) -> dict[str, Any]:
            return {
                "prefLabel": hit.get("prefLabel"),
                "@id": hit.get("@id"),
                "matchType": hit.get("matchType"),
            }

        results.append(
            {
                "concept": concept,
                "primary": _fmt(primary),
                "hits": [_fmt(h) for h in hits],
            }
        )

    print(json.dumps({"ontology": ONTOLOGY, "results": results}, indent=2, ensure_ascii=False))

    # Non-zero exit if any concept had no hits or missing URI.
    for row in results:
        if row.get("error"):
            return 2
        primary = row.get("primary") or {}
        if not primary.get("@id"):
            return 3

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        raise
