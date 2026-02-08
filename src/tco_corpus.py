"""TCO corpus ingestion and normalization.

Builds/loads `data/tco_corpus.jsonl` using BioPortal when online, with a
graceful offline fallback to an existing on-disk corpus.

Output contract (stable fields):
- tco_id: str
- label: str
- synonyms: list[str]
- text: str

For backward compatibility with existing code, we also populate:
- document_text: str  (same as `text`)
- definition: str
- parent_labels: list[str]

Artifacts remain in `data/`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests

from onto_config import OntologyConfig


BIOPORTAL_BASE_URL = "https://data.bioontology.org"
DEFAULT_TIMEOUT_S = 30


@dataclass(frozen=True)
class TcoDoc:
    tco_id: str
    label: str
    synonyms: tuple[str, ...]
    text: str
    definition: str = ""
    parent_labels: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "tco_id": self.tco_id,
            "label": self.label,
            "synonyms": list(self.synonyms),
            "text": self.text,
            # Back-compat
            "document_text": self.text,
            "definition": self.definition,
            "parent_labels": list(self.parent_labels),
        }


def _normalize_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def _api_get(session: requests.Session, endpoint: str, *, params: dict[str, Any] | None = None, timeout_s: int = DEFAULT_TIMEOUT_S) -> Any:
    url = f"{BIOPORTAL_BASE_URL}{endpoint}"
    resp = session.get(url, params=params or {}, timeout=timeout_s)
    if resp.status_code == 429:
        time.sleep(1.0)
        resp = session.get(url, params=params or {}, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _api_get_auth(session: requests.Session, endpoint: str, *, params: dict[str, Any] | None = None, timeout_s: int = DEFAULT_TIMEOUT_S) -> Any:
    params = dict(params or {})
    api_key = os.getenv("BIOPORTAL_API_KEY")
    if api_key:
        params["apikey"] = api_key
    return _api_get(session, endpoint, params=params, timeout_s=timeout_s)


def _fetch_parent_labels(session: requests.Session, class_details: dict[str, Any]) -> list[str]:
    parents_link = (class_details.get("links") or {}).get("parents")
    if not parents_link or not isinstance(parents_link, str):
        return []

    # BioPortal sometimes returns full URLs.
    endpoint = parents_link.replace(BIOPORTAL_BASE_URL, "")
    try:
        parents_data = _api_get_auth(session, endpoint)
    except Exception:
        return []

    collection = parents_data if isinstance(parents_data, list) else parents_data.get("collection", [])
    labels: list[str] = []
    for p in collection:
        lbl = p.get("prefLabel") or p.get("label")
        if lbl:
            labels.append(str(lbl))
    return labels


def fetch_tco_class_details(session: requests.Session, *, acronym: str, tco_id: str) -> dict[str, Any]:
    encoded = quote(tco_id, safe="")
    return _api_get_auth(session, f"/ontologies/{acronym}/classes/{encoded}")


def build_tco_doc_from_details(*, tco_id: str, details: dict[str, Any]) -> TcoDoc:
    label = str(details.get("prefLabel") or details.get("label") or "").strip()
    synonyms = _normalize_list_field(details.get("synonym"))
    definition_list = _normalize_list_field(details.get("definition"))
    definition = " ".join(str(x).strip() for x in definition_list if str(x).strip())

    parent_labels: list[str] = []
    # Parent labels require a second request; caller may choose to fill.

    parts: list[str] = [f"Label: {label}" if label else ""]
    if synonyms:
        parts.append(f"Synonyms: {', '.join(synonyms)}")
    if definition:
        parts.append(f"Definition: {definition}")
    # parent labels appended later if available

    text = "\n".join([p for p in parts if p])
    return TcoDoc(
        tco_id=tco_id,
        label=label or tco_id,
        synonyms=tuple(synonyms),
        text=text,
        definition=definition,
        parent_labels=tuple(parent_labels),
    )


def ensure_tco_corpus(
    *,
    config: OntologyConfig | None = None,
    acronym: str | None = None,
    label_ids: Iterable[str],
    output_path: str | Path = "data/tco_corpus.jsonl",
    prefer_bioportal: bool = True,
) -> list[dict[str, Any]]:
    """Build a JSONL corpus for the provided class IDs.

    If BioPortal access fails (offline / rate-limited / missing key), falls back
    to the existing on-disk corpus if present.

    Returns corpus records (list of dicts).
    """

    if not acronym:
        if config is None:
            raise TypeError("ensure_tco_corpus requires either config=... or acronym=...")
        acronym = config.acronym

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if prefer_bioportal:
        session = requests.Session()
        try:
            # quick connectivity check
            _api_get_auth(session, f"/ontologies/{acronym}")

            docs: list[TcoDoc] = []
            for tco_id in label_ids:
                details = fetch_tco_class_details(session, acronym=acronym, tco_id=tco_id)
                doc = build_tco_doc_from_details(tco_id=tco_id, details=details)

                parents = _fetch_parent_labels(session, details)
                if parents:
                    # Rebuild text with parent labels (bounded later at context build time).
                    parts = [f"Label: {doc.label}"]
                    if doc.synonyms:
                        parts.append(f"Synonyms: {', '.join(doc.synonyms)}")
                    if doc.definition:
                        parts.append(f"Definition: {doc.definition}")
                    parts.append(f"Parent classes: {', '.join(parents)}")
                    doc = TcoDoc(
                        tco_id=doc.tco_id,
                        label=doc.label,
                        synonyms=doc.synonyms,
                        text="\n".join(parts),
                        definition=doc.definition,
                        parent_labels=tuple(parents),
                    )

                docs.append(doc)
                time.sleep(0.05)  # courtesy; keeps builds stable-ish

            records = [d.to_record() for d in docs]
            _write_jsonl(output_path, records)
            return records
        except Exception:
            # Fall back to disk corpus if present.
            pass

    if output_path.exists():
        records = load_tco_corpus(output_path)
        # Persist normalized schema (adds stable `text` field) if needed.
        if any("text" not in r or not r.get("text") for r in records):
            _write_jsonl(output_path, records)
        return records

    raise RuntimeError(
        "Unable to build TCO corpus from BioPortal and no cached corpus exists at "
        f"{output_path}. Set BIOPORTAL_API_KEY or add a cached data/tco_corpus.jsonl."
    )


def load_tco_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Load corpus and normalize to include stable fields."""

    path = Path(path)
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        records.append(_normalize_record(rec))
    return records


def _normalize_record(rec: dict[str, Any]) -> dict[str, Any]:
    # Ensure stable fields exist.
    if "tco_id" not in rec and "doc_id" in rec:
        rec["tco_id"] = rec["doc_id"]

    text = rec.get("text") or rec.get("document_text") or ""
    rec["text"] = text
    rec["document_text"] = text

    rec["synonyms"] = rec.get("synonyms") or []
    rec["definition"] = rec.get("definition") or ""
    rec["parent_labels"] = rec.get("parent_labels") or []
    return rec


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    # Deterministic ordering: write in the provided order.
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n")
