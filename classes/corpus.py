"""Ontology corpus ingestion and normalization.

Builds/loads JSONL corpora using BioPortal when online, with a graceful offline
fallback to an existing on-disk corpus.

Output contract (stable fields):
- tco_id: str
- label: str
- synonyms: list[str]
- text: str

For backward compatibility with existing code and artifacts, we also populate:
- document_text: str  (same as `text`)
- definition: str
- parent_labels: list[str]
- ancestor_labels: list[str]
- sibling_labels: list[str]
- child_labels: list[str]

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

from .onto_config import OntologyConfig


BIOPORTAL_BASE_URL = "https://data.bioontology.org"
DEFAULT_TIMEOUT_S = 30


@dataclass(frozen=True)
class OntoDoc:
    tco_id: str
    label: str
    synonyms: tuple[str, ...]
    text: str
    definition: str = ""
    parent_labels: tuple[str, ...] = ()
    ancestor_labels: tuple[str, ...] = ()
    sibling_labels: tuple[str, ...] = ()
    child_labels: tuple[str, ...] = ()

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
            "ancestor_labels": list(self.ancestor_labels),
            "sibling_labels": list(self.sibling_labels),
            "child_labels": list(self.child_labels),
        }


def _normalize_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def _normalize_label_text(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    return " ".join(s.replace("_", " ").split())


def _dedupe_nonempty(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        sv = str(v or "").strip()
        if not sv:
            continue
        key = sv.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sv)
    return out


def _uri_tail(value: str) -> str:
    s = str(value or "").strip()
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _extract_collection(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        collection = payload.get("collection")
        if isinstance(collection, list):
            return [x for x in collection if isinstance(x, dict)]
        return [payload]
    return []


def _api_get(
    session: requests.Session,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> Any:
    url = f"{BIOPORTAL_BASE_URL}{endpoint}"
    resp = session.get(url, params=params or {}, timeout=timeout_s)
    if resp.status_code == 429:
        time.sleep(1.0)
        resp = session.get(url, params=params or {}, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _api_get_auth(
    session: requests.Session,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> Any:
    params = dict(params or {})
    api_key = os.getenv("BIOPORTAL_API_KEY")
    if api_key:
        params["apikey"] = api_key
    return _api_get(session, endpoint, params=params, timeout_s=timeout_s)


def _fetch_related_classes(
    session: requests.Session,
    class_details: dict[str, Any],
    *,
    relation: str,
    limit: int | None = None,
) -> list[dict[str, str]]:
    link = (class_details.get("links") or {}).get(relation)
    if not link or not isinstance(link, str):
        return []

    endpoint = link.replace(BIOPORTAL_BASE_URL, "")
    try:
        data = _api_get_auth(session, endpoint)
    except Exception:
        return []

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in _extract_collection(data):
        item_id = str(item.get("@id") or item.get("id") or "").strip()
        label = _normalize_label_text(item.get("prefLabel") or item.get("label") or _uri_tail(item_id))
        if not label:
            continue
        key = item_id.lower() or label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": item_id, "label": label})
        if limit is not None and len(out) >= limit:
            break
    return out


def _labels_from_related(related: Iterable[dict[str, str]]) -> list[str]:
    return _dedupe_nonempty(_normalize_label_text(item.get("label", "")) for item in related)


def _fetch_parent_labels(session: requests.Session, class_details: dict[str, Any]) -> list[str]:
    return _labels_from_related(_fetch_related_classes(session, class_details, relation="parents"))


def fetch_class_details(session: requests.Session, *, acronym: str, tco_id: str) -> dict[str, Any]:
    encoded = quote(tco_id, safe="")
    return _api_get_auth(session, f"/ontologies/{acronym}/classes/{encoded}")


def _get_cached_details(
    session: requests.Session,
    *,
    acronym: str,
    tco_id: str,
    details_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if tco_id in details_cache:
        return details_cache[tco_id]
    details = fetch_class_details(session, acronym=acronym, tco_id=tco_id)
    details_cache[tco_id] = details
    return details


def _build_reasoning_text(
    *,
    label: str,
    synonyms: Iterable[str],
    definition: str,
    parent_labels: Iterable[str],
    ancestor_labels: Iterable[str],
    sibling_labels: Iterable[str],
    child_labels: Iterable[str],
) -> str:
    pretty_label = _normalize_label_text(label)
    raw_label = str(label or "").strip()
    normalized_synonyms = _dedupe_nonempty(_normalize_label_text(s) for s in synonyms)
    parent_labels = _dedupe_nonempty(_normalize_label_text(p) for p in parent_labels)
    ancestor_labels = _dedupe_nonempty(_normalize_label_text(a) for a in ancestor_labels)
    sibling_labels = _dedupe_nonempty(_normalize_label_text(s) for s in sibling_labels)
    child_labels = _dedupe_nonempty(_normalize_label_text(c) for c in child_labels)
    definition = str(definition or "").strip()

    parts: list[str] = [f"Label: {pretty_label or raw_label}"]
    if raw_label and pretty_label and raw_label != pretty_label:
        parts.append(f"Label (raw): {raw_label}")
    if normalized_synonyms:
        parts.append(f"Synonyms: {', '.join(normalized_synonyms)}")
    if definition:
        parts.append(f"Definition: {definition}")
    if parent_labels:
        parts.append(f"Direct parent classes: {', '.join(parent_labels)}")
    if ancestor_labels:
        parts.append(f"Ancestor classes (broader): {', '.join(ancestor_labels)}")
    if sibling_labels:
        parts.append(f"Sibling classes (same parent): {', '.join(sibling_labels)}")
    if child_labels:
        parts.append(f"Child classes (subtypes): {', '.join(child_labels)}")

    relation_hints: list[str] = []
    if parent_labels:
        relation_hints.append(f"is-a {parent_labels[0]}")
    if sibling_labels:
        relation_hints.append(f"peer classes include {', '.join(sibling_labels[:3])}")
    if child_labels:
        relation_hints.append(f"subtypes include {', '.join(child_labels[:3])}")
    if relation_hints:
        parts.append(f"Reasoning cues: { '; '.join(relation_hints) }.")

    return "\n".join(parts)


def _fetch_hierarchy_labels(
    session: requests.Session,
    *,
    acronym: str,
    tco_id: str,
    class_details: dict[str, Any],
    details_cache: dict[str, dict[str, Any]],
    max_parents: int = 4,
    max_ancestors: int = 8,
    max_siblings: int = 8,
    max_children: int = 8,
) -> tuple[list[str], list[str], list[str], list[str]]:
    parent_related = _fetch_related_classes(session, class_details, relation="parents", limit=max_parents)
    parent_labels = _labels_from_related(parent_related)

    self_label = _normalize_label_text(class_details.get("prefLabel") or class_details.get("label") or "")
    self_id = str(tco_id or "").strip().lower()
    parent_set = {p.lower() for p in parent_labels}

    ancestor_related = _fetch_related_classes(session, class_details, relation="ancestors")
    ancestor_labels = []
    for anc in _labels_from_related(ancestor_related):
        low = anc.lower()
        if low == self_label.lower() or low in parent_set:
            continue
        ancestor_labels.append(anc)
        if len(ancestor_labels) >= max_ancestors:
            break
    ancestor_labels = _dedupe_nonempty(ancestor_labels)

    child_related = _fetch_related_classes(session, class_details, relation="children", limit=max_children)
    child_labels = []
    for child in _labels_from_related(child_related):
        if child.lower() == self_label.lower():
            continue
        child_labels.append(child)
    child_labels = _dedupe_nonempty(child_labels)[:max_children]

    sibling_labels: list[str] = []
    seen_siblings: set[str] = set()
    for parent in parent_related:
        parent_id = str(parent.get("id") or "").strip()
        if not parent_id:
            continue
        try:
            parent_details = _get_cached_details(
                session,
                acronym=acronym,
                tco_id=parent_id,
                details_cache=details_cache,
            )
        except Exception:
            continue

        sibling_related = _fetch_related_classes(session, parent_details, relation="children")
        for sibling in sibling_related:
            sibling_id = str(sibling.get("id") or "").strip().lower()
            sibling_label = _normalize_label_text(sibling.get("label") or "")
            if not sibling_label:
                continue
            if sibling_id and sibling_id == self_id:
                continue
            if not sibling_id and sibling_label.lower() == self_label.lower():
                continue
            key = sibling_id or sibling_label.lower()
            if key in seen_siblings:
                continue
            seen_siblings.add(key)
            sibling_labels.append(sibling_label)
            if len(sibling_labels) >= max_siblings:
                break
        if len(sibling_labels) >= max_siblings:
            break

    if not ancestor_labels and parent_related:
        grandparent_labels: list[str] = []
        for parent in parent_related:
            parent_id = str(parent.get("id") or "").strip()
            if not parent_id:
                continue
            try:
                parent_details = _get_cached_details(
                    session,
                    acronym=acronym,
                    tco_id=parent_id,
                    details_cache=details_cache,
                )
            except Exception:
                continue
            gp_related = _fetch_related_classes(
                session,
                parent_details,
                relation="parents",
                limit=max_ancestors,
            )
            for gp in _labels_from_related(gp_related):
                low = gp.lower()
                if low == self_label.lower() or low in parent_set:
                    continue
                grandparent_labels.append(gp)
                if len(grandparent_labels) >= max_ancestors:
                    break
            if len(grandparent_labels) >= max_ancestors:
                break
        ancestor_labels = _dedupe_nonempty(grandparent_labels)[:max_ancestors]

    return (
        parent_labels[:max_parents],
        ancestor_labels[:max_ancestors],
        sibling_labels[:max_siblings],
        child_labels[:max_children],
    )


def build_doc_from_details(*, tco_id: str, details: dict[str, Any]) -> OntoDoc:
    label = str(details.get("prefLabel") or details.get("label") or "").strip()
    synonyms = _dedupe_nonempty(_normalize_label_text(s) for s in _normalize_list_field(details.get("synonym")))
    definition_list = _normalize_list_field(details.get("definition"))
    definition = " ".join(str(x).strip() for x in definition_list if str(x).strip())

    text = _build_reasoning_text(
        label=label or tco_id,
        synonyms=synonyms,
        definition=definition,
        parent_labels=[],
        ancestor_labels=[],
        sibling_labels=[],
        child_labels=[],
    )
    return OntoDoc(
        tco_id=tco_id,
        label=label or tco_id,
        synonyms=tuple(synonyms),
        text=text,
        definition=definition,
        parent_labels=(),
        ancestor_labels=(),
        sibling_labels=(),
        child_labels=(),
    )


def ensure_corpus(
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
            raise TypeError("ensure_corpus requires either config=... or acronym=...")
        acronym = config.acronym

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if prefer_bioportal:
        session = requests.Session()
        try:
            # quick connectivity check
            _api_get_auth(session, f"/ontologies/{acronym}")

            docs: list[OntoDoc] = []
            details_cache: dict[str, dict[str, Any]] = {}
            for tco_id in label_ids:
                details = _get_cached_details(
                    session,
                    acronym=acronym,
                    tco_id=tco_id,
                    details_cache=details_cache,
                )
                doc = build_doc_from_details(tco_id=tco_id, details=details)
                parent_labels, ancestor_labels, sibling_labels, child_labels = _fetch_hierarchy_labels(
                    session,
                    acronym=acronym,
                    tco_id=tco_id,
                    class_details=details,
                    details_cache=details_cache,
                )
                doc = OntoDoc(
                    tco_id=doc.tco_id,
                    label=doc.label,
                    synonyms=doc.synonyms,
                    text=_build_reasoning_text(
                        label=doc.label,
                        synonyms=doc.synonyms,
                        definition=doc.definition,
                        parent_labels=parent_labels,
                        ancestor_labels=ancestor_labels,
                        sibling_labels=sibling_labels,
                        child_labels=child_labels,
                    ),
                    definition=doc.definition,
                    parent_labels=tuple(parent_labels),
                    ancestor_labels=tuple(ancestor_labels),
                    sibling_labels=tuple(sibling_labels),
                    child_labels=tuple(child_labels),
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
        records = load_corpus(output_path)
        # Persist normalized schema if needed.
        if any(
            (
                "text" not in r
                or not r.get("text")
                or "ancestor_labels" not in r
                or "sibling_labels" not in r
                or "child_labels" not in r
            )
            for r in records
        ):
            _write_jsonl(output_path, records)
        return records

    raise RuntimeError(
        "Unable to build corpus from BioPortal and no cached corpus exists at "
        f"{output_path}. Set BIOPORTAL_API_KEY or add a cached corpus JSONL."
    )


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
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

    rec["synonyms"] = _normalize_list_field(rec.get("synonyms"))
    rec["definition"] = str(rec.get("definition") or "").strip()
    rec["parent_labels"] = _normalize_list_field(rec.get("parent_labels"))
    rec["ancestor_labels"] = _normalize_list_field(rec.get("ancestor_labels"))
    rec["sibling_labels"] = _normalize_list_field(rec.get("sibling_labels"))
    rec["child_labels"] = _normalize_list_field(rec.get("child_labels"))

    text = str(rec.get("text") or rec.get("document_text") or "").strip()
    if not text:
        text = _build_reasoning_text(
            label=str(rec.get("label") or rec.get("tco_id") or ""),
            synonyms=rec["synonyms"],
            definition=rec["definition"],
            parent_labels=rec["parent_labels"],
            ancestor_labels=rec["ancestor_labels"],
            sibling_labels=rec["sibling_labels"],
            child_labels=rec["child_labels"],
        )
    rec["text"] = text
    rec["document_text"] = text
    return rec


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    # Deterministic ordering: write in the provided order.
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n")


# Backwards-compat aliases for any internal scripts that still use old names.
ensure_tco_corpus = ensure_corpus
load_tco_corpus = load_corpus
