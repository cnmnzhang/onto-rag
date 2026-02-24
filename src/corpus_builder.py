#!/usr/bin/env python3
"""src/corpus_builder.py

Parse AI-RHEUM.ttl into a rich JSONL corpus for RAG.
Three chunk types:
  - finding_chunk   : clinical findings with WHAT/WHY/HOW definitions
  - diagnosis_chunk : per-diagnosis summary aggregating associated findings
  - domain_chunk    : body-system domain summaries

Usage:
    from src.corpus_builder import build_corpus
    records = build_corpus("data/AI-RHEUM.ttl")
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

AIR_PREFIX = "http://purl.bioontology.org/ontology/AIR/"

DISEASE_KEYWORDS = {
    "rheumatoid arthritis": "DXRA",
    "systemic lupus erythematosus": "DXSLE",
    "lupus": "DXSLE",
    "ankylosing spondylitis": "DXANK",
    "psoriatic arthritis": "DXPSO",
    "gout": "DXGT",
    "pseudo-gout": "DXCPD",
    "cppd": "DXCPD",
    "calcium pyrophosphate": "DXCPD",
    "scleroderma": "DXPSS",
    "progressive systemic sclerosis": "DXPSS",
    "sjogren": "DXSJ",
    "sjögren": "DXSJ",
    "polymyalgia rheumatica": "DXPMR",
    "polymyositis": "DXPM",
    "dermatomyositis": "DXPM",
    "mixed connective tissue": "DXMCT",
    "giant cell arteritis": "DXGCR",
    "temporal arteritis": "DXGCR",
    "takayasu": "DXTAK",
    "polyarteritis nodosa": "DXPAN",
    "wegener": "DXWGR",
    "granulomatosis": "DXWGR",
    "behcet": "DXBEH",
    "behçet": "DXBEH",
    "henoch-schonlein": "DXHEN",
    "churg-strauss": "DXCHS",
    "vasculitis": "DXVNS",
    "juvenile rheumatoid": "DXJR1",
    "juvenile ra": "DXJR1",
    "adult still": "DXSTI",
    "rheumatic fever": "DXRHF",
    "lyme": "DXLYM",
    "gonococcal": "DXGCA",
    "bacterial arthritis": "DXBAC",
    "septic arthritis": "DXBAC",
    "reactive arthritis": "DXREI",
    "reiter": "DXREI",
    "enteropathic": "DXENT",
    "fibrositis": "DXFIB",
    "fibromyalgia": "DXFIB",
    "degenerative joint": "DXDJD",
    "osteoarthritis": "DXDJD",
    "raynaud": "DXPRA",
    "carpal tunnel": "DXCTS",
    "spinal stenosis": "DXSPI",
    "disc herniation": "DXDSK",
    "rotator cuff tear": "DXROT",
    "rotator cuff tendinitis": "DXRCT",
    "trochanteric bursitis": "DXTBU",
    "avascular necrosis": "DXNEC",
    "tuberculous arthritis": "DXTBA",
    "kawasaki": "DXKAW",
    "relapsing polychondritis": "DXPCH",
    "sarcoidosis": None,
}


def _strip(val: str) -> str:
    val = val.strip()
    val = re.sub(r"@en\s*$", "", val).strip()
    val = re.sub(r"\^\^xsd:string\s*$", "", val).strip()
    return val.strip('"').strip()


def _extract(block: str, predicate: str) -> str | None:
    m = re.search(predicate + r'\s+"""(.+?)"""', block, re.DOTALL)
    if m:
        return _strip(m.group(1))
    m = re.search(predicate + r'\s+"(.+?)"', block, re.DOTALL)
    if m:
        return _strip(m.group(1))
    return None


def _extract_all(block: str, predicate: str) -> list[str]:
    results = []
    for m in re.finditer(predicate + r'\s+"""(.+?)"""', block, re.DOTALL):
        results.append(_strip(m.group(1)))
    for m in re.finditer(predicate + r'\s+"(.+?)"', block, re.DOTALL):
        results.append(_strip(m.group(1)))
    return results


def _parse_structured(definition: str) -> dict[str, str]:
    parts = {"what": "", "why": "", "how": "", "refs": "", "full": definition}
    if not definition:
        return parts
    segments = definition.split("\t")
    current = "preamble"
    buckets: dict[str, list[str]] = defaultdict(list)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        u = seg.upper()
        if u.startswith("WHAT:"):
            current = "what"
            r = seg[5:].strip()
            if r:
                buckets[current].append(r)
        elif u.startswith("WHY:"):
            current = "why"
            r = seg[4:].strip()
            if r:
                buckets[current].append(r)
        elif u.startswith("HOW:"):
            current = "how"
            r = seg[4:].strip()
            if r:
                buckets[current].append(r)
        elif u.startswith("REFS:") or u.startswith("REF:"):
            current = "refs"
            r = re.sub(r"^REFS?:", "", seg, flags=re.IGNORECASE).strip()
            if r:
                buckets[current].append(r)
        else:
            buckets[current].append(seg)
    parts["what"] = " ".join(buckets.get("what", [])).strip()
    parts["why"] = " ".join(buckets.get("why", [])).strip()
    parts["how"] = " ".join(buckets.get("how", [])).strip()
    parts["refs"] = " ".join(buckets.get("refs", [])).strip()
    return parts


def _mine_diseases(text: str) -> list[str]:
    if not text:
        return []
    lower = text.lower()
    found: set[str] = set()
    for keyword, code in DISEASE_KEYWORDS.items():
        if keyword in lower and code:
            found.add(code)
    return sorted(found)


def _classify(notation: str) -> str:
    if notation.startswith("DX"):
        return "diagnosis"
    if notation.startswith("MF"):
        return "domain"
    if notation.startswith("U"):
        return "grouping"
    return "finding"


def _parse_ttl(ttl_text: str) -> list[dict[str, Any]]:
    blocks = re.split(r"\n\s*\n", ttl_text)
    classes = []
    for block in blocks:
        block = block.strip()
        if "a owl:Class" not in block:
            continue
        uri_m = re.match(r"<(.+?)>\s+a\s+owl:Class", block)
        if not uri_m:
            continue
        uri = uri_m.group(1)
        if not uri.startswith(AIR_PREFIX):
            continue
        label = _extract(block, "skos:prefLabel") or ""
        notation = _extract(block, "skos:notation") or ""
        synonyms = _extract_all(block, "skos:altLabel")
        definition = _extract(block, "skos:definition") or ""
        cui_m = re.findall(r'umls:cui\s+"""(C\d+)"""', block)
        umls_cui = cui_m[0] if cui_m else ""
        parent_uris = re.findall(r"rdfs:subClassOf\s+<(.+?)>", block)
        classes.append(
            {
                "uri": uri,
                "label": label,
                "notation": notation,
                "synonyms": synonyms,
                "definition": definition,
                "umls_cui": umls_cui,
                "parent_uris": parent_uris,
            }
        )
    return classes


def build_corpus(ttl_path: str | Path) -> list[dict[str, Any]]:
    """Parse TTL and return list of RAG-ready records."""
    ttl_text = Path(ttl_path).read_text(encoding="utf-8")
    classes = _parse_ttl(ttl_text)

    uri_to_label: dict[str, str] = {c["uri"]: c["label"] for c in classes if c["label"]}

    diagnoses = [c for c in classes if _classify(c["notation"]) == "diagnosis"]
    domains = [c for c in classes if _classify(c["notation"]) == "domain"]
    findings = [
        c for c in classes
        if _classify(c["notation"]) == "finding" and c["parent_uris"]
    ]

    # --- Finding chunks ---
    finding_chunks = []
    for cls in findings:
        label = cls["label"] or cls["notation"]
        definition = cls["definition"]
        parsed = _parse_structured(definition)
        diseases = _mine_diseases(definition)

        parent_labels, body_systems = [], []
        for p_uri in cls["parent_uris"]:
            p_label = uri_to_label.get(p_uri, "")
            if p_label:
                parent_labels.append(p_label)
            short = p_uri.split("/")[-1]
            if short.startswith("MF"):
                body_systems.append(p_label or short)

        parts = [f"Clinical Finding: {label}"]
        if cls["synonyms"]:
            parts.append(f"Also known as: {', '.join(cls['synonyms'])}")
        if cls["notation"]:
            parts.append(f"Code: {cls['notation']} | UMLS CUI: {cls['umls_cui']}")
        if body_systems:
            parts.append(f"Body system: {', '.join(body_systems)}")
        if parsed["what"]:
            parts.append(f"\nWhat it is: {parsed['what']}")
        elif definition and not any(k in definition for k in ["WHAT:", "WHY:", "HOW:"]):
            if len(definition) > 50:
                parts.append(f"\nDescription: {definition}")
        if parsed["why"]:
            parts.append(f"\nDiagnostic relevance: {parsed['why']}")
        if parsed["how"]:
            parts.append(f"\nHow to assess: {parsed['how']}")
        if diseases:
            parts.append(f"\nAssociated diagnoses: {', '.join(diseases)}")
        if parent_labels:
            parts.append(f"\nClassified under: {', '.join(parent_labels)}")

        finding_chunks.append(
            {
                "id": cls["uri"],
                "chunk_type": "finding_chunk",
                "label": label,
                "notation": cls["notation"],
                "umls_cui": cls["umls_cui"],
                "synonyms": cls["synonyms"],
                "body_systems": body_systems,
                "diseases_mentioned": diseases,
                "definition": definition,
                "definition_what": parsed["what"],
                "definition_why": parsed["why"],
                "definition_how": parsed["how"],
                "text": "\n".join(parts),
            }
        )

    # Disease-finding reverse index
    dx_to_findings: dict[str, list[dict]] = defaultdict(list)
    for fc in finding_chunks:
        for dx in fc["diseases_mentioned"]:
            dx_to_findings[dx].append(fc)

    # --- Diagnosis chunks ---
    dx_chunks = []
    for dx_cls in diagnoses:
        label = dx_cls["label"] or dx_cls["notation"]
        notation = dx_cls["notation"]
        associated = dx_to_findings.get(notation, [])

        by_system: dict[str, list[str]] = defaultdict(list)
        for f in associated:
            for sys in (f.get("body_systems") or ["General"]):
                by_system[sys].append(f["label"])

        # Add clinical keywords so embedding matches clinical vignette language
        clinical_keywords = {
            "DXRA": "rheumatoid arthritis symmetric polyarthritis RF anti-CCP morning stiffness erosions MCP PIP wrist synovitis",
            "DXSLE": "lupus malar rash photosensitivity anti-dsDNA ANA complement nephritis cytopenia oral ulcers",
            "DXGT": "gout monosodium urate crystals MTP podagra hyperuricemia negatively birefringent",
            "DXPSO": "psoriatic arthritis psoriasis DIP dactylitis nail pitting sausage digit asymmetric",
            "DXANK": "ankylosing spondylitis HLA-B27 sacroiliitis syndesmophytes inflammatory back pain morning stiffness",
            "DXPMR": "polymyalgia rheumatica proximal stiffness shoulder hip girdle ESR steroid responsive elderly",
            "DXCPD": "pseudogout calcium pyrophosphate CPPD chondrocalcinosis rhomboid crystals positively birefringent",
            "DXPSS": "scleroderma progressive systemic sclerosis Raynaud digital ulcers skin thickening anti-Scl70",
            "DXMCT": "mixed connective tissue disease MCTD RNP speckled ANA Raynaud overlap",
            "DXWGR": "Wegener granulomatosis ANCA c-ANCA upper airway renal vasculitis",
        }

        parts = [
            f"Diagnosis: {label}",
            f"Code: {notation}",
        ]
        if notation in clinical_keywords:
            parts.append(f"Clinical keywords: {clinical_keywords[notation]}")


        dx_chunks.append(
            {
                "id": f"{dx_cls['uri']}__diagnosis_summary",
                "chunk_type": "diagnosis_chunk",
                "label": label,
                "notation": notation,
                "umls_cui": dx_cls["umls_cui"],
                "synonyms": dx_cls["synonyms"],
                "associated_finding_codes": [f["notation"] for f in associated],
                "text": "\n".join(parts),
            }
        )

    # --- Domain chunks ---
    domain_uri_to_children: dict[str, list[dict]] = defaultdict(list)
    domain_uris = {d["uri"] for d in domains}
    for c in classes:
        for p_uri in c["parent_uris"]:
            if p_uri in domain_uris:
                domain_uri_to_children[p_uri].append(c)

    domain_chunks = []
    for dom_cls in domains:
        label = dom_cls["label"] or dom_cls["notation"]
        children = domain_uri_to_children.get(dom_cls["uri"], [])
        child_labels = sorted({c["label"] for c in children if c["label"]})

        parts = [
            f"Clinical Domain: {label}",
            f"Code: {dom_cls['notation']}",
            f"\nThis domain covers {len(child_labels)} clinical findings/concepts.",
        ]
        if child_labels:
            parts.append(f"\nFindings in this domain: {', '.join(child_labels)}")

        domain_chunks.append(
            {
                "id": f"{dom_cls['uri']}__domain_summary",
                "chunk_type": "domain_chunk",
                "label": label,
                "notation": dom_cls["notation"],
                "child_count": len(child_labels),
                "child_labels": child_labels,
                "text": "\n".join(parts),
            }
        )

    return finding_chunks + dx_chunks + domain_chunks


def build_rich_corpus(ttl_path: str | Path) -> list[dict[str, Any]]:
    """Return only records with substantive definition content (>50 chars)."""
    all_records = build_corpus(ttl_path)
    finding_chunks = [r for r in all_records if r.get("chunk_type") == "finding_chunk"]
    dx_chunks = [r for r in all_records if r.get("chunk_type") == "diagnosis_chunk"]
    domain_chunks = [r for r in all_records if r.get("chunk_type") == "domain_chunk"]

    rich_findings = [r for r in finding_chunks if len(r.get("definition", "")) > 50]
    return rich_findings + dx_chunks + domain_chunks


def save_corpus(records: list[dict], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    import sys
    ttl = sys.argv[1] if len(sys.argv) > 1 else "data/AI-RHEUM.ttl"
    records = build_corpus(ttl)
    rich = [r for r in records if r.get("chunk_type") == "finding_chunk" and len(r.get("definition","")) > 50]
    dx = [r for r in records if r.get("chunk_type") == "diagnosis_chunk"]
    dom = [r for r in records if r.get("chunk_type") == "domain_chunk"]
    print(f"Total: {len(records)} | Rich findings: {len(rich)} | Diagnoses: {len(dx)} | Domains: {len(dom)}")
    save_corpus(records, "data/ai_rheum_corpus_full.jsonl")
    save_corpus(rich + dx + dom, "data/ai_rheum_corpus_rich.jsonl")
    print("Saved: data/ai_rheum_corpus_full.jsonl and data/ai_rheum_corpus_rich.jsonl")
