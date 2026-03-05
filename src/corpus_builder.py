#!/usr/bin/env python3
"""corpus_builder_v3.py

Parse AI-RHEUM.ttl into a JSONL corpus for RAG, optimised for explainability
and teachability.

CHANGES FROM v2
---------------
1. NO-DEF FINDINGS INCLUDED: The v2 filter `len(definition) <= 50` silently
   dropped 359 findings — including clinically critical nodes like nail pitting,
   DIP swelling, MCP joints, sausage digit, temporal artery pain, and Schober's
   test. v3 includes all findings with a meaningful label (>10 chars), using the
   label + synonyms as the chunk text when no structured definition exists.
   chunk_type is still "finding_chunk"; a new field `has_definition` flags these.

2. EXPANDED _KEYWORDS: Added surface forms for:
   - PMR: temporal artery, cranial arteritis, proximal muscle, proximal stiffness,
     morning stiffness, shoulder girdle, hip girdle, arterial biopsy
   - PSA: nail pitting, nail pit, dip joint, dactylitis, sausage finger,
     sausage toe, juxta-articular, enthesitis, enthes
   - RA:  rheumatoid nodule, mcp joint, metacarpophalangeal, anti-ccp,
     rf positive, erosion (articular), synovitis
   - Gout: tophi, tophus, metatarsophalangeal, mtp joint, urate crystal
   - ANK: sacroiliac, bamboo spine, schober, spondylarthrop

3. altLabel mining: skos:altLabel values are now mined for diagnosis links
   at "medium" confidence (same as WHAT/description).

4. Convenience field `relevant_to_eval` added to finding chunks — True if the
   finding links to at least one of the 6 eval-target diagnoses at high or
   medium confidence. Use this for retrieval pre-filtering to suppress noise
   from the 291 permanently unlinked findings.

5. Build stats extended: reports per-dx finding counts before and after fix.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

AIR_PREFIX = "http://purl.bioontology.org/ontology/AIR/"

DIAGNOSIS_CLINICAL_KEYWORDS: dict[str, str] = {
    "DXRA":  (
        "rheumatoid arthritis symmetric polyarthritis RF anti-CCP morning stiffness "
        "erosions MCP PIP wrist synovitis pannus"
    ),
    "DXSLE": (
        "systemic lupus erythematosus malar rash photosensitivity anti-dsDNA ANA "
        "complement nephritis cytopenia oral ulcers serositis"
    ),
    "DXGT":  (
        "gout monosodium urate crystals MTP podagra hyperuricemia "
        "negatively birefringent acute monoarthritis"
    ),
    "DXPSO": (
        "psoriatic arthritis psoriasis DIP dactylitis nail pitting sausage digit "
        "asymmetric oligoarthritis enthesitis"
    ),
    "DXANK": (
        "ankylosing spondylitis HLA-B27 sacroiliitis syndesmophytes inflammatory "
        "back pain morning stiffness axial spondyloarthropathy"
    ),
    "DXPMR": (
        "polymyalgia rheumatica proximal stiffness shoulder girdle hip girdle "
        "ESR elevated CRP steroid responsive prednisone 15mg elderly bilateral "
        "aching morning stiffness temporal arteritis GCA"
    ),
}

# ---------------------------------------------------------------------------
# Keyword map — order matters: longer / more specific phrases first.
# v3: substantially expanded with PMR, PSA, RA, Gout, ANK surface forms.
# ---------------------------------------------------------------------------
_KEYWORDS: dict[str, str] = {
    # ── RA ──────────────────────────────────────────────────────────────────
    "rheumatoid arthritis":       "DXRA",
    "rheumatoid arteritis":       "DXRA",
    "rheumatoid factor":          "DXRA",
    "rheumatoid nodule":          "DXRA",
    "juvenile rheumatoid":        "DXRA",
    "rheumatoid":                 "DXRA",
    "metacarpophalangeal":        "DXRA",
    "mcp joint":                  "DXRA",
    " mcp ":                      "DXRA",   # space-padded to avoid "mcpherson"
    "anti-ccp":                   "DXRA",
    "rf positive":                "DXRA",
    "articular erosion":          "DXRA",
    "pannus":                     "DXRA",
    "synovitis":                  "DXRA",
    # ── SLE ─────────────────────────────────────────────────────────────────
    "systemic lupus erythematosus": "DXSLE",
    "systemic lupus":             "DXSLE",
    "lupus erythematosus":        "DXSLE",
    "lupus":                      "DXSLE",
    " sle":                       "DXSLE",
    "anti-dna":                   "DXSLE",
    "antinuclear":                "DXSLE",
    "fana":                       "DXSLE",
    # ── Gout ────────────────────────────────────────────────────────────────
    "monosodium urate":           "DXGT",
    "urate crystal":              "DXGT",
    "metatarsophalangeal":        "DXGT",
    "mtp joint":                  "DXGT",
    " mtp ":                      "DXGT",
    "mtp 1":                      "DXGT",
    "mtp1":                       "DXGT",
    "mtp 2":                      "DXGT",
    "gouty tophus":               "DXGT",
    "gouty erosion":              "DXGT",
    "tophus":                     "DXGT",
    "tophi":                      "DXGT",
    "hyperuricemia":              "DXGT",
    "uric acid":                  "DXGT",
    "pseudogout":                 "DXGT",
    "pseudo-gout":                "DXGT",
    "gout":                       "DXGT",
    "gouty":                      "DXGT",
    # ── Psoriatic Arthritis ──────────────────────────────────────────────────
    "psoriatic arthritis":        "DXPSO",
    "psoriatic":                  "DXPSO",
    "psoriasis":                  "DXPSO",
    "nail pitting":               "DXPSO",
    "nail pit":                   "DXPSO",
    "dip joint":                  "DXPSO",
    "dip joints":                 "DXPSO",
    "dactylitis":                 "DXPSO",
    "sausage finger":             "DXPSO",
    "sausage toe":                "DXPSO",
    "juxta-articular swelling":   "DXPSO",
    "enthesitis":                 "DXPSO",
    "enthes":                     "DXPSO",
    # ── Ankylosing Spondylitis ───────────────────────────────────────────────
    "ankylosing spondylitis":     "DXANK",
    "sacroiliac":                 "DXANK",
    "sacroiliitis":               "DXANK",
    "syndesmophyte":              "DXANK",
    "bamboo spine":               "DXANK",
    "schober":                    "DXANK",
    "spondylarthrop":             "DXANK",
    "spondylitis":                "DXANK",
    "hla-b27":                    "DXANK",
    "hla b27":                    "DXANK",
    # ── PMR / GCA ────────────────────────────────────────────────────────────
    "polymyalgia rheumatica":     "DXPMR",
    "polymyalgia":                "DXPMR",
    "giant cell arteritis":       "DXPMR",
    "temporal arteritis":         "DXPMR",
    "cranial arteritis":          "DXPMR",
    "temporal artery":            "DXPMR",
    "arterial biopsy":            "DXPMR",
    "proximal muscle weakness":   "DXPMR",
    "proximal stiffness":         "DXPMR",
    "shoulder girdle":            "DXPMR",
    "hip girdle":                 "DXPMR",
    "morning stiffness":          "DXPMR",   # also RA but PMR is most specific for label hits
}

_BOILERPLATE_THRESHOLD = 4

# Minimum label length to include a no-def finding.
# Keeps single-word or trivially short labels out ("Age", "Sex", "GT", etc.)
_MIN_LABEL_LEN = 12


# ---------------------------------------------------------------------------
# TTL parsing helpers  (unchanged from v2)
# ---------------------------------------------------------------------------

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
    """Split a WHAT/WHY/HOW definition string into its three sections."""
    parts: dict[str, str] = {"what": "", "why": "", "how": ""}
    if not definition:
        return parts

    segments = definition.split("\t")
    if len(segments) == 1:
        segments = re.split(r'\n(?=(?:WHAT|WHY|HOW|REFS?):\s*)', definition)
    if len(segments) == 1:
        segments = re.split(r'(?<!\w)(?=(?:WHAT|WHY|HOW|REFS?):\s)', definition)

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
        else:
            buckets[current].append(seg)

    parts["what"] = " ".join(buckets.get("what", [])).strip()
    parts["why"]  = " ".join(buckets.get("why",  [])).strip()
    parts["how"]  = " ".join(buckets.get("how",  [])).strip()
    return parts


def _classify(notation: str) -> str:
    if notation.startswith("DX"):
        return "diagnosis"
    if notation.startswith("MF") or notation.startswith("U"):
        return "grouping"
    return "finding"


def _mine_text_for_diagnoses(
    text: str,
    source_label: str,
) -> list[dict[str, str]]:
    """Return {code, confidence, source} dicts for any eval-target diagnoses
    found in *text*.

    Confidence by source:
      "why"         → "high"
      "label"       → "high"
      "altlabel"    → "high"   (v3 new)
      "what"        → "medium"
      "description" → "medium"
    """
    if not text:
        return []
    lower = text.lower()
    found: dict[str, str] = {}
    for keyword, code in _KEYWORDS.items():
        if keyword in lower and code not in found:
            found[code] = source_label
    confidence_map = {
        "why":         "high",
        "label":       "high",
        "altlabel":    "high",
        "what":        "medium",
        "description": "medium",
    }
    conf = confidence_map.get(source_label, "medium")
    return [{"code": code, "confidence": conf, "source": source_label}
            for code in sorted(found.keys())]


def _merge_dx_links(
    *link_lists: list[dict[str, str]],
) -> list[dict[str, str]]:
    rank = {"high": 0, "medium": 1, "low": 2}
    best: dict[str, dict[str, str]] = {}
    for lst in link_lists:
        for item in lst:
            code = item["code"]
            if code not in best or rank[item["confidence"]] < rank[best[code]["confidence"]]:
                best[code] = item
    return sorted(best.values(), key=lambda x: x["code"])


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
        label       = _extract(block, "skos:prefLabel") or ""
        notation    = _extract(block, "skos:notation") or ""
        synonyms    = _extract_all(block, "skos:altLabel")
        definition  = _extract(block, "skos:definition") or ""
        cui_m       = re.findall(r'umls:cui\s+"""(C\d+)"""', block)
        umls_cui    = cui_m[0] if cui_m else ""
        parent_uris = re.findall(r"rdfs:subClassOf\s+<(.+?)>", block)
        classes.append({
            "uri":         uri,
            "label":       label,
            "notation":    notation,
            "synonyms":    synonyms,
            "definition":  definition,
            "umls_cui":    umls_cui,
            "parent_uris": parent_uris,
        })
    return classes


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

def build_corpus(ttl_path: str | Path) -> tuple[list[dict[str, Any]], dict]:
    ttl_text = Path(ttl_path).read_text(encoding="utf-8")
    classes  = _parse_ttl(ttl_text)

    uri_to_label: dict[str, str] = {c["uri"]: c["label"] for c in classes if c["label"]}

    raw_findings  = [c for c in classes if _classify(c["notation"]) == "finding"]
    raw_diagnoses = [c for c in classes if _classify(c["notation"]) == "diagnosis"]

    # ------------------------------------------------------------------ #
    # Finding chunks                                                       #
    # ------------------------------------------------------------------ #
    finding_chunks: list[dict[str, Any]] = []
    skipped_trivial = 0
    unlinked: list[str] = []
    boilerplate: list[str] = []

    for cls in raw_findings:
        label = cls["label"] or cls["notation"]

        # v3: only skip truly trivial entries (no label or very short label)
        # Previously: `if len(cls["definition"]) <= 50: continue`
        # That incorrectly dropped 359 findings with empty definitions.
        if len(label) < _MIN_LABEL_LEN:
            skipped_trivial += 1
            continue

        has_definition = bool(cls["definition"])
        parsed = _parse_structured(cls["definition"])

        body_systems = [
            uri_to_label.get(p, p.split("/")[-1])
            for p in cls["parent_uris"]
            if p.split("/")[-1].startswith("MF")
        ]

        # ------ diagnosis linking: multi-source with confidence ------

        # 1. Label (high confidence)
        label_links = _mine_text_for_diagnoses(label, "label")

        # 2. Alt labels (high confidence — v3 new)
        altlabel_links: list[dict] = []
        for syn in cls["synonyms"]:
            altlabel_links += _mine_text_for_diagnoses(syn, "altlabel")

        # 3. WHY section (high confidence)
        why_links = _mine_text_for_diagnoses(parsed["why"], "why")

        # Detect boilerplate WHY
        is_boilerplate_why = len(why_links) >= _BOILERPLATE_THRESHOLD
        if is_boilerplate_why:
            boilerplate.append(cls["notation"])
            why_links = [dict(item, confidence="low") for item in why_links]

        # 4. WHAT / unstructured description (medium confidence)
        description_text = parsed["what"] or (
            cls["definition"] if not any(
                k in cls["definition"] for k in ("WHAT:", "WHY:", "HOW:")
            ) else ""
        )
        desc_links = _mine_text_for_diagnoses(description_text, "description")

        # Merge — best confidence wins per code
        all_links = _merge_dx_links(label_links, altlabel_links, why_links, desc_links)

        dx_codes_high_med = sorted(
            {lnk["code"] for lnk in all_links if lnk["confidence"] != "low"}
        )
        dx_codes_all = sorted({lnk["code"] for lnk in all_links})

        if not dx_codes_all:
            unlinked.append(cls["notation"])

        relevant_to_eval = bool(dx_codes_high_med)

        # ------ build text chunk ------
        lines: list[str] = [f"Clinical Finding: {label}"]
        if cls["synonyms"]:
            lines.append(f"Also known as: {', '.join(cls['synonyms'])}")
        if body_systems:
            lines.append(f"Body system: {', '.join(body_systems)}")
        lines.append(f"Code: {cls['notation']}  |  UMLS CUI: {cls['umls_cui']}")

        if parsed["why"]:
            if is_boilerplate_why:
                lines.append(f"\nDiagnostic context (general): {parsed['why']}")
            else:
                lines.append(f"\nDiagnostic significance: {parsed['why']}")
        if parsed["what"]:
            lines.append(f"\nWhat it is: {parsed['what']}")
        elif description_text and not parsed["why"]:
            lines.append(f"\nDescription: {description_text}")
        if parsed["how"]:
            lines.append(f"\nHow to assess: {parsed['how']}")

        # For no-def findings: add a label-derived description hint so the
        # chunk is not just a bare header — helps embedding quality.
        if not has_definition:
            lines.append(f"\nNote: This finding is identified by its label only; no structured definition is present in the source ontology.")

        finding_chunks.append({
            "id":                  cls["uri"],
            "chunk_type":          "finding_chunk",
            "label":               label,
            "notation":            cls["notation"],
            "umls_cui":            cls["umls_cui"],
            "synonyms":            cls["synonyms"],
            "body_systems":        body_systems,
            "linked_diagnoses":    all_links,
            "linked_dx_codes":     dx_codes_high_med,
            "has_boilerplate_why": is_boilerplate_why,
            "has_definition":      has_definition,       # v3 new
            "relevant_to_eval":    relevant_to_eval,     # v3 new
            "definition_what":     parsed["what"],
            "definition_why":      parsed["why"],
            "definition_how":      parsed["how"],
            "text":                "\n".join(lines),
        })

    # ------------------------------------------------------------------ #
    # Diagnosis chunks                                                     #
    # ------------------------------------------------------------------ #
    dx_to_finding_labels: dict[str, list[str]] = defaultdict(list)
    for fc in finding_chunks:
        for dx in fc["linked_dx_codes"]:
            dx_to_finding_labels[dx].append(fc["label"])

    diagnosis_chunks: list[dict[str, Any]] = []
    for cls in raw_diagnoses:
        label    = cls["label"] or cls["notation"]
        notation = cls["notation"]

        associated_labels = sorted(set(dx_to_finding_labels.get(notation, [])))
        clinical_kw       = DIAGNOSIS_CLINICAL_KEYWORDS.get(notation, "")

        lines = [f"Diagnosis: {label}", f"Code: {notation}"]
        if cls["umls_cui"]:
            lines[1] += f"  |  UMLS CUI: {cls['umls_cui']}"
        if clinical_kw:
            lines.append(f"Clinical keywords: {clinical_kw}")
        if associated_labels:
            lines.append(
                f"\nAssociated clinical findings ({len(associated_labels)}):\n"
                + "\n".join(f"  - {lbl}" for lbl in associated_labels)
            )

        diagnosis_chunks.append({
            "id":                        f"{cls['uri']}__diagnosis",
            "chunk_type":                "diagnosis_chunk",
            "label":                     label,
            "notation":                  notation,
            "umls_cui":                  cls["umls_cui"],
            "synonyms":                  cls["synonyms"],
            "associated_finding_labels": associated_labels,
            "eval_target":               notation in DIAGNOSIS_CLINICAL_KEYWORDS,
            "text":                      "\n".join(lines),
        })

    stats = {
        "skipped_trivial":   skipped_trivial,
        "unlinked_findings": unlinked,
        "boilerplate_why":   boilerplate,
    }
    return finding_chunks + diagnosis_chunks, stats


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def build_rich_corpus(ttl_path: str | Path) -> list[dict[str, Any]]:
    records, _ = build_corpus(ttl_path)
    return records


def save_corpus(records: list[dict], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[corpus] Saved {len(records)} records → {output_path}")


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    ttl = sys.argv[1] if len(sys.argv) > 1 else "data/AI-RHEUM.ttl"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/ai_rheum_corpus_v3.jsonl"

    records, stats = build_corpus(ttl)
    findings  = [r for r in records if r["chunk_type"] == "finding_chunk"]
    diagnoses = [r for r in records if r["chunk_type"] == "diagnosis_chunk"]

    with_def   = [r for r in findings if r["has_definition"]]
    no_def     = [r for r in findings if not r["has_definition"]]
    has_why    = [r for r in findings if r["definition_why"]]
    has_how    = [r for r in findings if r["definition_how"]]
    relevant   = [r for r in findings if r["relevant_to_eval"]]
    boilerplate_n = len(stats["boilerplate_why"])

    print(f"\n[corpus] ── Build summary ──────────────────────────────────────")
    print(f"[corpus] Total records:              {len(records)}")
    print(f"[corpus] Finding chunks:             {len(findings)}")
    print(f"[corpus]   with full definition:     {len(with_def)}")
    print(f"[corpus]   label-only (no def):      {len(no_def)}")
    print(f"[corpus]   with WHY section:         {len(has_why)} ({100*len(has_why)//len(findings)}%)")
    print(f"[corpus]   with HOW section:         {len(has_how)} ({100*len(has_how)//len(findings)}%)")
    print(f"[corpus]   boilerplate WHY (demoted):{boilerplate_n}")
    print(f"[corpus]   relevant to eval targets: {len(relevant)} ({100*len(relevant)//len(findings)}%)")
    print(f"[corpus]   skipped (trivial label):  {stats['skipped_trivial']}")
    print(f"[corpus]   unlinked findings:        {len(stats['unlinked_findings'])}")
    print(f"[corpus] Diagnosis chunks:           {len(diagnoses)}")

    print(f"\n[corpus] ── Eval-target diagnosis summary ──────────────────────")
    for r in sorted(diagnoses, key=lambda x: x["notation"]):
        if not r["eval_target"]:
            continue
        n = len(r.get("associated_finding_labels", []))
        print(f"[corpus]   {r['notation']:8s}  {r['label'][:45]:45s}  findings={n:3d}")

    print(f"\n[corpus] ── Confidence breakdown ───────────────────────────────")
    from collections import Counter
    conf_counter: Counter = Counter()
    for r in findings:
        for lnk in r["linked_diagnoses"]:
            conf_counter[lnk["confidence"]] += 1
    for k, v in conf_counter.most_common():
        print(f"[corpus]   {k:8s}: {v}")

    save_corpus(records, out)