from __future__ import annotations

import re
from collections import defaultdict


def _uri_tail(label: str) -> str:
    s = str(label).strip()
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


class LabelNormalizer:
    """Normalize model-emitted labels into allowed URI labels or NONE.

    Resolution order:
    1) exact label match
    2) exact NONE sentinel
    3) alias map (URI tail, compact tail, safe acronym aliases)
    4) fallback to NONE
    """

    def __init__(self, allowed_labels: list[str], none_label: str = "NONE"):
        self.none_label = none_label
        self.allowed_labels = tuple(str(x) for x in allowed_labels)
        self.allowed_set = set(self.allowed_labels)
        self.alias_map = self._build_alias_map(self.allowed_labels)

    def _build_alias_map(self, allowed_labels: tuple[str, ...]) -> dict[str, str]:
        candidates: dict[str, set[str]] = defaultdict(set)

        for uri in allowed_labels:
            tail = _uri_tail(uri)
            compact_tail = re.sub(r"[^a-z0-9]", "", tail.lower())

            variants = {
                uri.lower(),
                tail.lower(),
                compact_tail,
            }

            # Safe acronym extraction from URI tail (e.g., Rheumatoid_Arthritis -> RA).
            words = [w for w in re.split(r"[^A-Za-z0-9]+", tail) if w]
            if len(words) >= 2:
                acronym = "".join(w[0] for w in words).upper()
                if 2 <= len(acronym) <= 8:
                    variants.add(acronym.lower())

            # Preserve short all-caps tails (e.g., AIR/RA).
            if re.fullmatch(r"[A-Z0-9]{2,10}", tail):
                variants.add(tail.lower())

            for v in variants:
                if v:
                    candidates[v].add(uri)

        # Keep only unambiguous aliases.
        alias_map: dict[str, str] = {}
        for alias, targets in candidates.items():
            if len(targets) == 1:
                alias_map[alias] = next(iter(targets))
        return alias_map

    def normalize_label(self, value: object) -> str:
        if not isinstance(value, str):
            return self.none_label

        raw = value.strip().strip('"').strip("'")
        if not raw:
            return self.none_label
        if raw in self.allowed_set:
            return raw
        if raw.upper() == self.none_label:
            return self.none_label

        norm = raw.lower()
        if norm in self.alias_map:
            return self.alias_map[norm]

        tail = _uri_tail(raw).lower()
        if tail in self.alias_map:
            return self.alias_map[tail]

        compact = re.sub(r"[^a-z0-9]", "", norm)
        if compact in self.alias_map:
            return self.alias_map[compact]

        return self.none_label