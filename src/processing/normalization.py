"""Shared metadata & heading normalization used across ETL adapter, retrieval filters, and domain models."""

import re

UNKNOWN_SENTINELS = frozenset({"UNKNOWN_PROJECT", "UNKNOWN_BUILDING", "UNKNOWN_PHASE"})

# Drawing title-block words that can falsely follow "Blok" in legacy extractions
_BOGUS_BUILDING_LABELS = frozenset({
    "blok peil",
    "blok nr",
    "blok no",
    "blok plan",
    "blok schema",
    "blok datum",
})

# Canonical Dutch project phases used as Qdrant payload filter keys
_PHASE_CANONICAL_MAP: dict[str, str] = {
    "vo": "Voorontwerp",
    "voorontwerp": "Voorontwerp",
    "preliminary": "Voorontwerp",
    "preliminary design": "Voorontwerp",
    "do": "Definitief Ontwerp",
    "definitief ontwerp": "Definitief Ontwerp",
    "final design": "Definitief Ontwerp",
    "ao": "Aanbesteding",
    "aanbesteding": "Aanbesteding",
    "bestek": "Aanbesteding",
    "bestekfase": "Aanbesteding",
    "tender": "Aanbesteding",
    "uo": "Uitvoering",
    "uitvoering": "Uitvoering",
    "uitvoeringsdossier": "Uitvoering",
    "execution": "Uitvoering",
    "ab": "As-Built",
    "as-built": "As-Built",
    "as built": "As-Built",
    "oplevering": "As-Built",
    "opleveringsdossier": "As-Built",
}

_DOSSIER_CODE_RE = re.compile(r"^(\d{4,5})[._\-](\d{3})$")
_POSTAL_CODE_HEADING_RE = re.compile(r"^(\d{4})\.?\s+[A-ZÀ-ÖØ-Þ]")


def strip_unknown_sentinel(value: str | None) -> str | None:
    """Return None for blank strings or legacy UNKNOWN_* sentinel placeholders."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in UNKNOWN_SENTINELS:
        return None
    return cleaned


def normalize_project(value: str | None) -> str | None:
    """Normalize project identifiers (e.g. `6038.004` / `06038-004` -> `06038.004`)."""
    cleaned = strip_unknown_sentinel(value)
    if cleaned is None:
        return None
    m = _DOSSIER_CODE_RE.match(cleaned)
    if m:
        return f"{m.group(1).zfill(5)}.{m.group(2)}"
    return cleaned


def normalize_phase(value: str | None) -> str | None:
    """Normalize verbose bilingual or abbreviated phase labels to canonical Dutch filter keys.

    Examples:
        `'Uitvoering (Execution)'` -> `'Uitvoering'`
        `'Aanbesteding (Tender / Specifications)'` -> `'Aanbesteding'`
        `'VO'` -> `'Voorontwerp'`
        `'UNKNOWN_PHASE'` -> `None`
    """
    cleaned = strip_unknown_sentinel(value)
    if cleaned is None:
        return None
    # Strip parenthetical English translations like "Uitvoering (Execution)"
    base = re.sub(r"\s*\([^)]*\)\s*", " ", cleaned).strip()
    key = re.sub(r"[_\s]+", " ", base.lower()).strip()
    if key in _PHASE_CANONICAL_MAP:
        return _PHASE_CANONICAL_MAP[key]
    return base or None


def split_buildings(value: str | None) -> list[str]:
    """Split a building string (e.g. `'Blok P - Blok L - Spoed'`) into clean normalized tags."""
    cleaned = strip_unknown_sentinel(value)
    if cleaned is None:
        return []
    raw_parts = re.split(r"\s+-\s+|[,;/]+", cleaned)
    result: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        item = " ".join(part.split()).strip()
        if not item:
            continue
        lowered = item.lower()
        if lowered in _BOGUS_BUILDING_LABELS or item.upper() in UNKNOWN_SENTINELS:
            continue
        if lowered not in seen:
            seen.add(lowered)
            result.append(item)
    return result


def normalize_building(value: str | None) -> str | None:
    """Normalize a single or composite building string; returns None if empty or bogus."""
    parts = split_buildings(value)
    return " - ".join(parts) if parts else None


def is_bogus_heading(heading: str | None) -> bool:
    """Return True if a heading is a known regex false positive (e.g. Belgian postal code '8000. Brugge')."""
    if not heading:
        return False
    m = _POSTAL_CODE_HEADING_RE.match(heading.strip())
    if m and 1000 <= int(m.group(1)) <= 9999:
        return True
    return False
