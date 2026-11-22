"""Metadata filter extraction & canonicalization from user queries (imports `domain` only)."""

import re
from collections.abc import Iterable

from domain import (
    infer_building,
    infer_phase,
    infer_project_id_from_path,
    normalize_phase,
    normalize_project,
    split_buildings,
)

_EXPLICIT_FIELD_RE: dict[str, re.Pattern[str]] = {
    "project": re.compile(r"\bproject\s*[:=]\s*([A-Za-z0-9._-]+)", re.IGNORECASE),
    "building": re.compile(
        r'\b(?:building|blok|gebouw)\s*[:=]\s*(?:"([^"]+)"|'
        r"(Blok\s+[A-Za-z0-9]+|Vleugel\s+[A-Za-z0-9]+|Gebouw\s+[A-Za-z0-9]+|Campus\s+[A-Za-z0-9]+)|"
        r"([A-Za-z0-9_-]+))",
        re.IGNORECASE,
    ),
    "phase": re.compile(
        r"\b(?:phase|fase)\s*[:=]\s*(Definitief\s+Ontwerp|Preliminary\s+Design|"
        r"Final\s+Design|As[\s-]Built|[A-Za-z0-9_-]+)",
        re.IGNORECASE,
    ),
}


def _match_known_building(query: str, known_buildings: Iterable[str] | None) -> str | None:
    if not known_buildings:
        return None
    for bldg in sorted(
        {b.strip() for b in known_buildings if b and b.strip()}, key=len, reverse=True
    ):
        if re.search(rf"(?<!\w){re.escape(bldg)}(?!\w)", query, re.IGNORECASE):
            return bldg
    return None


def extract_metadata_filters(
    query: str,
    *,
    known_buildings: Iterable[str] | None = None,
) -> dict[str, str]:
    """Extract canonical `project`, `building`, and `phase` filters from a query string."""
    filters: dict[str, str] = {}

    for field, pattern in _EXPLICIT_FIELD_RE.items():
        m = pattern.search(query)
        if not m:
            continue
        raw_val = next((g for g in m.groups() if g is not None), "").strip()
        if field == "project":
            norm = normalize_project(raw_val)
        elif field == "phase":
            norm = normalize_phase(raw_val)
        else:
            parts = split_buildings(raw_val)
            norm = parts[0] if parts else None
        if norm:
            filters[field] = norm

    if "project" not in filters:
        if proj := infer_project_id_from_path(query):
            filters["project"] = proj

    if "building" not in filters:
        if bldg := infer_building(query):
            parts = split_buildings(bldg)
            if parts:
                filters["building"] = parts[0]
        elif dynamic_bldg := _match_known_building(query, known_buildings):
            filters["building"] = dynamic_bldg

    if "phase" not in filters:
        if ph := infer_phase(query, allow_abbreviations=False):
            filters["phase"] = ph

    return filters
