"""Metadata filter extraction & canonicalization from user queries (imports `domain` only)."""

import re

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
        r"\b(?:building|blok)\s*[:=]\s*(?:(Blok\s+[A-Za-z0-9]+|Vleugel\s+[A-Za-z0-9]+|"
        r"Gebouw\s+[A-Za-z0-9]+|Technische\s+Ruimte|Intensieve\s+Zorgen)|([A-Za-z0-9_-]+))",
        re.IGNORECASE,
    ),
    "phase": re.compile(
        r"\b(?:phase|fase)\s*[:=]\s*(Definitief\s+Ontwerp|Preliminary\s+Design|"
        r"Final\s+Design|As[\s-]Built|[A-Za-z0-9_-]+)",
        re.IGNORECASE,
    ),
}


def extract_metadata_filters(query: str) -> dict[str, str]:
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
        proj = infer_project_id_from_path(query)
        if proj:
            filters["project"] = proj

    if "building" not in filters:
        bldg = infer_building(query)
        if bldg:
            parts = split_buildings(bldg)
            if parts:
                filters["building"] = parts[0]

    if "phase" not in filters:
        ph = infer_phase(query, allow_abbreviations=False)
        if ph:
            filters["phase"] = ph

    return filters
