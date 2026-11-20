"""Metadata filter extraction & normalization from user queries."""

import re

from etl.metadata_matcher import (
    infer_building,
    infer_phase,
    infer_project_id_from_path,
)
from processing.normalization import (
    normalize_phase,
    normalize_project,
    split_buildings,
)

_EXPLICIT_FIELD_RE = {
    "project": re.compile(r"\bproject\s*[:=]\s*([A-Za-z0-9._-]+)", re.IGNORECASE),
    "building": re.compile(
        r"\b(?:building|blok)\s*[:=]\s*(?:(Blok\s+[A-Za-z0-9]+|Vleugel\s+[A-Za-z0-9]+|Gebouw\s+[A-Za-z0-9]+|Technische\s+Ruimte|Intensieve\s+Zorgen)|([A-Za-z0-9_-]+))",
        re.IGNORECASE,
    ),
    "phase": re.compile(
        r"\b(?:phase|fase)\s*[:=]\s*(Definitief\s+Ontwerp|Preliminary\s+Design|Final\s+Design|As[\s-]Built|[A-Za-z0-9_-]+)",
        re.IGNORECASE,
    ),
}


def extract_metadata_filters(query: str) -> dict[str, str]:
    """Extract canonical `project`, `building`, and `phase` filters from a natural-language query.

    Supports both explicit `key:value` overrides (`building:Blok L`) and natural Flemish/English
    engineering phrasing (`"luchtgroepen in Blok L tijdens uitvoering voor 06038.004"`).
    """
    filters: dict[str, str] = {}

    # 1. Explicit key:value syntax takes highest precedence
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

    # 2. Natural-language fallback for fields not explicitly specified
    if "project" not in filters:
        proj = infer_project_id_from_path(query)
        if proj:
            norm_proj = normalize_project(proj)
            if norm_proj:
                filters["project"] = norm_proj

    if "building" not in filters:
        bldg = infer_building(query)
        if bldg:
            parts = split_buildings(bldg)
            if parts:
                filters["building"] = parts[0]

    if "phase" not in filters:
        # Disallow 2-letter abbreviations in free-form queries to avoid false matches
        ph = infer_phase(query, allow_abbreviations=False)
        if ph:
            norm_ph = normalize_phase(ph)
            if norm_ph:
                filters["phase"] = norm_ph

    return filters
