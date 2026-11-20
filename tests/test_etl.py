"""Unit tests for shared metadata & heading normalization (`processing/normalization.py`)."""

from processing.normalization import (
    is_bogus_heading,
    normalize_building,
    normalize_phase,
    normalize_project,
    split_buildings,
)


def test_normalize_project() -> None:
    assert normalize_project("UNKNOWN_PROJECT") is None
    assert normalize_project("  ") is None
    assert normalize_project("6038.004") == "06038.004"
    assert normalize_project("06038-004") == "06038.004"
    assert normalize_project("PRJ-2021-104") == "PRJ-2021-104"


def test_normalize_phase() -> None:
    assert normalize_phase("UNKNOWN_PHASE") is None
    assert normalize_phase("Uitvoering (Execution)") == "Uitvoering"
    assert normalize_phase("Aanbesteding (Tender / Specifications)") == "Aanbesteding"
    assert normalize_phase("Definitief Ontwerp (Final Design)") == "Definitief Ontwerp"
    assert normalize_phase("Voorontwerp (Preliminary Design)") == "Voorontwerp"
    assert normalize_phase("As-Built") == "As-Built"
    assert normalize_phase("UO") == "Uitvoering"


def test_split_and_normalize_buildings() -> None:
    assert split_buildings("UNKNOWN_BUILDING") == []
    assert split_buildings("Blok Peil") == []
    assert split_buildings("Blok P - Blok L - Spoed - Blok L") == ["Blok P", "Blok L", "Spoed"]
    assert normalize_building("Blok P - Blok L") == "Blok P - Blok L"


def test_is_bogus_heading() -> None:
    assert is_bogus_heading("8000. Brugge") is True
    assert is_bogus_heading("8000 Brugge") is True
    assert is_bogus_heading("3.2 Luchtgroepen") is False
    assert is_bogus_heading("1. Algemene Voorwaarden") is False
