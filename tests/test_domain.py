"""Unit tests for canonical `domain.py` contracts, normalization, and multi-building sync."""

from domain import (
    DocCategory,
    DocumentMetadata,
    ParsedDoc,
    ParsedSection,
    compute_doc_id,
    infer_building,
    infer_doc_category,
    infer_phase,
    infer_project_id_from_path,
    infer_project_id_from_sharepoint_url,
    is_bogus_heading,
    normalize_building,
    normalize_phase,
    normalize_project,
    split_buildings,
)


def test_normalize_project_phase_and_buildings() -> None:
    assert normalize_project("UNKNOWN_PROJECT") is None
    assert normalize_project("  ") is None
    assert normalize_project("6038.004") == "06038.004"
    assert normalize_project("06038-004") == "06038.004"
    assert normalize_project("PRJ-2021-104") == "PRJ-2021-104"

    assert normalize_phase("UNKNOWN_PHASE") is None
    assert normalize_phase("Uitvoering (Execution)") == "Uitvoering"
    assert normalize_phase("Aanbesteding (Tender / Specifications)") == "Aanbesteding"
    assert normalize_phase("Definitief Ontwerp (Final Design)") == "Definitief Ontwerp"
    assert normalize_phase("Voorontwerp (Preliminary Design)") == "Voorontwerp"
    assert normalize_phase("As-Built") == "As-Built"
    assert normalize_phase("UO") == "Uitvoering"
    assert normalize_phase("Prijsvraag") == "Aanbesteding"

    assert split_buildings("UNKNOWN_BUILDING") == []
    assert split_buildings("Blok Peil") == []
    assert split_buildings("Blok P - Blok L - Spoed - Blok L") == ["Blok P", "Blok L", "Spoed"]
    assert normalize_building("Blok P - Blok L") == "Blok P - Blok L"
    assert normalize_building("blok l") == "Blok L"
    assert split_buildings("spoed") == ["Spoed"]
    assert infer_building("Renovatie L-blok en MT-blok") == "Blok L - Blok MT"
    assert infer_building("Verbouwing OP-zalen 8 en 9 en operatiezaal 4") == "Operatiekwartier"


def test_document_metadata_strips_sentinels_and_syncs_buildings() -> None:
    meta = DocumentMetadata(
        source_path="General/05_Uitvoering/Spec.pdf",
        project="6038.004",
        building="Blok P - Blok L - Spoed",
        phase="Uitvoering (Execution)",
    )
    assert meta.project == "06038.004"
    assert meta.phase == "Uitvoering"
    assert meta.buildings == ["Blok P", "Blok L", "Spoed"]
    assert meta.building == "Blok P - Blok L - Spoed"

    unknown = DocumentMetadata(
        source_path="General/a.pdf",
        project="UNKNOWN_PROJECT",
        building="UNKNOWN_BUILDING",
        phase="UNKNOWN_PHASE",
    )
    assert unknown.project is None
    assert unknown.building is None
    assert unknown.buildings == []
    assert unknown.phase is None


def test_taxonomy_heuristics_and_false_positive_guards() -> None:
    assert (
        infer_project_id_from_sharepoint_url(
            "https://ingeniumnv.sharepoint.com/sites/06038004-INT/Shared%20Documents/a.pdf"
        )
        == "06038.004"
    )
    assert infer_project_id_from_path("General/00_Info/17-341-1883 - Data sheets.pdf") is None
    assert infer_project_id_from_path("General/00_Info/1003-004 lijst IGEG's _001.pdf") is None
    assert infer_project_id_from_path("PRJ_104_AZ_Sint_Jan/Spec.pdf") == "PRJ_104"

    assert infer_building("BLOK\nPEIL\nPLAN NR") is None
    assert infer_building("BLOK K\nBLOK L") == "Blok K - Blok L"

    assert infer_phase("General/04_Prijsvraag/ALG/Bestek.pdf") == "Aanbesteding"
    assert infer_phase("General/03_Ontwerp 1/Nota.pdf") == "Definitief Ontwerp"
    assert infer_phase("MODULE 101 - DO! - AI2 - AB", allow_abbreviations=False) is None
    assert infer_phase("PRJ_101/02_DO/Schema.pdf", allow_abbreviations=True) == "Definitief Ontwerp"

    assert infer_doc_category("Bestek_HVAC.docx") == DocCategory.SPECIFICATION
    assert infer_doc_category("Documentenlijst_HVAC.xlsx") == DocCategory.DOCUMENT_REGISTER
    assert is_bogus_heading("8000. Brugge") is True
    assert is_bogus_heading("3.2 Luchtgroepen") is False


def test_parsed_doc_json_roundtrip_and_deterministic_id() -> None:
    doc_id = compute_doc_id("General/05_Uitvoering/Spec.pdf")
    assert len(doc_id) == 16
    assert doc_id == compute_doc_id("General/05_Uitvoering/Spec.pdf")

    doc = ParsedDoc(
        document_id=doc_id,
        content_hash="sha256-deadbeef",
        markdown="Text",
        sections=[
            ParsedSection(
                text="Text", page_numbers=[1], section_header="1. Intro", block_type="text"
            )
        ],
        metadata=DocumentMetadata(
            source_path="General/05_Uitvoering/Spec.pdf", project="06038.004"
        ),
    )
    restored = ParsedDoc.model_validate_json(doc.model_dump_json())
    assert restored == doc
