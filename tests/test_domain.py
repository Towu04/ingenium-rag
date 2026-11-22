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
    assert normalize_project("1234.001") == "01234.001"
    assert normalize_project("01234-001") == "01234.001"
    assert normalize_project("12345-002a") == "12345.002A"
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
    assert split_buildings("Blok A - Blok C - West Wing - Blok C") == [
        "Blok A",
        "Blok C",
        "West Wing",
    ]
    assert normalize_building("Blok A - Blok C") == "Blok A - Blok C"
    assert normalize_building("blok c") == "Blok C"
    assert infer_building("Renovatie C-blok en Vleugel Noord") == "Blok C - Vleugel Noord"
    assert infer_building("Campus Oost en Gebouw 2") == "Gebouw 2 - Campus Oost"
    assert (
        infer_building(
            "Renovatie zone-alpha",
            extra_aliases={"zone-alpha": "Zone Alpha"},
        )
        == "Zone Alpha"
    )


def test_document_metadata_strips_sentinels_and_syncs_buildings() -> None:
    meta = DocumentMetadata(
        source_path="General/05_Uitvoering/Spec.pdf",
        project="1234.001",
        building="Blok A - Blok C - West Wing",
        phase="Uitvoering (Execution)",
    )
    assert meta.project == "01234.001"
    assert meta.phase == "Uitvoering"
    assert meta.buildings == ["Blok A", "Blok C", "West Wing"]
    assert meta.building == "Blok A - Blok C - West Wing"

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
            "https://company.sharepoint.com/sites/12345001-INT/Shared%20Documents/a.pdf"
        )
        == "12345.001"
    )
    assert (
        infer_project_id_from_sharepoint_url(
            "https://company.sharepoint.com/sites/12345002A-INT/Shared%20Documents/General/a.pdf"
        )
        == "12345.002A"
    )
    assert (
        infer_project_id_from_sharepoint_url(
            "https://company.sharepoint.com/sites/67890001B-INT/Shared%20Documents/a.pdf"
        )
        == "67890.001B"
    )
    assert infer_project_id_from_path("General/00_Info/17-341-1883 - Data sheets.pdf") is None
    assert infer_project_id_from_path("General/00_Info/1003-004 lijst IGEG's _001.pdf") is None
    assert infer_project_id_from_path("PRJ_104_Campus_Noord/Spec.pdf") == "PRJ_104"

    assert infer_building("BLOK\nPEIL\nPLAN NR") is None
    assert infer_building("BLOK B\nBLOK C") == "Blok B - Blok C"

    assert infer_phase("General/04_Prijsvraag/ALG/Bestek.pdf") == "Aanbesteding"
    assert infer_phase("General/03_Ontwerp 1/Nota.pdf") == "Definitief Ontwerp"
    assert infer_phase("MODULE 101 - DO! - AI2 - AB", allow_abbreviations=False) is None
    assert infer_phase("PRJ_101/02_DO/Schema.pdf", allow_abbreviations=True) == "Definitief Ontwerp"

    assert infer_doc_category("Bestek_HVAC.docx") == DocCategory.SPECIFICATION
    assert infer_doc_category("Documentenlijst_HVAC.xlsx") == DocCategory.DOCUMENT_REGISTER
    assert is_bogus_heading("8000. Stad") is True
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
            source_path="General/05_Uitvoering/Spec.pdf", project="12345.001"
        ),
    )
    restored = ParsedDoc.model_validate_json(doc.model_dump_json())
    assert restored == doc
