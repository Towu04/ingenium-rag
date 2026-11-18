"""Unit tests for Phase 1 (Pydantic schemas) and Phase 2 (Catalog & Metadata Matcher)."""

from pathlib import Path

import pytest

from etl.catalog import (
    DocumentRegisterIndex,
    ProjectCatalog,
    is_document_register_file,
)
from etl.metadata_matcher import (
    MetadataMatcher,
    infer_building,
    infer_doc_category,
    infer_phase,
)
from etl.schemas import (
    ContentBlock,
    DocCategory,
    DocumentMetadata,
    ExtractedDocument,
    ExtractionMethod,
)


# ===========================================================================
# Phase 1: Schema Serialization & Contract Validation Tests
# ===========================================================================


def test_extracted_document_schema_roundtrip() -> None:
    """Verify ExtractedDocument serializes to JSON and deserializes without loss."""
    meta = DocumentMetadata(
        doc_id="a1b2c3d4e5f60718",
        file_name="Bestek_HVAC_Blok_A.pdf",
        file_extension=".pdf",
        file_hash_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        sharepoint_url="https://ingenium.sharepoint.com/sites/AZSintJan/PRJ-2021-04",
        relative_folder_path="PRJ-2021-04/02_Aanbesteding/Blok_A",
        project_id="PRJ-2021-04",
        project_name="AZ Sint-Jan Brugge",
        building="Blok A",
        phase="Aanbesteding (Tender / Specifications)",
        doc_category=DocCategory.SPECIFICATION,
        extraction_method=ExtractionMethod.PYMUPDF_DIGITAL,
        is_scanned=False,
        is_hvac_diagram=False,
        extraction_confidence=0.98,
        quality_warnings=[],
    )
    block = ContentBlock(
        block_id="blk-001",
        page_number=14,
        section_heading="3.2 Luchtgroepen (Ventilation Units)",
        sheet_name=None,
        block_type="table",
        markdown_content=(
            "| Unit ID | Building | Airflow (m3/h) | Heat Recovery |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| LBK-01 | Blok A | 12500 | Twin-coil |"
        ),
        table_headers=["Unit ID", "Building", "Airflow (m3/h)", "Heat Recovery"],
    )
    doc = ExtractedDocument(
        metadata=meta,
        blocks=[block],
    )

    json_payload = doc.model_dump_json(indent=2)
    assert "full_markdown_with_anchors" not in json_payload

    restored = ExtractedDocument.model_validate_json(json_payload)

    assert restored.metadata.doc_id == "a1b2c3d4e5f60718"
    assert restored.metadata.project_id == "PRJ-2021-04"
    assert restored.metadata.doc_category == DocCategory.SPECIFICATION
    assert restored.metadata.extraction_method == ExtractionMethod.PYMUPDF_DIGITAL
    assert len(restored.blocks) == 1
    assert restored.blocks[0].page_number == 14
    assert restored.blocks[0].table_headers == [
        "Unit ID",
        "Building",
        "Airflow (m3/h)",
        "Heat Recovery",
    ]
    assert "<!-- DOC: Bestek_HVAC_Blok_A.pdf" in restored.full_markdown_with_anchors
    assert "<!-- PAGE: 14 | SECTION: 3.2 Luchtgroepen (Ventilation Units) -->" in restored.full_markdown_with_anchors


def test_document_metadata_confidence_bounds() -> None:
    """Verify extraction_confidence enforces [0.0, 1.0] bounds."""
    with pytest.raises(Exception):
        DocumentMetadata(
            doc_id="id1",
            file_name="test.pdf",
            file_extension=".pdf",
            file_hash_sha256="abc",
            relative_folder_path="folder",
            extraction_method=ExtractionMethod.PYMUPDF_DIGITAL,
            extraction_confidence=1.5,
        )


# ===========================================================================
# Phase 2: Layer 3 — Flemish/Dutch Engineering Regex Heuristics
# ===========================================================================


def test_flagship_az_sint_jan_sample_path() -> None:
    """Test the flagship AZ Sint-Jan Brugge path specified in plan.md Phase 2."""
    sample_path = (
        "PRJ_104_AZ_Sint_Jan/02_Aanbesteding/Blok_B_Operatiekwartier/Bestek_HVAC_Ventilatie.pdf"
    )
    matcher = MetadataMatcher()
    result = matcher.match(sample_path)

    assert result.project_id == "PRJ_104"
    assert result.project_name == "AZ Sint-Jan Brugge"
    assert result.phase == "Aanbesteding (Tender / Specifications)"
    assert "Blok B" in result.building
    assert "Operatiekwartier" in result.building
    assert result.doc_category == DocCategory.SPECIFICATION


@pytest.mark.parametrize(
    "input_path,expected_phase",
    [
        ("PRJ_101/01_VO/Nota_HVAC.pdf", "Voorontwerp (Preliminary Design)"),
        ("PRJ_101/Voorontwerp/Nota_HVAC.pdf", "Voorontwerp (Preliminary Design)"),
        ("PRJ_101/02_DO/Berekening.pdf", "Definitief Ontwerp (Final Design)"),
        ("PRJ_101/Definitief_Ontwerp/Plan.pdf", "Definitief Ontwerp (Final Design)"),
        ("PRJ_101/03_AO/Meetstaat.xlsx", "Aanbesteding (Tender / Specifications)"),
        ("PRJ_101/Aanbesteding/Bestek.docx", "Aanbesteding (Tender / Specifications)"),
        ("PRJ_101/04_UO/Schema.pdf", "Uitvoering (Execution)"),
        ("PRJ_101/Uitvoering/Verslag.docx", "Uitvoering (Execution)"),
        ("PRJ_101/05_AB/PID_LBK01.pdf", "As-Built"),
        ("PRJ_101/As-Built/Oplevering_HVAC.pdf", "As-Built"),
    ],
)
def test_phase_detection_heuristics(input_path: str, expected_phase: str) -> None:
    """Verify all Dutch/English engineering phase codes map to canonical phase names."""
    assert infer_phase(input_path) == expected_phase


@pytest.mark.parametrize(
    "input_text,expected_substrings",
    [
        ("Blok_A/Ventilatie.pdf", ["Blok A"]),
        ("Vleugel_Zuid/Meetstaat.xlsx", ["Vleugel Zuid"]),
        ("Gebouw_3/Niv_-1/Technische_Ruimte/Schema.pdf", ["Gebouw 3", "Niv -1", "Technische Ruimte"]),
        ("Verdieping_+2/Spoed/Plan.pdf", ["Verdieping +2", "Spoed"]),
        ("Campus_Sint-Jan/Labo/Toestellenlijst.xlsx", ["Campus Sint Jan", "Labo"]),
    ],
)
def test_building_and_zone_detection(input_text: str, expected_substrings: list[str]) -> None:
    """Verify structural buildings, floors, and hospital zones are detected."""
    detected = infer_building(input_text)
    assert detected is not None
    for expected in expected_substrings:
        assert expected in detected


@pytest.mark.parametrize(
    "file_path,expected_category",
    [
        ("Bestek_HVAC_Lot2.docx", DocCategory.SPECIFICATION),
        ("Lastenboek_Sanitair.pdf", DocCategory.SPECIFICATION),
        ("Berekeningsnota_Koellast.pdf", DocCategory.CALCULATION),
        ("Meetstaat_HVAC_Blok_A.xlsx", DocCategory.SCHEDULE_TABLE),
        ("Toestellenlijst_Luchtgroepen.xlsx", DocCategory.SCHEDULE_TABLE),
        ("Overzicht_Ventilatie_Debieten.xlsx", DocCategory.DOCUMENT_REGISTER),
        ("Luchtgroep_LBK01_Gegevens.xlsx", DocCategory.SCHEDULE_TABLE),
        ("Principeschema_Stokplaats.pdf", DocCategory.DRAWING_DIAGRAM),
        ("PID_Stoomketels.pdf", DocCategory.DRAWING_DIAGRAM),
        ("Grondplan_Niv_0.pdf", DocCategory.DRAWING_DIAGRAM),
        ("Werfverslag_014.docx", DocCategory.REPORT_MEETING),
        ("Documentenlijst_AO_HVAC.xlsx", DocCategory.DOCUMENT_REGISTER),
        ("Borderel_Plannen.xlsx", DocCategory.DOCUMENT_REGISTER),
    ],
)
def test_doc_category_heuristics(file_path: str, expected_category: DocCategory) -> None:
    """Verify document category mapping from Flemish engineering filenames."""
    assert infer_doc_category(file_path) == expected_category


# ===========================================================================
# Phase 2: Layer 2 — Master Google Sheet Project Catalog
# ===========================================================================


def test_project_catalog_csv_loading_and_matching(tmp_path: Path) -> None:
    """Test loading a semicolon-delimited Google Sheet CSV export and matching folders."""
    catalog_csv = tmp_path / "projects_catalog.csv"
    catalog_csv.write_text(
        "project_id;project_name;folder_name;sharepoint_url;building\n"
        "PRJ-2021-104;AZ Sint-Jan Brugge - Renovatie OK;PRJ_104_AZ_Sint_Jan;"
        "https://ingenium.sharepoint.com/sites/azsintjan/Shared%20Documents/PRJ_104_AZ_Sint_Jan;Blok B\n"
        "PRJ-2023-208;AZ Sint-Jan Brugge - Spoed & Labo;PRJ_208_Spoed_Labo;"
        "https://ingenium.sharepoint.com/sites/azsintjan/Shared%20Documents/PRJ_208_Spoed_Labo;Blok D\n",
        encoding="utf-8",
    )

    catalog = ProjectCatalog.from_file(catalog_csv)
    assert len(catalog.entries) == 2

    matcher = MetadataMatcher(catalog=catalog)
    matched = matcher.match(
        "PRJ_104_AZ_Sint_Jan/02_Aanbesteding/Blok_B_Operatiekwartier/Bestek_HVAC_Ventilatie.pdf"
    )

    assert matched.project_id == "PRJ-2021-104"
    assert matched.project_name == "AZ Sint-Jan Brugge - Renovatie OK"
    assert (
        matched.sharepoint_url
        == "https://ingenium.sharepoint.com/sites/azsintjan/Shared%20Documents/PRJ_104_AZ_Sint_Jan"
    )
    assert "layer2_catalog" in matched.resolution_sources


# ===========================================================================
# Phase 2: Layer 1 — Excel / CSV Document Register Lookup (Highest Priority)
# ===========================================================================


def test_document_register_layer1_overrides_heuristics(tmp_path: Path) -> None:
    """Verify Layer 1 Document Register lookup overrides folder/filename heuristics."""
    reg_file = tmp_path / "00_Documentenlijst_HVAC.csv"
    reg_file.write_text(
        "Documentnummer;Bestandsnaam;Titel;Gebouw;Fase;Revisie;Type\n"
        "ING-104-HVAC-001;Bestek_HVAC_Ventilatie.pdf;Technisch Bestek Luchtbehandeling;"
        "Blok C - Intensieve Zorgen;Definitief Ontwerp;Rev.C;Bestek\n"
        "ING-104-SCH-012;LBK_Centrale_01.pdf;Principeschema Luchtgroep 1;"
        "Blok A - Technische Ruimte;As-Built;Rev.2;Principeschema\n",
        encoding="utf-8",
    )

    assert is_document_register_file(reg_file)

    reg_index = DocumentRegisterIndex()
    entries = reg_index.load_register_file(reg_file)
    assert len(entries) == 2

    matcher = MetadataMatcher(register_index=reg_index)

    # Path says 02_Aanbesteding and Blok_B, but Layer 1 Register explicitly assigns
    # Blok C - Intensieve Zorgen and Definitief Ontwerp!
    matched = matcher.match(
        "PRJ_104_AZ_Sint_Jan/02_Aanbesteding/Blok_B/Bestek_HVAC_Ventilatie.pdf"
    )

    assert matched.building == "Blok C - Intensieve Zorgen"
    assert matched.phase == "Definitief Ontwerp (Final Design)"
    assert matched.revision == "Rev.C"
    assert matched.doc_code == "ING-104-HVAC-001"
    assert matched.doc_category == DocCategory.SPECIFICATION
    assert "layer1_document_register" in matched.resolution_sources

    # Match by document code embedded in filename
    matched_by_code = matcher.match(
        "PRJ_104_AZ_Sint_Jan/Plannen/ING-104-SCH-012_v2.pdf"
    )
    assert matched_by_code.building == "Blok A - Technische Ruimte"
    assert matched_by_code.phase == "As-Built"
    assert matched_by_code.doc_category == DocCategory.DRAWING_DIAGRAM
    assert matched_by_code.revision == "Rev.2"


def test_excel_document_register_with_header_offset(tmp_path: Path) -> None:
    """Verify Excel (.xlsx) Documentenlijst with title rows above the table header is parsed."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Documentenlijst"
    # Typical Flemish engineering header block in rows 1-3
    ws.append(["INGENIUM NV - DOCUMENTENLIJST"])
    ws.append(["Project:", "PRJ-104 AZ Sint-Jan Brugge"])
    ws.append([])
    # Actual table header at row 4
    ws.append(["Doc_Nr", "Bestandsnaam", "Omschrijving", "Gebouw", "Fase", "Rev"])
    ws.append([
        "HVAC-M-005",
        "Meetstaat_Luchtgroepen_OK.xlsx",
        "Meetstaat Ventilatie Operatiekwartier",
        "Blok B - Operatiekwartier",
        "Aanbesteding",
        "A",
    ])
    xlsx_path = tmp_path / "Borderel_HVAC_Lot2.xlsx"
    wb.save(xlsx_path)

    reg_index = DocumentRegisterIndex()
    entries = reg_index.load_register_file(xlsx_path)
    assert len(entries) == 1
    assert entries[0].doc_code == "HVAC-M-005"

    matcher = MetadataMatcher(register_index=reg_index)
    meta = matcher.build_document_metadata(
        file_path=xlsx_path,
        relative_file_path="PRJ_104_AZ_Sint_Jan/Meetstaat_Luchtgroepen_OK.xlsx",
        extraction_method=ExtractionMethod.OPENPYXL_TABLE,
    )
    assert meta.building == "Blok B - Operatiekwartier"
    assert meta.phase == "Aanbesteding (Tender / Specifications)"
    assert meta.doc_category == DocCategory.SCHEDULE_TABLE
    assert meta.file_extension == ".xlsx"

