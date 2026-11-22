"""Unit tests for Stage 1 metadata resolution (`etl.catalog` & `etl.metadata`)."""

from pathlib import Path

import openpyxl
import pytest

from domain import DocCategory, ExtractionMethod
from etl import (
    DocumentRegisterIndex,
    MetadataMatcher,
    ProjectCatalog,
    is_document_register_file,
)


def test_flagship_sample_path_metadata_resolution() -> None:
    sample_path = (
        "PRJ_104_Demo_Project/02_Aanbesteding/Blok_B_Operatiekwartier/Bestek_HVAC_Ventilatie.pdf"
    )
    result = MetadataMatcher().match(sample_path)
    assert result.project == "PRJ_104"
    assert result.phase == "Aanbesteding"
    assert result.building == "Blok B"
    assert result.doc_category == DocCategory.SPECIFICATION


def test_unresolved_defaults_are_none_and_sharepoint_url_resolves_project() -> None:
    matcher = MetadataMatcher()
    unresolved = matcher.match("General/00_Info/zaal4.pdf")
    assert (unresolved.project, unresolved.building, unresolved.phase) == (None, None, None)

    sp_url = (
        "https://company.sharepoint.com/sites/12345001-INT/Shared%20Documents/"
        "General/00_Info/17-341-1883%20-%20Data%20sheets.pdf"
    )
    matched = matcher.match("General/00_Info/17-341-1883 - Data sheets.pdf", sharepoint_url=sp_url)
    assert matched.project == "12345.001"
    assert "layer2_5_sharepoint_site_project" in matched.resolution_sources


def test_project_catalog_csv_loading_and_matching(tmp_path: Path) -> None:
    catalog_csv = tmp_path / "projects_catalog.csv"
    catalog_csv.write_text(
        "project_id;project_name;folder_name;sharepoint_url;building\n"
        "PRJ-2021-104;Campus Noord - Renovatie OK;PRJ_104_Campus_Noord;"
        "https://company.sharepoint.com/sites/campus-noord/Shared%20Documents/PRJ_104_Campus_Noord;Blok B\n",
        encoding="utf-8",
    )
    matcher = MetadataMatcher(catalog=ProjectCatalog.from_file(catalog_csv))
    matched = matcher.match(
        "PRJ_104_Campus_Noord/02_Aanbesteding/Blok_B_Operatiekwartier/Bestek_HVAC_Ventilatie.pdf"
    )
    assert matched.project == "PRJ-2021-104"
    assert matched.project_name == "Campus Noord - Renovatie OK"
    assert "layer2_catalog" in matched.resolution_sources


def test_document_register_layer1_overrides_heuristics(tmp_path: Path) -> None:
    reg_file = tmp_path / "00_Documentenlijst_HVAC.csv"
    reg_file.write_text(
        "Documentnummer;Bestandsnaam;Titel;Gebouw;Fase;Revisie;Type\n"
        "ING-104-HVAC-001;Bestek_HVAC_Ventilatie.pdf;Technisch Bestek Luchtbehandeling;"
        "Blok C - Intensieve Zorgen;Definitief Ontwerp;Rev.C;Bestek\n",
        encoding="utf-8",
    )
    assert is_document_register_file(reg_file)
    reg_index = DocumentRegisterIndex()
    assert len(reg_index.load_register_file(reg_file)) == 1

    matched = MetadataMatcher(register_index=reg_index).match(
        "PRJ_104_Campus_Noord/02_Aanbesteding/Blok_B/Bestek_HVAC_Ventilatie.pdf"
    )
    assert matched.building == "Blok C - Intensieve Zorgen"
    assert matched.phase == "Definitief Ontwerp"
    assert matched.revision == "Rev.C"
    assert matched.doc_code == "ING-104-HVAC-001"
    assert matched.doc_category == DocCategory.SPECIFICATION


def test_excel_document_register_with_header_offset(tmp_path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(["INGENIUM NV - DOCUMENTENLIJST"])
    ws.append(["Project:", "PRJ-104 Campus Noord"])
    ws.append([])
    ws.append(["Doc_Nr", "Bestandsnaam", "Omschrijving", "Gebouw", "Fase", "Rev"])
    ws.append(
        [
            "HVAC-M-005",
            "Meetstaat_Luchtgroepen_OK.xlsx",
            "Meetstaat",
            "Blok B - Operatiekwartier",
            "Aanbesteding",
            "A",
        ]
    )
    xlsx_path = tmp_path / "Borderel_HVAC_Lot2.xlsx"
    wb.save(xlsx_path)

    reg_index = DocumentRegisterIndex()
    assert len(reg_index.load_register_file(xlsx_path)) == 1

    _, _, meta = MetadataMatcher(register_index=reg_index).build_document_metadata(
        file_path=xlsx_path,
        relative_file_path="PRJ_104_Campus_Noord/Meetstaat_Luchtgroepen_OK.xlsx",
        extraction_method=ExtractionMethod.OPENPYXL_TABLE,
    )
    assert meta.buildings == ["Blok B", "Operatiekwartier"]
    assert meta.phase == "Aanbesteding"
    assert meta.extra["doc_category"] == DocCategory.SCHEDULE_TABLE.value


@pytest.mark.parametrize(
    "path_str,expected_phase",
    [
        ("PRJ_101/01_VO/Nota.pdf", "Voorontwerp"),
        ("PRJ_101/02_DO/Nota.pdf", "Definitief Ontwerp"),
        ("PRJ_101/03_Ontwerp 1/Nota.pdf", "Definitief Ontwerp"),
        ("PRJ_101/04_Prijsvraag/Bestek.pdf", "Aanbesteding"),
        ("PRJ_101/05_Uitvoering/Verslag.docx", "Uitvoering"),
        ("PRJ_101/07_As-built/Plan.pdf", "As-Built"),
    ],
)
def test_canonical_phase_resolution(path_str: str, expected_phase: str) -> None:
    assert MetadataMatcher().match(path_str).phase == expected_phase
