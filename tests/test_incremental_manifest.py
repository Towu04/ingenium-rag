"""Unit and End-to-End tests for Phase 4 (Incremental Pipeline & CLI) and Phase 5 (Fixtures E2E)."""

from pathlib import Path

import docx
import openpyxl
import pymupdf as fitz
from typer.testing import CliRunner

from etl.cli import app
from etl.pipeline import ETLPipeline
from etl.schemas import DocCategory, ExtractedDocument

runner = CliRunner()
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def populate_synthetic_fixtures(base_dir: Path) -> tuple[Path, Path]:
    """Generate realistic AZ Sint-Jan Brugge sample files (.docx, .pdf, .xlsx, catalog.csv) in base_dir."""
    base_dir.mkdir(parents=True, exist_ok=True)
    catalog_csv = base_dir / "projects_catalog.csv"
    catalog_csv.write_text(
        "project_id;project_name;folder_name;sharepoint_url\n"
        "PRJ-2021-104;AZ Sint-Jan Brugge - HVAC Renovatie;PRJ_104_AZ_Sint_Jan;"
        "https://ingenium.sharepoint.com/sites/AZSintJan/PRJ_104_AZ_Sint_Jan\n",
        encoding="utf-8",
    )

    raw_root = base_dir / "raw_sharepoint"
    proj_dir = raw_root / "PRJ_104_AZ_Sint_Jan" / "02_Aanbesteding" / "Blok_B_Operatiekwartier"
    proj_dir.mkdir(parents=True, exist_ok=True)

    # 1. Word (.docx) with an HVAC ventilation table
    docx_path = proj_dir / "Bestek_HVAC_Ventilatie.docx"
    doc = docx.Document()
    doc.add_heading("1. Algemene Bepalingen", level=1)
    doc.add_paragraph("Technische voorschriften voor de luchtbehandeling van het operatiekwartier.")
    doc.add_heading("3.2 Luchtgroepen (Ventilation Units)", level=2)
    tbl = doc.add_table(rows=1, cols=4)
    for idx, text in enumerate(["Unit ID", "Building", "Airflow (m3/h)", "Heat Recovery"]):
        tbl.rows[0].cells[idx].text = text
    for row_vals in [
        ["LBK-OK-01", "Blok B - Operatiekwartier", "15000", "Twin-coil"],
        ["LBK-OK-02", "Blok B - Operatiekwartier", "12500", "Twin-coil"],
    ]:
        row_cells = tbl.add_row().cells
        for idx, val in enumerate(row_vals):
            row_cells[idx].text = val
    doc.save(str(docx_path))

    # 2. Multi-page Digital PDF (.pdf) specification
    pdf_path = proj_dir / "Berekeningsnota_Luchtdebieten.pdf"
    pdf_doc = fitz.open()
    p1 = pdf_doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 70),
        "1. Uitgangspunten Hygiënische Ventilatie\n"
        "Voor het Operatiekwartier in Blok B van AZ Sint-Jan Brugge wordt een "
        "luchtverversing van minimaal 20 luchtwisselingen per uur aangehouden conform "
        "de Belgische ziekenhuisnormen en NBN EN ISO 14644.",
        fontsize=11,
    )
    p2 = pdf_doc.new_page(width=595, height=842)
    p2.insert_text(
        (50, 70),
        "2.1 Overdrukregime Operatiezalen\n"
        "Elke operatiezaal wordt op +15 Pa overdruk gehouden ten opzichte van de "
        "omliggende sas- en gangzones om infectierisico's te minimaliseren.",
        fontsize=11,
    )
    pdf_doc.save(str(pdf_path))
    pdf_doc.close()

    # 3. Excel (.xlsx) equipment list
    xlsx_path = proj_dir / "Toestellenlijst_Luchtgroepen.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Luchtgroepen"
    ws.append(["Unit ID", "Zone", "Debiet (m3/h)", "Opvoerhoogte (Pa)"])
    ws.append(["LBK-OK-01", "OK Zaal 1-4", "15000", "650"])
    ws.append(["LBK-OK-02", "OK Zaal 5-8", "12500", "600"])
    wb.save(xlsx_path)

    return raw_root, catalog_csv


def test_persist_fixtures_and_run_e2e_pipeline(tmp_path: Path) -> None:
    """Phase 5 E2E verification: generate synthetic fixtures and verify full pipeline."""
    # Attempt to persist fixtures in tests/fixtures/, falling back to tmp_path if read-only
    try:
        raw_root, catalog_csv = populate_synthetic_fixtures(FIXTURES_DIR)
    except PermissionError:
        raw_root, catalog_csv = populate_synthetic_fixtures(tmp_path / "fixtures")
    assert (
        raw_root
        / "PRJ_104_AZ_Sint_Jan"
        / "02_Aanbesteding"
        / "Blok_B_Operatiekwartier"
        / "Bestek_HVAC_Ventilatie.docx"
    ).exists()

    md_dir = tmp_path / "processed" / "markdown"
    json_dir = tmp_path / "processed" / "json"
    manifest_path = tmp_path / "processed" / "manifest.json"
    limitations_path = tmp_path / "logs" / "limitations_report.jsonl"

    pipeline = ETLPipeline(
        catalog_path=catalog_csv,
        markdown_out_dir=md_dir,
        json_out_dir=json_dir,
        manifest_path=manifest_path,
        limitations_report_path=limitations_path,
    )

    # First run: all 3 files should be processed
    run1 = pipeline.run(raw_root)
    assert run1.discovered_files == 3
    assert run1.processed_files == 3
    assert run1.skipped_unchanged_files == 0
    assert run1.failed_files == 0

    md_files = list(md_dir.glob("*.md"))
    json_files = list(json_dir.glob("*.json"))
    assert len(md_files) == 3
    assert len(json_files) == 3

    # Validate every output JSON file against ExtractedDocument schema
    categories_found = set()
    for jf in json_files:
        raw_json_text = jf.read_text(encoding="utf-8")
        assert "full_markdown_with_anchors" not in raw_json_text
        doc = ExtractedDocument.model_validate_json(raw_json_text)
        assert doc.metadata.project_id == "PRJ-2021-104"
        assert "Blok B" in doc.metadata.building
        assert doc.metadata.phase == "Aanbesteding (Tender / Specifications)"
        categories_found.add(doc.metadata.doc_category)
        assert "<!-- DOC:" in doc.full_markdown_with_anchors

    assert DocCategory.SPECIFICATION in categories_found
    assert DocCategory.CALCULATION in categories_found
    assert DocCategory.SCHEDULE_TABLE in categories_found

    # Second run on unchanged folder: 0 processed, 3 skipped via SHA-256 manifest
    run2 = pipeline.run(raw_root)
    assert run2.discovered_files == 3
    assert run2.processed_files == 0
    assert run2.skipped_unchanged_files == 3


def test_incremental_update_when_new_project_folder_added(tmp_path: Path) -> None:
    """Verify adding a new project folder only processes the newly added file."""
    raw_root, catalog_csv = populate_synthetic_fixtures(tmp_path / "workspace")
    md_dir = tmp_path / "out" / "md"
    json_dir = tmp_path / "out" / "json"
    manifest_path = tmp_path / "out" / "manifest.json"
    limitations_path = tmp_path / "out" / "limitations.jsonl"

    pipeline = ETLPipeline(
        catalog_path=catalog_csv,
        markdown_out_dir=md_dir,
        json_out_dir=json_dir,
        manifest_path=manifest_path,
        limitations_report_path=limitations_path,
    )

    first_run = pipeline.run(raw_root)
    assert first_run.processed_files == 3

    # Add a new project folder with 1 new Word specification
    new_proj_dir = raw_root / "PRJ_209_AZ_Sint_Jan" / "04_Uitvoering" / "Blok_A_Spoed"
    new_proj_dir.mkdir(parents=True, exist_ok=True)
    new_docx = new_proj_dir / "Werfverslag_01_HVAC.docx"
    d = docx.Document()
    d.add_heading("1. Werfvergadering Spoed", level=1)
    d.add_paragraph("Bespreking vordering montage luchtkanalen Blok A Spoed.")
    d.save(str(new_docx))

    second_run = pipeline.run(raw_root)
    assert second_run.discovered_files == 4
    assert second_run.processed_files == 1
    assert second_run.skipped_unchanged_files == 3
    assert second_run.extracted_documents[0].metadata.project_id == "PRJ_209"
    assert "Blok A" in second_run.extracted_documents[0].metadata.building
    assert "Spoed" in second_run.extracted_documents[0].metadata.building
    assert second_run.extracted_documents[0].metadata.phase == "Uitvoering (Execution)"
    assert second_run.extracted_documents[0].metadata.doc_category == DocCategory.REPORT_MEETING


def test_cli_inventory_run_and_report_commands(tmp_path: Path) -> None:
    """Verify the Typer CLI commands (`inventory`, `run`, `report`) execute cleanly."""
    raw_root, catalog_csv = populate_synthetic_fixtures(tmp_path / "cli_test")
    md_dir = tmp_path / "cli_out" / "md"
    json_dir = tmp_path / "cli_out" / "json"
    manifest_path = tmp_path / "cli_out" / "manifest.json"
    limitations_path = tmp_path / "cli_out" / "limitations.jsonl"

    res_inv = runner.invoke(
        app,
        ["inventory", str(raw_root), "--catalog", str(catalog_csv)],
    )
    assert res_inv.exit_code == 0
    assert "PRJ-2021-104" in res_inv.stdout

    res_run = runner.invoke(
        app,
        [
            "run",
            str(raw_root),
            "--catalog",
            str(catalog_csv),
            "--markdown-out",
            str(md_dir),
            "--json-out",
            str(json_dir),
            "--manifest",
            str(manifest_path),
            "--limitations-log",
            str(limitations_path),
        ],
    )
    assert res_run.exit_code == 0
    assert "Newly Processed Files" in res_run.stdout

    res_rep = runner.invoke(
        app,
        [
            "report",
            "--manifest",
            str(manifest_path),
            "--limitations-log",
            str(limitations_path),
        ],
    )
    assert res_rep.exit_code == 0
    assert "Extracted Documents Report" in res_rep.stdout
