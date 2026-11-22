"""Unit tests for Stage 1 extractors (`PdfExtractor`, `DocxExtractor`, `XlsxExtractor`) emitting `ParsedDoc`."""

from pathlib import Path

import docx
import openpyxl
import pymupdf as fitz

from domain import DocCategory, ExtractionMethod, ParsedDoc
from etl import (
    DocxExtractor,
    MetadataMatcher,
    PdfExtractor,
    QualityLogger,
    XlsxExtractor,
    inspect_pdf_document,
    score_callout_readability,
)


def test_docx_extractor_emits_parsed_doc_and_ignores_toc_lines(tmp_path: Path) -> None:
    docx_path = tmp_path / "Bestek_HVAC_Blok_A.docx"
    doc = docx.Document()
    doc.add_paragraph("1.\tPerceel HVAC\t4")
    doc.add_heading("1. Algemene Voorwaarden", level=1)
    doc.add_paragraph("Technische specificaties voor Demo Project.")
    doc.add_heading("3.2 Luchtgroepen", level=2)
    table = doc.add_table(rows=2, cols=3)
    for idx, val in enumerate(["Unit ID", "Building", "Airflow (m3/h)"]):
        table.rows[0].cells[idx].text = val
    for idx, val in enumerate(["LBK-01", "Blok A", "12500"]):
        table.rows[1].cells[idx].text = val
    doc.save(str(docx_path))

    parsed = DocxExtractor().extract(
        docx_path,
        relative_path="PRJ_104_Demo_Project/02_Aanbesteding/Blok_A/Bestek_HVAC_Blok_A.docx",
    )
    assert isinstance(parsed, ParsedDoc)
    assert parsed.metadata.project == "PRJ_104"
    assert parsed.metadata.buildings == ["Blok A"]
    assert parsed.metadata.phase == "Aanbesteding"
    assert parsed.metadata.extra["extraction_method"] == ExtractionMethod.PYTHON_DOCX.value

    headings = [s.section_header for s in parsed.sections if s.section_header]
    assert all("\t4" not in h for h in headings)
    tables = [s for s in parsed.sections if s.block_type == "table"]
    assert len(tables) == 1
    assert "| LBK-01 | Blok A | 12500 |" in tables[0].text


def test_xlsx_extractor_separates_kv_preamble_and_skips_unused_wb_templates(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "12345.001_E_OVERZICHT_WIJZIGINGSBERICHTEN.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Samenvatting"
    ws.append(["Overzicht wijzigingsberichten"])
    ws.append(["Projectnummer:", "12345001"])
    ws.append(["Naam wijzigingsbericht", "Beschrijving", "Bedrag", "Advies ING"])
    ws.append(["WB E1", "Branddetectie werf", "855.59", "Goedgekeurd"])
    ws.append(["0", "0", "0", ""])

    ws_blank = wb.create_sheet("x17")
    ws_blank.append(["Art.", "Omschrijving", "EP", "Som"])
    ws_blank.append(["", "SUBTOTAAL", "", "0"])
    wb.save(xlsx_path)

    parsed = XlsxExtractor().extract(xlsx_path)
    assert parsed.metadata.project == "12345.001"
    sheets = {s.sheet_name for s in parsed.sections}
    assert "Samenvatting" in sheets
    assert "x17" not in sheets

    tables = [s for s in parsed.sections if s.block_type == "table"]
    assert len(tables) == 1
    assert (
        tables[0].text.splitlines()[0]
        == "| Naam wijzigingsbericht | Beschrijving | Bedrag | Advies ING |"
    )
    assert "WB E1" in tables[0].text
    assert "| 0 | 0 | 0 |" not in tables[0].text


def test_pdf_extractor_rejects_postal_code_heading_strips_footers_and_coalesces_prose(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "Brandpreventieverslag.pdf"
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 70),
        "Stadsbestuur Stad\n8000 Stad\n1. Brandpreventie\nEerste paragraaf op pagina een.\nzijde 1 / 2",
        fontsize=11,
    )
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 70), "Vervolg van sectie een op pagina twee.\nzijde 2 / 2", fontsize=11)
    doc.save(str(pdf_path))
    doc.close()

    assert len(inspect_pdf_document(pdf_path)) == 2
    parsed = PdfExtractor().extract(pdf_path)
    headings = [s.section_header for s in parsed.sections]
    assert "8000. Stad" not in headings
    assert all("zijde " not in s.text for s in parsed.sections)

    sec1 = next(s for s in parsed.sections if s.section_header == "1. Brandpreventie")
    assert sec1.page_numbers == [1, 2]
    assert "Vervolg van sectie een" in sec1.text


def test_pdf_scanned_flags_needs_ocr_and_diagram_quarantines_noisy_callouts(tmp_path: Path) -> None:
    # 1. Scanned PDF -> 0 sections, needs_ocr=True, logged to limitations JSONL
    scanned_pdf = tmp_path / "Scan_Archief.pdf"
    sdoc = fitz.open()
    spage = sdoc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
    pix.clear_with(240)
    spage.insert_image(fitz.Rect(20, 20, 575, 820), pixmap=pix)
    sdoc.save(str(scanned_pdf))
    sdoc.close()

    scanned = PdfExtractor().extract(scanned_pdf)
    assert scanned.metadata.extra["is_scanned"] is True
    assert scanned.metadata.extra["needs_ocr"] is True
    assert scanned.sections == []

    ql = QualityLogger(report_path=tmp_path / "limitations.jsonl")
    rec = ql.evaluate_and_log(scanned)
    assert rec is not None
    assert rec.limitation_type == "scanned_pdf_needs_ocr"
    assert rec.needs_ocr is True

    # 2. Oversized A1 HVAC diagram with noisy OCR glyphs vs readable callouts
    noisy_pdf = tmp_path / "Principeschema_Noisy.pdf"
    ndoc = fitz.open()
    npage = ndoc.new_page(width=2384, height=1684)
    npage.insert_text((100, 100), "1\nf\n2\nI\n~n\n{\n}\n%3\nBLOK\nPEIL", fontsize=12)
    ndoc.save(str(noisy_pdf))
    ndoc.close()

    filtered = PdfExtractor(metadata_matcher=MetadataMatcher()).extract(
        noisy_pdf, drop_noisy_callouts=True
    )
    assert filtered.metadata.building is None
    assert filtered.metadata.extra["doc_category"] == DocCategory.DRAWING_DIAGRAM.value
    assert filtered.metadata.extra["dropped_noisy_callouts"] == 1
    assert filtered.sections == []

    unfiltered = PdfExtractor().extract(noisy_pdf, drop_noisy_callouts=False)
    assert len(unfiltered.sections) == 1
    assert score_callout_readability("- LBK-01 Luchtgroep\n- Debiet 12500 m3/h") == 1.0


def test_docx_deduplicates_horizontally_merged_cells_and_infers_cover_building(
    tmp_path: Path,
) -> None:
    docx_path = tmp_path / "Verslag_Algemeen.docx"
    doc = docx.Document()
    doc.add_heading("1. Projectomschrijving", level=1)
    doc.add_paragraph("Project 12345.001 — Renovatie C-blok en Vleugel Noord.")

    table = doc.add_table(rows=4, cols=3)
    # Row 0: full-width merged title banner
    banner_cell = table.rows[0].cells[0].merge(table.rows[0].cells[2])
    banner_cell.text = "Luchtgroepen overzicht"
    # Row 1: column headers
    for idx, val in enumerate(["Unit", "Zone", "Debiet"]):
        table.rows[1].cells[idx].text = val
    # Row 2: normal row
    for idx, val in enumerate(["LBK-01", "Blok C", "12500"]):
        table.rows[2].cells[idx].text = val
    # Row 3: horizontally merged col 0+1
    merged_sub = table.rows[3].cells[0].merge(table.rows[3].cells[1])
    merged_sub.text = "Totaal"
    table.rows[3].cells[2].text = "12500"
    doc.save(str(docx_path))

    parsed = DocxExtractor().extract(docx_path, relative_path="General/05_Uitvoering/Verslag.docx")
    assert parsed.metadata.buildings == ["Blok C", "Vleugel Noord"]
    assert "cover_page_building" in parsed.metadata.extra["resolution_sources"]

    tables = [s for s in parsed.sections if s.block_type == "table"]
    assert len(tables) == 1
    assert "Luchtgroepen overzicht" in (tables[0].section_header or "")
    lines = tables[0].text.splitlines()
    assert lines[0] == "| Unit | Zone | Debiet |"
    assert "| Totaal |  | 12500 |" in lines
    assert all("Totaal | Totaal" not in line for line in lines)


def test_pdf_digital_cover_page_building_fallback(tmp_path: Path) -> None:
    pdf_path = tmp_path / "Technisch_Verslag.pdf"
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 70),
        "Technisch Dossier\n1. Scope\nAanpassing klimatisatie in A-blok en Gebouw 2.",
        fontsize=11,
    )
    doc.save(str(pdf_path))
    doc.close()

    parsed = PdfExtractor().extract(pdf_path, relative_path="General/05_Uitvoering/Verslag.pdf")
    assert parsed.metadata.buildings == ["Blok A", "Gebouw 2"]
    assert "cover_page_building" in parsed.metadata.extra["resolution_sources"]


def test_xlsx_extractor_truncates_large_sheet_early_and_logs_limitation(tmp_path: Path) -> None:
    from etl.quality import QualityLogger

    xlsx_path = tmp_path / "telemetry_readings.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Kwartuurwaarden"
    ws.append(["Timestamp", "ActivePower_kW", "ReactivePower_kvar"])
    for i in range(1, 51):
        ws.append([f"2024-01-01 00:{i:02d}", 100.0 + i, 12.5])
    wb.save(xlsx_path)

    parsed = XlsxExtractor(max_rows_per_sheet=10).extract(
        xlsx_path, relative_path="General/00_Berekeningen/telemetry_readings.xlsx"
    )
    assert parsed.metadata.extra.get("xlsx_truncated_sheets") == ["Kwartuurwaarden"]
    table_sec = next(s for s in parsed.sections if s.block_type == "table")
    data_lines = [ln for ln in table_sec.text.splitlines() if ln.startswith("| 2024-01-01")]
    assert len(data_lines) == 10
    assert "truncated to the first 10 data rows" in table_sec.text

    qlog = QualityLogger(report_path=tmp_path / "limitations.jsonl")
    rec = qlog.evaluate_and_log(parsed)
    assert rec is not None
    assert rec.limitation_type == "xlsx_sheet_truncated"
    assert qlog.read_records()[0].limitation_type == "xlsx_sheet_truncated"
