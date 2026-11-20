"""Unit tests for Phase 3: Document Extractors (DOCX, XLSX, PDF) and QualityLogger."""

from pathlib import Path

import docx
import openpyxl
import pymupdf as fitz  # PyMuPDF

from etl.extractors import (
    DocxExtractor,
    PdfExtractor,
    XlsxExtractor,
    inspect_pdf_document,
)
from etl.metadata_matcher import MetadataMatcher
from etl.quality import QualityLogger
from etl.schemas import DocCategory, ExtractionMethod


# ===========================================================================
# 1. Word (.docx) Extractor Tests
# ===========================================================================


def test_docx_extractor_preserves_heading_hierarchy_and_tables(tmp_path: Path) -> None:
    """Verify DocxExtractor maintains hierarchical section_heading and renders Markdown tables."""
    docx_path = tmp_path / "Bestek_HVAC_Blok_A.docx"

    doc = docx.Document()
    doc.add_heading("1. Algemene Voorwaarden", level=1)
    doc.add_paragraph(
        "Dit technisch bestek beschrijft de HVAC-installaties voor AZ Sint-Jan Brugge."
    )

    doc.add_heading("3. HVAC Installaties", level=1)
    doc.add_paragraph("Overzicht van de luchtbehandeling en ventilatiegroepen.")

    doc.add_heading("3.2 Luchtgroepen (Ventilation Units)", level=2)
    doc.add_paragraph("De luchtgroepen dienen te voldoen aan onderstaande specificaties:")

    table = doc.add_table(rows=1, cols=4)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Unit ID"
    hdr_cells[1].text = "Building"
    hdr_cells[2].text = "Airflow (m3/h)"
    hdr_cells[3].text = "Heat Recovery"

    row1 = table.add_row().cells
    row1[0].text = "LBK-01"
    row1[1].text = "Blok A"
    row1[2].text = "12500"
    row1[3].text = "Twin-coil"

    row2 = table.add_row().cells
    row2[0].text = "LBK-02"
    row2[1].text = "Blok A"
    row2[2].text = "18000"
    row2[3].text = "Warmtewiel"

    doc.save(str(docx_path))

    extractor = DocxExtractor(metadata_matcher=MetadataMatcher())
    extracted = extractor.extract(
        file_path=docx_path,
        relative_path="PRJ_104_AZ_Sint_Jan/02_Aanbesteding/Blok_A/Bestek_HVAC_Blok_A.docx",
    )

    assert extracted.metadata.project_id == "PRJ_104"
    assert extracted.metadata.building == "Blok A"
    assert extracted.metadata.phase == "Aanbesteding (Tender / Specifications)"
    assert extracted.metadata.doc_category == DocCategory.SPECIFICATION
    assert extracted.metadata.extraction_method == ExtractionMethod.PYTHON_DOCX

    table_blocks = [b for b in extracted.blocks if b.block_type == "table"]
    assert len(table_blocks) == 1
    tbl_block = table_blocks[0]

    assert tbl_block.section_heading == "3. HVAC Installaties > 3.2 Luchtgroepen (Ventilation Units)"
    assert tbl_block.table_headers == ["Unit ID", "Building", "Airflow (m3/h)", "Heat Recovery"]
    assert "| LBK-01 | Blok A | 12500 | Twin-coil |" in tbl_block.markdown_content
    assert "| LBK-02 | Blok A | 18000 | Warmtewiel |" in tbl_block.markdown_content

    # Check top-level and section anchor comments in full_markdown_with_anchors
    assert "<!-- DOC: Bestek_HVAC_Blok_A.docx | PROJECT: PRJ_104" in extracted.full_markdown_with_anchors
    assert (
        "<!-- SECTION: 3. HVAC Installaties > 3.2 Luchtgroepen (Ventilation Units) -->"
        in extracted.full_markdown_with_anchors
    )


# ===========================================================================
# 2. Excel (.xlsx) Extractor Tests
# ===========================================================================


def test_xlsx_extractor_multi_sheet_schedule_and_anchors(tmp_path: Path) -> None:
    """Verify XlsxExtractor parses multi-sheet schedules with sheet_name anchors."""
    xlsx_path = tmp_path / "Toestellenlijst_Ventilatie_Blok_B.xlsx"

    wb = openpyxl.Workbook()
    ws1 = wb.active
    assert ws1 is not None
    ws1.title = "Luchtgroepen"
    ws1.append(["AZ Sint-Jan Brugge - Toestellenlijst Luchtgroepen"])
    ws1.append(["Unit ID", "Locatie", "Debiet (m3/h)", "Filterklasse"])
    ws1.append(["LBK-OK-01", "Blok B - Operatiekwartier", "15000", "ePM1 80% + HEPA H14"])
    ws1.append(["LBK-OK-02", "Blok B - Operatiekwartier", "9500", "ePM1 80% + HEPA H14"])

    ws2 = wb.create_sheet("Roosters")
    ws2.append(["Rooster Code", "Ruimte", "Type", "Debiet (m3/h)"])
    ws2.append(["R-101", "OK Zaal 1", "Laminair Plafond", "4500"])
    ws2.append(["R-102", "OK Zaal 2", "Laminair Plafond", "4500"])

    wb.save(xlsx_path)

    extractor = XlsxExtractor(metadata_matcher=MetadataMatcher())
    extracted = extractor.extract(
        file_path=xlsx_path,
        relative_path="PRJ_104_AZ_Sint_Jan/03_Uitvoering/Blok_B_Operatiekwartier/Toestellenlijst_Ventilatie_Blok_B.xlsx",
    )

    assert extracted.metadata.extraction_method == ExtractionMethod.OPENPYXL_TABLE
    assert extracted.metadata.doc_category == DocCategory.SCHEDULE_TABLE
    assert "Blok B" in extracted.metadata.building

    table_blocks = [b for b in extracted.blocks if b.block_type == "table"]
    assert len(table_blocks) == 2

    assert table_blocks[0].sheet_name == "Luchtgroepen"
    assert table_blocks[0].table_headers == ["Unit ID", "Locatie", "Debiet (m3/h)", "Filterklasse"]
    assert "| LBK-OK-01 | Blok B - Operatiekwartier | 15000 | ePM1 80% + HEPA H14 |" in table_blocks[0].markdown_content

    assert table_blocks[1].sheet_name == "Roosters"
    assert table_blocks[1].table_headers == ["Rooster Code", "Ruimte", "Type", "Debiet (m3/h)"]
    assert "<!-- SHEET: Luchtgroepen" in extracted.full_markdown_with_anchors
    assert "<!-- SHEET: Roosters" in extracted.full_markdown_with_anchors


# ===========================================================================
# 3. PDF (.pdf) Extractor & QualityLogger Tests
# ===========================================================================


def test_pdf_extractor_digital_multipage_with_page_anchors_and_table(tmp_path: Path) -> None:
    """Verify PdfExtractor preserves 1-indexed page numbers, section headings, and tables."""
    pdf_path = tmp_path / "Bestek_HVAC_Blok_A.pdf"

    doc = fitz.open()
    # Page 1: General specification text
    p1 = doc.new_page(width=595, height=842)  # A4
    p1.insert_text(
        (50, 70),
        "1. Algemene Voorwaarden\n"
        "Dit document bevat de technische specificaties voor de luchtbehandelingskasten "
        "in AZ Sint-Jan Brugge Campus Sint-Jan Blok A. Alle installaties moeten voldoen "
        "aan de Belgische normen NBN EN 16798 en ziekenhuisrichtlijnen.",
        fontsize=11,
    )

    # Page 2: Section heading + drawn table
    p2 = doc.new_page(width=595, height=842)  # A4
    p2.insert_text(
        (50, 60),
        "3.2 Luchtgroepen (Ventilation Units)\n"
        "Overzicht van de ontwerpdebieten en warmteterugwinning per luchtgroep in Blok A:",
        fontsize=11,
    )

    # Draw a 3-row x 4-column table grid on Page 2 so PyMuPDF find_tables() detects it
    x_cols = [50, 150, 260, 390, 520]
    y_rows = [120, 145, 170, 195]
    for y in y_rows:
        p2.draw_line((x_cols[0], y), (x_cols[-1], y))
    for x in x_cols:
        p2.draw_line((x, y_rows[0]), (x, y_rows[-1]))

    table_data = [
        ["Unit ID", "Building", "Airflow (m3/h)", "Heat Recovery"],
        ["LBK-01", "Blok A", "12500", "Twin-coil"],
        ["LBK-02", "Blok A", "16000", "Warmtewiel"],
    ]
    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            p2.insert_text((x_cols[c_idx] + 6, y_rows[r_idx] + 16), val, fontsize=10)

    doc.save(str(pdf_path))
    doc.close()

    inspections = inspect_pdf_document(pdf_path)
    assert len(inspections) == 2
    assert not inspections[0].is_scanned_page
    assert not inspections[1].is_scanned_page

    extractor = PdfExtractor(metadata_matcher=MetadataMatcher())
    extracted = extractor.extract(
        file_path=pdf_path,
        relative_path="PRJ-2021-04/02_Aanbesteding/Blok_A/Bestek_HVAC_Blok_A.pdf",
    )

    assert extracted.metadata.is_scanned is False
    assert extracted.metadata.is_hvac_diagram is False
    assert extracted.metadata.extraction_method == ExtractionMethod.PYMUPDF_DIGITAL
    assert extracted.metadata.project_id == "PRJ-2021-04"
    assert extracted.metadata.building == "Blok A"

    page_numbers = {b.page_number for b in extracted.blocks}
    assert 1 in page_numbers
    assert 2 in page_numbers

    assert "<!-- PAGE: 1 | SECTION: 1. Algemene Voorwaarden -->" in extracted.full_markdown_with_anchors
    assert "<!-- PAGE: 2 | SECTION: 3.2 Luchtgroepen (Ventilation Units) -->" in extracted.full_markdown_with_anchors
    assert "LBK-01" in extracted.full_markdown_with_anchors


def test_pdf_extractor_scanned_and_hvac_diagram_quality_logging(tmp_path: Path) -> None:
    """Verify Scanned PDFs and Oversized A1 HVAC Diagrams are detected and logged to limitations_report.jsonl."""
    report_jsonl = tmp_path / "logs" / "limitations_report.jsonl"
    quality_logger = QualityLogger(report_path=report_jsonl)
    extractor = PdfExtractor(metadata_matcher=MetadataMatcher())

    # 1. Create a simulated Scanned PDF (< 50 selectable chars, raster image on A4 page)
    scanned_pdf = tmp_path / "Scan_Archief_1998_Blok_A.pdf"
    doc_scan = fitz.open()
    page_scan = doc_scan.new_page(width=595, height=842)
    # Create a small valid PNG pixmap and insert it covering the page
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
    pix.clear_with(240)
    page_scan.insert_image(fitz.Rect(20, 20, 575, 820), pixmap=pix)
    doc_scan.save(str(scanned_pdf))
    doc_scan.close()

    extracted_scan = extractor.extract(
        file_path=scanned_pdf,
        relative_path="PRJ_104_AZ_Sint_Jan/05_As-Built/Blok_A/Scan_Archief_1998_Blok_A.pdf",
    )
    assert extracted_scan.metadata.is_scanned is True
    assert extracted_scan.metadata.extraction_method == ExtractionMethod.PYMUPDF_OCR

    lim_scan = quality_logger.evaluate_and_log(extracted_scan)
    assert lim_scan is not None

    # 2. Create an Oversized A1 HVAC Diagram PDF (2384 x 1684 pt > 1250 pt) with sparse callout text
    diagram_pdf = tmp_path / "Principeschema_LBK01_OK.pdf"
    doc_diag = fitz.open()
    page_diag = doc_diag.new_page(width=2384, height=1684)  # A1 landscape
    page_diag.insert_text(
        (100, 100),
        "LBK-01 | 12500 m3/h | Blok B - Operatiekwartier | Uitvoering",
        fontsize=12,
    )
    doc_diag.save(str(diagram_pdf))
    doc_diag.close()

    extracted_diag = extractor.extract(
        file_path=diagram_pdf,
        relative_path="PRJ_104_AZ_Sint_Jan/Principeschema_LBK01_OK.pdf",
    )
    assert extracted_diag.metadata.is_hvac_diagram is True
    assert extracted_diag.metadata.extraction_method == ExtractionMethod.PYMUPDF_DIAGRAM
    assert extracted_diag.metadata.doc_category == DocCategory.DRAWING_DIAGRAM
    assert "Blok B" in extracted_diag.metadata.building
    assert extracted_diag.metadata.phase == "Uitvoering (Execution)"
    assert any(b.block_type == "diagram_callout" for b in extracted_diag.blocks)

    lim_diag = quality_logger.evaluate_and_log(extracted_diag)
    assert lim_diag is not None
    assert lim_diag.limitation_type == "complex_hvac_diagram"

    logged_records = quality_logger.read_records()
    assert len(logged_records) == 2


def test_xlsx_extractor_does_not_reparse_document_register(tmp_path: Path) -> None:
    """Verify XlsxExtractor strictly extracts blocks/tables and does not re-populate register_index."""
    reg_xlsx = tmp_path / "00_Documentenlijst_HVAC.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Documentenlijst"
    ws.append(["Doc_Nr", "Bestandsnaam", "Omschrijving", "Gebouw", "Fase"])
    ws.append(["HVAC-01", "Bestek_HVAC.pdf", "Bestek", "Blok A", "Aanbesteding"])
    wb.save(reg_xlsx)

    matcher = MetadataMatcher()
    extractor = XlsxExtractor(metadata_matcher=matcher)
    extracted = extractor.extract(file_path=reg_xlsx, relative_path="00_Documentenlijst_HVAC.xlsx")

    # Pass 1 (DocumentRegisterIndex) is the sole place where register_index is populated;
    # XlsxExtractor must not populate matcher.register_index during Pass 2.
    assert len(matcher.register_index.entries) == 0
    assert extracted.metadata.doc_category == DocCategory.DOCUMENT_REGISTER
    assert len(extracted.blocks) == 1


def test_pdf_extractor_rejects_postal_code_heading_and_flags_needs_ocr(tmp_path: Path) -> None:
    """Belgian postal codes (e.g. '8000 Brugge') must not become section headings."""
    pdf_path = tmp_path / "Brandpreventieverslag.pdf"
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 70),
        "Stadsbestuur Brugge\n8000 Brugge\n"
        "1. Organisatie van de brandpreventie\n"
        "De deuren tussen het compartiment en het trappenhuis hebben EI1 30.\n"
        "zijde 1 / 2",
        fontsize=11,
    )
    doc.save(str(pdf_path))
    doc.close()

    extracted = PdfExtractor(metadata_matcher=MetadataMatcher()).extract(pdf_path)
    headings = [b.section_heading for b in extracted.blocks]
    assert "8000. Brugge" not in headings
    assert any(h == "1. Organisatie van de brandpreventie" for h in headings)
    assert all("zijde 1 / 2" not in b.markdown_content for b in extracted.blocks)

    # Scanned PDF with 0 blocks must set needs_ocr=True and log scanned_pdf_needs_ocr
    scanned_pdf = tmp_path / "Scanned_Empty.pdf"
    sdoc = fitz.open()
    spage = sdoc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
    pix.clear_with(240)
    spage.insert_image(fitz.Rect(20, 20, 575, 820), pixmap=pix)
    sdoc.save(str(scanned_pdf))
    sdoc.close()

    scanned = PdfExtractor(metadata_matcher=MetadataMatcher()).extract(scanned_pdf)
    assert scanned.metadata.is_scanned is True
    assert scanned.metadata.needs_ocr is True

    ql = QualityLogger(report_path=tmp_path / "limitations.jsonl")
    rec = ql.evaluate_and_log(scanned)
    assert rec is not None
    assert rec.limitation_type == "scanned_pdf_needs_ocr"
    assert rec.needs_ocr is True


def test_xlsx_extractor_separates_key_value_preamble_and_skips_template_rows(
    tmp_path: Path,
) -> None:
    """Key-value header rows become a text preamble; real multi-column header becomes table_headers."""
    xlsx_path = tmp_path / "06038.004_E_OVERZICHT_WIJZIGINGSBERICHTEN.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Samenvatting"
    ws.append(["Overzicht wijzigingsberichten"])
    ws.append(["Projectnummer:", "6038004"])
    ws.append(["Projectnaam:", "AZ St. Jan Brugge - Renovatie OP-zalen 8 en 9"])
    ws.append(["Naam wijzigingsbericht", "Beschrijving", "Bedrag", "Advies ING"])
    ws.append(["WB E1", "Branddetectie werf", "855.59", "Goedgekeurd"])
    ws.append(["WB+ perceel volgnummer", "0", "0", ""])
    wb.save(xlsx_path)

    extracted = XlsxExtractor(metadata_matcher=MetadataMatcher()).extract(xlsx_path)
    assert extracted.metadata.project_id == "06038.004"

    preamble_blocks = [b for b in extracted.blocks if b.block_type == "text"]
    table_blocks = [b for b in extracted.blocks if b.block_type == "table"]
    assert len(preamble_blocks) == 1
    assert "Projectnummer:" in preamble_blocks[0].markdown_content
    assert len(table_blocks) == 1
    assert table_blocks[0].table_headers == [
        "Naam wijzigingsbericht",
        "Beschrijving",
        "Bedrag",
        "Advies ING",
    ]
    assert "WB E1" in table_blocks[0].markdown_content
    assert "WB+ perceel volgnummer" not in table_blocks[0].markdown_content

