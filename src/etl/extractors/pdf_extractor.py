"""PDF (.pdf) extractor supporting Digital PDFs, Scanned PDFs (OCR), and oversized HVAC Diagrams.

Implements the two-step strategy defined in Section 6.A of `plan.md`:
  1. Inspection Step (`inspect_pdf`): Uses PyMuPDF (`fitz`) to inspect page dimensions,
     selectable text density, raster image coverage, and vector drawing density per page.
  2. Parsing Step (`extract`): Extracts 1-indexed page-level text, hierarchical headings,
     Markdown tables, and HVAC diagram callouts using native PyMuPDF (`pymupdf`).
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import List, Optional, Tuple, Union

import pymupdf as fitz  # PyMuPDF

from etl.config import (
    DEFAULT_BUILDING,
    DEFAULT_PHASE,
    HIGH_VECTOR_DRAWINGS_THRESHOLD,
    MIN_DIGITAL_TEXT_CHARS_PER_PAGE,
    OVERSIZED_PAGE_MIN_DIMENSION_PT,
    SPARSE_DIAGRAM_CHAR_THRESHOLD,
)
from etl.extractors.base import (
    BaseExtractor,
    ENGINEERING_HEADING_RE,
    build_markdown_table,
    clean_cell_text,
)
from etl.metadata_matcher import infer_building, infer_phase
from etl.schemas import (
    ContentBlock,
    DocCategory,
    ExtractedDocument,
    ExtractionMethod,
)

logger = logging.getLogger(__name__)

# Standalone page numbers or running footers like "zijde 1 / 3", "pagina 2 van 8", "web: www.trane.com zijde 1 / 3"
_PAGE_FOOTER_RE = re.compile(
    r"^(?:\d{1,3}|(?:web:\s*\S+\s+)?(?:zijde|pagina|page|blz\.?)\s*\d+\s*(?:/|van|of)\s*\d+)$",
    re.IGNORECASE,
)

# Regex for typical Belgian HVAC diagram callouts (e.g. LBK-01, AHU-02, 12500 m3/h, 12.500 m³/h)
HVAC_CALLOUT_TOKEN_RE = re.compile(
    r"(\b(?:LBK|AHU|KWL|VAV|CAV|WP|CV|KK)[_\-]?\d{1,3}[A-Z]?\b|"
    r"\b\d[\d.,\s]*m[³3]/h\b|"
    r"\b\d+(?:[.,]\d+)?\s*(?:kW|Pa|l/s|°C)\b|"
    r"\bDN\s*\d{2,4}\b)",
    re.IGNORECASE,
)


@dataclass
class PageInspection:
    """Per-page inspection metrics computed via PyMuPDF."""

    page_number: int  # 1-indexed
    width_pt: float
    height_pt: float
    selectable_char_count: int
    image_count: int
    vector_drawing_count: int
    is_oversized: bool
    is_scanned_page: bool
    is_hvac_diagram_page: bool


def inspect_pdf_document(pdf_path: Union[str, Path]) -> List[PageInspection]:
    """Inspect all pages of a PDF using PyMuPDF (`fitz`) to classify digital, scanned, or HVAC drawing pages."""
    doc = fitz.open(str(pdf_path))
    inspections: List[PageInspection] = []
    try:
        for idx, page in enumerate(doc):
            page_num = idx + 1
            rect = page.rect
            width_pt = float(rect.width)
            height_pt = float(rect.height)
            max_dim = max(width_pt, height_pt)

            raw_text = page.get_text("text") or ""
            stripped_text = re.sub(r"\s+", "", raw_text)
            char_count = len(stripped_text)

            images = page.get_images(full=False)
            image_count = len(images)

            try:
                drawings = page.get_drawings()
                vector_count = len(drawings)
            except Exception:
                vector_count = 0

            is_oversized = max_dim > OVERSIZED_PAGE_MIN_DIMENSION_PT
            # Scanned page: < 50 selectable characters and has raster image (or completely blank text)
            is_scanned_page = (
                char_count < MIN_DIGITAL_TEXT_CHARS_PER_PAGE
                and not is_oversized
                and (image_count > 0 or vector_count < 20)
            )

            # HVAC diagram / Technical drawing: oversized (A2/A1/A0) OR very high vector drawing density
            # with sparse callout text
            has_hvac_callouts = len(HVAC_CALLOUT_TOKEN_RE.findall(raw_text)) >= 2
            is_hvac_diagram_page = is_oversized or (
                vector_count >= HIGH_VECTOR_DRAWINGS_THRESHOLD
                and (char_count < 1200 or has_hvac_callouts)
            )

            inspections.append(
                PageInspection(
                    page_number=page_num,
                    width_pt=width_pt,
                    height_pt=height_pt,
                    selectable_char_count=char_count,
                    image_count=image_count,
                    vector_drawing_count=vector_count,
                    is_oversized=is_oversized,
                    is_scanned_page=is_scanned_page,
                    is_hvac_diagram_page=is_hvac_diagram_page,
                )
            )
    finally:
        doc.close()

    return inspections


def _rect_contains_or_overlaps(outer: fitz.Rect, inner: fitz.Rect) -> bool:
    """Return True if inner bbox center lies inside outer bbox."""
    cx = (inner.x0 + inner.x1) / 2.0
    cy = (inner.y0 + inner.y1) / 2.0
    return (outer.x0 - 2 <= cx <= outer.x1 + 2) and (outer.y0 - 2 <= cy <= outer.y1 + 2)


class PdfExtractor(BaseExtractor):
    """Extracts page-anchored Markdown text, tables, and diagram callouts from PDFs."""

    def extract(
        self,
        file_path: Union[str, Path],
        relative_path: Optional[Union[str, Path]] = None,
        sharepoint_url: Optional[str] = None,
    ) -> ExtractedDocument:
        path = Path(file_path)
        rel_path = Path(relative_path) if relative_path is not None else Path(path.name)

        inspections = inspect_pdf_document(path)
        total_pages = len(inspections)
        scanned_pages = sum(1 for p in inspections if p.is_scanned_page)
        diagram_pages = sum(1 for p in inspections if p.is_hvac_diagram_page)

        is_scanned = total_pages > 0 and (scanned_pages / total_pages) >= 0.5
        is_hvac_diagram = total_pages > 0 and (
            (diagram_pages / total_pages) >= 0.5
            or self.metadata_matcher.match(rel_path, sharepoint_url=sharepoint_url).doc_category
            == DocCategory.DRAWING_DIAGRAM
        )

        # Choose extraction method reflecting actual PyMuPDF parser mode
        if is_scanned:
            extraction_method = ExtractionMethod.PYMUPDF_OCR
        elif is_hvac_diagram:
            extraction_method = ExtractionMethod.PYMUPDF_DIAGRAM
        else:
            extraction_method = ExtractionMethod.PYMUPDF_DIGITAL

        blocks, warnings, confidence, title_block_meta, needs_ocr = (
            self._extract_pages_with_pymupdf(
                path, inspections, is_scanned=is_scanned, is_hvac_diagram=is_hvac_diagram
            )
        )
        if is_scanned and not blocks:
            needs_ocr = True

        metadata = self.metadata_matcher.build_document_metadata(
            file_path=path,
            relative_file_path=rel_path,
            extraction_method=extraction_method,
            is_scanned=is_scanned,
            is_hvac_diagram=is_hvac_diagram,
            needs_ocr=needs_ocr,
            extraction_confidence=confidence,
            quality_warnings=warnings,
            sharepoint_url=sharepoint_url,
        )

        # Enrich building / phase from drawing title block if not resolved by path/catalog/register
        if metadata.building == DEFAULT_BUILDING and title_block_meta.get("building"):
            metadata.building = title_block_meta["building"]
        if metadata.phase == DEFAULT_PHASE and title_block_meta.get("phase"):
            metadata.phase = title_block_meta["phase"]
        if is_hvac_diagram and metadata.doc_category == DocCategory.UNKNOWN:
            metadata.doc_category = DocCategory.DRAWING_DIAGRAM

        return ExtractedDocument(
            metadata=metadata,
            blocks=blocks,
        )

    def _extract_pages_with_pymupdf(
        self,
        pdf_path: Path,
        inspections: List[PageInspection],
        is_scanned: bool,
        is_hvac_diagram: bool,
    ) -> Tuple[List[ContentBlock], List[str], float, dict, bool]:
        doc = fitz.open(str(pdf_path))
        blocks: List[ContentBlock] = []
        warnings: List[str] = []
        block_counter = 0
        active_heading: Optional[str] = None
        title_block_meta: dict = {}

        low_text_pages = 0
        needs_ocr = False

        try:
            for idx, page in enumerate(doc):
                page_num = idx + 1
                inspection = inspections[idx]

                # 1. Detect tables on this page via PyMuPDF find_tables()
                table_rects: List[fitz.Rect] = []
                page_table_items: List[Tuple[float, str, List[str]]] = []
                try:
                    found_tables = page.find_tables()
                    for tbl in found_tables.tables:
                        raw_grid: List[List[str]] = []
                        if hasattr(tbl, "rows") and tbl.rows:
                            for row in tbl.rows:
                                row_cells: List[str] = []
                                for cell in row.cells:
                                    if cell is None:
                                        row_cells.append("")
                                    else:
                                        cell_txt = page.get_text("text", clip=fitz.Rect(cell)) or ""
                                        row_cells.append(clean_cell_text(cell_txt))
                                raw_grid.append(row_cells)
                        else:
                            extracted_grid = tbl.extract() or []
                            raw_grid = [
                                [clean_cell_text(c) for c in row] for row in extracted_grid
                            ]

                        cleaned_grid = [r for r in raw_grid if any(r)]
                        if len(cleaned_grid) < 2:
                            continue
                        # Verify at least 2 non-empty columns in header
                        if sum(1 for c in cleaned_grid[0] if c) < 2:
                            continue

                        md_table, clean_headers = build_markdown_table(
                            cleaned_grid[0], cleaned_grid[1:]
                        )
                        if md_table:
                            t_rect = fitz.Rect(tbl.bbox)
                            table_rects.append(t_rect)
                            page_table_items.append((t_rect.y0, md_table, clean_headers))
                except Exception:
                    pass

                # 2. Extract text blocks outside detected table bounding boxes
                raw_blocks = page.get_text("blocks") or []
                # Sort top-to-bottom, left-to-right
                raw_blocks.sort(key=lambda b: (round(float(b[1]), 1), round(float(b[0]), 1)))

                text_items: List[Tuple[float, str]] = []
                for b in raw_blocks:
                    # b = (x0, y0, x1, y1, text, block_no, block_type)
                    if len(b) >= 7 and b[6] != 0:
                        continue
                    b_rect = fitz.Rect(b[0], b[1], b[2], b[3])
                    if any(_rect_contains_or_overlaps(t_rect, b_rect) for t_rect in table_rects):
                        continue
                    txt = str(b[4]).strip()
                    if txt:
                        text_items.append((float(b[1]), txt))

                # 3. If scanned page with no selectable text, attempt local OCR fallback
                if inspection.is_scanned_page and not text_items and not page_table_items:
                    ocr_text = self._try_local_page_ocr(page, pdf_path.name, page_num)
                    if ocr_text:
                        text_items.append((0.0, ocr_text))
                    else:
                        low_text_pages += 1
                        needs_ocr = True
                        warnings.append(
                            f"Page {page_num}: Scanned raster page with < {MIN_DIGITAL_TEXT_CHARS_PER_PAGE} "
                            "selectable chars and no local Tesseract OCR output."
                        )

                # Inspect text for title block metadata on HVAC diagrams
                full_page_text = "\n".join(t for _, t in text_items)
                if is_hvac_diagram or inspection.is_hvac_diagram_page:
                    if "building" not in title_block_meta:
                        bldg = infer_building(full_page_text)
                        if bldg:
                            title_block_meta["building"] = bldg
                    if "phase" not in title_block_meta:
                        # Disallow 2-letter abbreviations (`DO`/`AO`/`AB`) in drawing text to prevent
                        # BMS Digital/Analog Output labels from masquerading as project phases.
                        ph = infer_phase(full_page_text, allow_abbreviations=False)
                        if ph:
                            title_block_meta["phase"] = ph

                # 4. Merge text and table items in vertical reading order on this page
                combined_events: List[Tuple[float, str, object]] = []
                for y0, txt in text_items:
                    combined_events.append((y0, "text", txt))
                for y0, md_tbl, hdrs in page_table_items:
                    combined_events.append((y0, "table", (md_tbl, hdrs)))
                combined_events.sort(key=lambda item: item[0])

                if is_hvac_diagram or inspection.is_hvac_diagram_page:
                    # Emit HVAC diagram callout blocks and legend tables
                    callout_lines: List[str] = []
                    for _, kind, payload in combined_events:
                        if kind == "text":
                            for line in str(payload).splitlines():
                                clean_line = line.strip()
                                if clean_line:
                                    callout_lines.append(clean_line)
                        elif kind == "table":
                            md_tbl, hdrs = payload  # type: ignore[misc]
                            block_counter += 1
                            blocks.append(
                                ContentBlock(
                                    block_id=f"pdf-blk-{block_counter:04d}",
                                    page_number=page_num,
                                    section_heading=active_heading or "HVAC Diagram Legend / Schedule",
                                    sheet_name=None,
                                    block_type="table",
                                    markdown_content=md_tbl,
                                    table_headers=hdrs,
                                )
                            )

                    if callout_lines:
                        block_counter += 1
                        blocks.append(
                            ContentBlock(
                                block_id=f"pdf-blk-{block_counter:04d}",
                                page_number=page_num,
                                section_heading=active_heading or "HVAC Diagram Callouts",
                                sheet_name=None,
                                block_type="diagram_callout",
                                markdown_content="\n".join(f"- {line}" for line in callout_lines),
                                table_headers=None,
                            )
                        )
                    else:
                        low_text_pages += 1
                        warnings.append(
                            f"Page {page_num}: Oversized/vector HVAC diagram with no readable callout text."
                        )
                    continue

                # Standard Digital / OCR Page processing: track headings & emit text/table blocks
                pending_lines: List[str] = []

                def flush_page_text(p_num: int, heading: Optional[str]) -> None:
                    nonlocal block_counter
                    if not pending_lines:
                        return
                    content = "\n\n".join(pending_lines).strip()
                    pending_lines.clear()
                    if not content:
                        return
                    block_counter += 1
                    blocks.append(
                        ContentBlock(
                            block_id=f"pdf-blk-{block_counter:04d}",
                            page_number=p_num,
                            section_heading=heading,
                            sheet_name=None,
                            block_type="text",
                            markdown_content=content,
                            table_headers=None,
                        )
                    )

                for _, kind, payload in combined_events:
                    if kind == "table":
                        flush_page_text(page_num, active_heading)
                        md_tbl, hdrs = payload  # type: ignore[misc]
                        block_counter += 1
                        blocks.append(
                            ContentBlock(
                                block_id=f"pdf-blk-{block_counter:04d}",
                                page_number=page_num,
                                section_heading=active_heading,
                                sheet_name=None,
                                block_type="table",
                                markdown_content=md_tbl,
                                table_headers=hdrs,
                            )
                        )
                    else:
                        # Inspect lines inside text block for section headings
                        paragraph_lines: List[str] = []
                        for raw_line in str(payload).splitlines():
                            line = raw_line.strip()
                            if not line or _PAGE_FOOTER_RE.match(line):
                                continue
                            heading_match = self._detect_pdf_line_heading(line)
                            if heading_match is not None:
                                if paragraph_lines:
                                    pending_lines.append(" ".join(paragraph_lines))
                                    paragraph_lines.clear()
                                flush_page_text(page_num, active_heading)
                                level, clean_h = heading_match
                                active_heading = clean_h
                                md_hashes = "#" * min(level, 4)
                                pending_lines.append(f"{md_hashes} {clean_h}")
                            else:
                                paragraph_lines.append(line)
                        if paragraph_lines:
                            pending_lines.append(" ".join(paragraph_lines))

                flush_page_text(page_num, active_heading)
        finally:
            doc.close()

        # Compute overall extraction confidence from page readability metrics
        total_pages = max(len(inspections), 1)
        if not blocks:
            confidence = 0.2
        elif is_scanned:
            readable_ratio = max(0.0, 1.0 - (low_text_pages / total_pages))
            confidence = round(0.35 + 0.45 * readable_ratio, 3)
        elif is_hvac_diagram:
            total_chars = sum(len(b.markdown_content.strip()) for b in blocks)
            confidence = 0.45 if total_chars < SPARSE_DIAGRAM_CHAR_THRESHOLD else 0.78
        else:
            readable_ratio = max(0.0, 1.0 - (low_text_pages / total_pages))
            confidence = round(0.5 + 0.5 * readable_ratio, 3)

        return blocks, warnings, confidence, title_block_meta, needs_ocr

    @staticmethod
    def _detect_pdf_line_heading(line: str) -> Optional[Tuple[int, str]]:
        """Detect if a single PDF line is a numbered engineering section heading."""
        if len(line) > 110:
            return None
        clean = re.sub(r"^#+\s*", "", line).strip()
        m = ENGINEERING_HEADING_RE.match(clean)
        if not m:
            return None
        num_str = m.group(2)
        title_str = m.group(3).strip()
        # Reject postal codes (e.g. "8000 Brugge"), calendar years (e.g. "2024 Verslag"),
        # and large reference numbers as top-level chapter numbers.
        top_num = int(num_str.split(".")[0])
        if "." not in num_str and top_num >= 100:
            return None
        # Avoid matching table data or measurements like "12.5 m3/h"
        if re.match(r"^(?:m[³3]/h|kW|Pa|mm|cm|kg|EUR|€)\b", title_str, re.IGNORECASE):
            return None
        level = min(num_str.count(".") + 1, 4)
        return level, f"{num_str} {title_str}" if "." in num_str else f"{num_str}. {title_str}"

    @staticmethod
    def _try_local_page_ocr(page: fitz.Page, file_name: str = "", page_num: int = 0) -> str:
        """Attempt local Tesseract OCR via PyMuPDF if Tesseract is installed on the host."""
        try:
            tp = page.get_textpage_ocr(flags=0, full=False)
            return (page.get_text("text", textpage=tp) or "").strip()
        except Exception as exc:
            logger.warning(
                "Local OCR unavailable/failed for '%s' page %d: %s",
                file_name,
                page_num,
                exc,
            )
            return ""
