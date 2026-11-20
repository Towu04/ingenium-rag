"""PDF (`.pdf`) extractor emitting `domain.ParsedDoc` directly.

Supports digital PDFs, scanned PDFs (with explicit `needs_ocr` flagging), and oversized HVAC/CAD
diagrams with readability filtering of noisy OCR callout blocks.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz

from config import get_settings
from domain import (
    DocCategory,
    ExtractionMethod,
    ParsedDoc,
    ParsedSection,
    infer_building,
    infer_phase,
    split_buildings,
)
from etl.extractors.common import (
    ENGINEERING_HEADING_RE,
    build_markdown_table,
    clean_cell_text,
    finalize_sections,
)
from etl.metadata import MetadataMatcher

logger = logging.getLogger(__name__)

_PAGE_FOOTER_RE = re.compile(
    r"^(?:\d{1,3}|(?:web:\s*\S+\s+)?(?:zijde|pagina|page|blz\.?)\s*\d+\s*(?:/|van|of)\s*\d+)$",
    re.IGNORECASE,
)

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

    page_number: int
    width_pt: float
    height_pt: float
    selectable_char_count: int
    image_count: int
    vector_drawing_count: int
    is_oversized: bool
    is_scanned_page: bool
    is_hvac_diagram_page: bool


def inspect_pdf_document(pdf_path: str | Path) -> list[PageInspection]:
    """Classify each PDF page as digital, scanned, or HVAC/CAD diagram via PyMuPDF."""
    settings = get_settings()
    doc = fitz.open(str(pdf_path))
    inspections: list[PageInspection] = []
    try:
        for idx, page in enumerate(doc):
            rect = page.rect
            width_pt, height_pt = float(rect.width), float(rect.height)
            raw_text = page.get_text("text") or ""
            char_count = len(re.sub(r"\s+", "", raw_text))
            image_count = len(page.get_images(full=False))
            try:
                vector_count = len(page.get_drawings())
            except Exception:
                vector_count = 0

            is_oversized = max(width_pt, height_pt) > settings.oversized_page_min_dimension_pt
            is_scanned_page = (
                char_count < settings.min_digital_text_chars_per_page
                and not is_oversized
                and (image_count > 0 or vector_count < 20)
            )
            has_hvac_callouts = len(HVAC_CALLOUT_TOKEN_RE.findall(raw_text)) >= 2
            is_hvac_diagram_page = is_oversized or (
                vector_count >= settings.high_vector_drawings_threshold
                and (has_hvac_callouts or char_count < 300)
            )
            inspections.append(
                PageInspection(
                    page_number=idx + 1,
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


def _rect_overlaps(outer: fitz.Rect, inner: fitz.Rect) -> bool:
    cx = (inner.x0 + inner.x1) / 2.0
    cy = (inner.y0 + inner.y1) / 2.0
    return (outer.x0 - 2 <= cx <= outer.x1 + 2) and (outer.y0 - 2 <= cy <= outer.y1 + 2)


class PdfExtractor:
    """Extracts page-anchored Markdown prose, tables, and diagram callouts into `ParsedDoc`."""

    def __init__(self, metadata_matcher: MetadataMatcher | None = None) -> None:
        self.metadata_matcher = metadata_matcher or MetadataMatcher()

    def extract(
        self,
        file_path: str | Path,
        relative_path: str | Path | None = None,
        *,
        sharepoint_url: str | None = None,
        drop_noisy_callouts: bool = True,
    ) -> ParsedDoc:
        path = Path(file_path)
        rel_path = Path(relative_path) if relative_path is not None else Path(path.name)

        inspections = inspect_pdf_document(path)
        total_pages = len(inspections)
        scanned_pages = sum(1 for p in inspections if p.is_scanned_page)

        matched_cat = self.metadata_matcher.match(
            rel_path, sharepoint_url=sharepoint_url
        ).doc_category
        if matched_cat in {
            DocCategory.REPORT_MEETING,
            DocCategory.SPECIFICATION,
            DocCategory.CALCULATION,
            DocCategory.SCHEDULE_TABLE,
        }:
            for p in inspections:
                if not p.is_oversized:
                    p.is_hvac_diagram_page = False

        diagram_pages = sum(1 for p in inspections if p.is_hvac_diagram_page)
        is_scanned = total_pages > 0 and (scanned_pages / total_pages) >= 0.5
        is_hvac_diagram = total_pages > 0 and (
            (diagram_pages / total_pages) >= 0.5 or matched_cat == DocCategory.DRAWING_DIAGRAM
        )

        if is_scanned:
            method = ExtractionMethod.PYMUPDF_OCR
        elif is_hvac_diagram:
            method = ExtractionMethod.PYMUPDF_DIAGRAM
        else:
            method = ExtractionMethod.PYMUPDF_DIGITAL

        raw_sections, warnings, confidence, title_meta, needs_ocr = self._extract_pages(
            path, inspections, is_scanned=is_scanned, is_hvac_diagram=is_hvac_diagram
        )
        sections, dropped_callouts = finalize_sections(
            raw_sections,
            coalesce_prose=True,
            drop_noisy_callouts=drop_noisy_callouts,
        )
        if dropped_callouts:
            logger.info(
                "Quarantined %d noisy diagram_callout block(s) from '%s'.",
                dropped_callouts,
                rel_path.name,
            )
        if is_scanned and not sections:
            needs_ocr = True

        doc_id, content_hash, metadata = self.metadata_matcher.build_document_metadata(
            file_path=path,
            relative_file_path=rel_path,
            extraction_method=method,
            is_scanned=is_scanned,
            is_hvac_diagram=is_hvac_diagram,
            needs_ocr=needs_ocr,
            extraction_confidence=confidence,
            quality_warnings=warnings,
            dropped_noisy_callouts=dropped_callouts,
            sharepoint_url=sharepoint_url,
        )

        # Enrich unresolved building/phase from drawing title block or Page-1 cover text
        if metadata.building is None and title_meta.get("building"):
            bldgs = split_buildings(title_meta["building"])
            metadata.buildings = bldgs
            metadata.building = " - ".join(bldgs) if bldgs else None
        if metadata.building is None and sections:
            page1_text = "\n".join(s.text for s in sections if 1 in s.page_numbers)[:1200]
            if not page1_text:
                page1_text = "\n".join(s.text for s in sections)[:1200]
            if inferred_bldg := infer_building(page1_text):
                bldgs = split_buildings(inferred_bldg)
                metadata.buildings = bldgs
                metadata.building = " - ".join(bldgs) if bldgs else None
                sources = list(metadata.extra.get("resolution_sources", []))
                sources.append("cover_page_building")
                metadata.extra["resolution_sources"] = sources
        if metadata.phase is None and title_meta.get("phase"):
            metadata.phase = title_meta["phase"]
        if is_hvac_diagram and metadata.extra.get("doc_category") == DocCategory.UNKNOWN.value:
            metadata.extra["doc_category"] = DocCategory.DRAWING_DIAGRAM.value

        return ParsedDoc(
            document_id=doc_id,
            content_hash=content_hash,
            markdown="\n\n".join(s.text for s in sections),
            sections=sections,
            metadata=metadata,
        )

    def _extract_pages(
        self,
        pdf_path: Path,
        inspections: list[PageInspection],
        *,
        is_scanned: bool,
        is_hvac_diagram: bool,
    ) -> tuple[list[ParsedSection], list[str], float, dict[str, str], bool]:
        settings = get_settings()
        doc = fitz.open(str(pdf_path))
        sections: list[ParsedSection] = []
        warnings: list[str] = []
        active_heading: str | None = None
        title_block_meta: dict[str, str] = {}
        low_text_pages = 0
        needs_ocr = False

        try:
            for idx, page in enumerate(doc):
                page_num = idx + 1
                inspection = inspections[idx]

                # 1. Tables via PyMuPDF find_tables()
                table_rects: list[fitz.Rect] = []
                page_tables: list[tuple[float, str]] = []
                try:
                    for tbl in page.find_tables().tables:
                        raw_grid: list[list[str]] = []
                        if hasattr(tbl, "rows") and tbl.rows:
                            for row in tbl.rows:
                                raw_grid.append(
                                    [
                                        clean_cell_text(
                                            page.get_text("text", clip=fitz.Rect(cell)) or ""
                                        )
                                        if cell is not None
                                        else ""
                                        for cell in row.cells
                                    ]
                                )
                        else:
                            raw_grid = [
                                [clean_cell_text(c) for c in row] for row in (tbl.extract() or [])
                            ]
                        cleaned = [r for r in raw_grid if any(r)]
                        if len(cleaned) < 2 or sum(1 for c in cleaned[0] if c) < 2:
                            continue
                        md_tbl, _ = build_markdown_table(cleaned[0], cleaned[1:])
                        if md_tbl:
                            t_rect = fitz.Rect(tbl.bbox)
                            table_rects.append(t_rect)
                            page_tables.append((t_rect.y0, md_tbl))
                except Exception:
                    pass

                # 2. Text blocks outside detected tables
                raw_blocks = sorted(
                    page.get_text("blocks") or [],
                    key=lambda b: (round(float(b[1]), 1), round(float(b[0]), 1)),
                )
                text_items: list[tuple[float, str]] = []
                for b in raw_blocks:
                    if len(b) >= 7 and b[6] != 0:
                        continue
                    b_rect = fitz.Rect(b[0], b[1], b[2], b[3])
                    if any(_rect_overlaps(t_rect, b_rect) for t_rect in table_rects):
                        continue
                    txt = str(b[4]).strip()
                    if txt:
                        text_items.append((float(b[1]), txt))

                # 3. Scanned-page OCR fallback
                if inspection.is_scanned_page and not text_items and not page_tables:
                    ocr_text = self._try_local_page_ocr(page, pdf_path.name, page_num)
                    if ocr_text:
                        text_items.append((0.0, ocr_text))
                    else:
                        low_text_pages += 1
                        needs_ocr = True
                        warnings.append(
                            f"Page {page_num}: Scanned raster page with < "
                            f"{settings.min_digital_text_chars_per_page} selectable chars and no OCR output."
                        )

                full_page_text = "\n".join(t for _, t in text_items)
                if is_hvac_diagram or inspection.is_hvac_diagram_page:
                    if "building" not in title_block_meta:
                        if bldg := infer_building(full_page_text):
                            title_block_meta["building"] = bldg
                    if "phase" not in title_block_meta:
                        if ph := infer_phase(full_page_text, allow_abbreviations=False):
                            title_block_meta["phase"] = ph

                # 4. Merge in vertical reading order
                events: list[tuple[float, str, str]] = [
                    *((y0, "text", txt) for y0, txt in text_items),
                    *((y0, "table", md_tbl) for y0, md_tbl in page_tables),
                ]
                events.sort(key=lambda item: item[0])

                if is_hvac_diagram or inspection.is_hvac_diagram_page:
                    callout_lines: list[str] = []
                    for _, kind, payload in events:
                        if kind == "text":
                            callout_lines.extend(
                                line.strip() for line in payload.splitlines() if line.strip()
                            )
                        else:
                            sections.append(
                                ParsedSection(
                                    text=payload,
                                    page_numbers=[page_num],
                                    section_header=active_heading
                                    or "HVAC Diagram Legend / Schedule",
                                    block_type="table",
                                )
                            )
                    if callout_lines:
                        sections.append(
                            ParsedSection(
                                text="\n".join(f"- {line}" for line in callout_lines),
                                page_numbers=[page_num],
                                section_header=active_heading or "HVAC Diagram Callouts",
                                block_type="diagram_callout",
                            )
                        )
                    else:
                        low_text_pages += 1
                        warnings.append(
                            f"Page {page_num}: Oversized/vector HVAC diagram with no readable callout text."
                        )
                    continue

                # Standard Digital / OCR Page
                pending_lines: list[str] = []

                def flush_text(p_num: int, heading: str | None) -> None:
                    if not pending_lines:
                        return
                    content = "\n\n".join(pending_lines).strip()
                    pending_lines.clear()
                    if content:
                        sections.append(
                            ParsedSection(
                                text=content,
                                page_numbers=[p_num],
                                section_header=heading,
                                block_type="text",
                            )
                        )

                for _, kind, payload in events:
                    if kind == "table":
                        flush_text(page_num, active_heading)
                        sections.append(
                            ParsedSection(
                                text=payload,
                                page_numbers=[page_num],
                                section_header=active_heading,
                                block_type="table",
                            )
                        )
                    else:
                        para_lines: list[str] = []
                        for raw_line in payload.splitlines():
                            line = raw_line.strip()
                            if not line or _PAGE_FOOTER_RE.match(line):
                                continue
                            heading_match = self._detect_line_heading(line)
                            if heading_match is not None:
                                if para_lines:
                                    pending_lines.append(" ".join(para_lines))
                                    para_lines.clear()
                                flush_text(page_num, active_heading)
                                level, clean_h = heading_match
                                active_heading = clean_h
                                pending_lines.append(f"{'#' * min(level, 4)} {clean_h}")
                            else:
                                para_lines.append(line)
                        if para_lines:
                            pending_lines.append(" ".join(para_lines))

                flush_text(page_num, active_heading)
        finally:
            doc.close()

        total_pages = max(len(inspections), 1)
        if not sections:
            confidence = 0.2
        elif is_scanned:
            readable_ratio = max(0.0, 1.0 - (low_text_pages / total_pages))
            confidence = round(0.35 + 0.45 * readable_ratio, 3)
        elif is_hvac_diagram:
            total_chars = sum(len(s.text.strip()) for s in sections)
            confidence = 0.45 if total_chars < settings.sparse_diagram_char_threshold else 0.78
        else:
            readable_ratio = max(0.0, 1.0 - (low_text_pages / total_pages))
            confidence = round(0.5 + 0.5 * readable_ratio, 3)

        return sections, warnings, confidence, title_block_meta, needs_ocr

    @staticmethod
    def _detect_line_heading(line: str) -> tuple[int, str] | None:
        if len(line) > 110:
            return None
        clean = re.sub(r"^#+\s*", "", line).strip()
        m = ENGINEERING_HEADING_RE.match(clean)
        if not m:
            return None
        num_str, title_str = m.group(2), m.group(3).strip()
        if "." not in num_str and int(num_str) >= 100:
            return None
        if re.match(r"^(?:m[³3]/h|kW|Pa|mm|cm|kg|EUR|€)\b", title_str, re.IGNORECASE):
            return None
        level = min(num_str.count(".") + 1, 4)
        return level, f"{num_str} {title_str}" if "." in num_str else f"{num_str}. {title_str}"

    @staticmethod
    def _try_local_page_ocr(page: fitz.Page, file_name: str = "", page_num: int = 0) -> str:
        try:
            tp = page.get_textpage_ocr(flags=0, full=False)
            return (page.get_text("text", textpage=tp) or "").strip()
        except Exception as exc:
            logger.warning(
                "Local OCR unavailable/failed for '%s' page %d: %s", file_name, page_num, exc
            )
            return ""
