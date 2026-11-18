"""Word (.docx) specification extractor preserving heading hierarchy and inline Markdown tables."""

from pathlib import Path
import re
from typing import Dict, List, Optional, Union

import docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from etl.extractors.base import (
    BaseExtractor,
    ENGINEERING_HEADING_RE,
    build_markdown_table,
    clean_cell_text,
)
from etl.schemas import (
    ContentBlock,
    ExtractedDocument,
    ExtractionMethod,
)


def _iter_docx_blocks(doc: DocxDocument):
    """Yield Paragraph and Table objects in exact document body order."""
    if hasattr(doc, "iter_inner_content"):
        yield from doc.iter_inner_content()
        return

    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _detect_heading_level(paragraph: Paragraph) -> Optional[int]:
    """Detect whether a Word paragraph is a heading and return its level (1..4) or None."""
    text = paragraph.text.strip()
    if not text:
        return None

    style_name = (paragraph.style.name if paragraph.style else "") or ""
    style_lower = style_name.lower()

    # Standard English & Dutch Word heading styles (Heading 1..4, Kop 1..4, Title)
    if style_lower in {"title", "titel"}:
        return 1
    match_style = re.search(r"(?:heading|kop|titre)\s*(\d)", style_lower)
    if match_style:
        return min(int(match_style.group(1)), 4)

    # Fallback: numbered engineering heading pattern (e.g., "1. Algemene Voorwaarden",
    # "3.2 Luchtgroepen (Ventilation Units)") when short and either bold or single-line
    if len(text) <= 120 and "\n" not in text:
        m_num = ENGINEERING_HEADING_RE.match(text)
        if m_num:
            num_part = m_num.group(2)
            depth = num_part.count(".") + 1
            # Check if runs are bold or if it starts with a clear section number
            runs = [r for r in paragraph.runs if r.text.strip()]
            is_bold = bool(runs) and all(bool(r.bold) for r in runs)
            if is_bold or depth <= 3:
                return min(depth, 4)

    return None


class DocxExtractor(BaseExtractor):
    """Extracts structured blocks and Markdown tables from Word (.docx) specifications."""

    def extract(
        self,
        file_path: Union[str, Path],
        relative_path: Optional[Union[str, Path]] = None,
    ) -> ExtractedDocument:
        path = Path(file_path)
        rel_path = Path(relative_path) if relative_path is not None else Path(path.name)

        doc = docx.Document(str(path))
        heading_hierarchy: Dict[int, str] = {}
        active_heading: Optional[str] = None

        blocks: List[ContentBlock] = []
        pending_paragraphs: List[str] = []
        block_counter = 0

        def flush_pending_text() -> None:
            nonlocal block_counter
            if not pending_paragraphs:
                return
            combined = "\n\n".join(pending_paragraphs).strip()
            pending_paragraphs.clear()
            if not combined:
                return
            block_counter += 1
            blocks.append(
                ContentBlock(
                    block_id=f"docx-blk-{block_counter:04d}",
                    page_number=None,
                    section_heading=active_heading,
                    sheet_name=None,
                    block_type="text",
                    markdown_content=combined,
                    table_headers=None,
                )
            )

        for item in _iter_docx_blocks(doc):
            if isinstance(item, Paragraph):
                raw_text = item.text.strip()
                if not raw_text:
                    continue

                level = _detect_heading_level(item)
                if level is not None:
                    flush_pending_text()
                    # Clean leading markdown hashes if any
                    clean_heading = re.sub(r"^#+\s*", "", raw_text).strip()
                    heading_hierarchy[level] = clean_heading
                    # Clear deeper levels
                    for deeper in [k for k in heading_hierarchy if k > level]:
                        del heading_hierarchy[deeper]

                    ordered_headings = [
                        heading_hierarchy[k] for k in sorted(heading_hierarchy.keys())
                    ]
                    active_heading = " > ".join(ordered_headings)
                    md_hashes = "#" * min(level, 4)
                    pending_paragraphs.append(f"{md_hashes} {clean_heading}")
                else:
                    pending_paragraphs.append(raw_text)

            elif isinstance(item, Table):
                flush_pending_text()
                raw_rows: List[List[str]] = []
                for row in item.rows:
                    cells = [clean_cell_text(cell.text) for cell in row.cells]
                    if any(cells):
                        raw_rows.append(cells)

                if not raw_rows:
                    continue

                headers = raw_rows[0]
                data_rows = raw_rows[1:] if len(raw_rows) > 1 else []
                md_table, clean_headers = build_markdown_table(headers, data_rows)
                if not md_table:
                    continue

                block_counter += 1
                blocks.append(
                    ContentBlock(
                        block_id=f"docx-blk-{block_counter:04d}",
                        page_number=None,
                        section_heading=active_heading,
                        sheet_name=None,
                        block_type="table",
                        markdown_content=md_table,
                        table_headers=clean_headers,
                    )
                )

        flush_pending_text()

        metadata = self.metadata_matcher.build_document_metadata(
            file_path=path,
            relative_file_path=rel_path,
            extraction_method=ExtractionMethod.PYTHON_DOCX,
            is_scanned=False,
            is_hvac_diagram=False,
            extraction_confidence=1.0 if blocks else 0.2,
            quality_warnings=[],
        )

        return ExtractedDocument(
            metadata=metadata,
            blocks=blocks,
        )
