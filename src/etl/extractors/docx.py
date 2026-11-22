"""Word (`.docx`) extractor emitting `domain.ParsedDoc` directly."""

import re
from pathlib import Path

import docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from config import get_settings
from domain import (
    ExtractionMethod,
    ParsedDoc,
    ParsedSection,
    infer_building,
    split_buildings,
)
from etl.extractors.common import (
    ENGINEERING_HEADING_RE,
    build_markdown_table,
    clean_cell_text,
    finalize_sections,
)
from etl.metadata import MetadataMatcher


def _iter_docx_blocks(doc: DocxDocument):
    if hasattr(doc, "iter_inner_content"):
        yield from doc.iter_inner_content()
        return
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _detect_heading_level(paragraph: Paragraph) -> int | None:
    text = paragraph.text.strip()
    if not text:
        return None
    style_lower = ((paragraph.style.name if paragraph.style else "") or "").lower()

    if "toc" in style_lower or "inhoud" in style_lower:
        return None
    if re.search(r"(?:\t|\.{4,}\s*)\d{1,4}$", text):
        return None

    if style_lower in {"title", "titel"}:
        return 1
    match_style = re.search(r"(?:heading|kop|titre)\s*(\d)", style_lower)
    if match_style:
        return min(int(match_style.group(1)), 4)

    if len(text) <= 120 and "\n" not in text:
        m_num = ENGINEERING_HEADING_RE.match(text)
        if m_num:
            num_part = m_num.group(2)
            depth = num_part.count(".") + 1
            runs = [r for r in paragraph.runs if r.text.strip()]
            is_bold = bool(runs) and all(bool(r.bold) for r in runs)
            if is_bold or depth <= 3:
                return min(depth, 4)
    return None


class DocxExtractor:
    """Extracts hierarchical headings, prose, and Markdown tables from `.docx` into `ParsedDoc`."""

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

        doc = docx.Document(str(path))
        heading_hierarchy: dict[int, str] = {}
        active_heading: str | None = None
        raw_sections: list[ParsedSection] = []
        pending_paragraphs: list[str] = []

        def flush_text() -> None:
            if not pending_paragraphs:
                return
            combined = "\n\n".join(pending_paragraphs).strip()
            pending_paragraphs.clear()
            if combined:
                raw_sections.append(
                    ParsedSection(
                        text=combined,
                        page_numbers=[],
                        section_header=active_heading,
                        block_type="text",
                    )
                )

        for item in _iter_docx_blocks(doc):
            if isinstance(item, Paragraph):
                raw_text = item.text.strip()
                if not raw_text:
                    continue
                level = _detect_heading_level(item)
                if level is not None:
                    flush_text()
                    clean_heading = re.sub(r"^#+\s*", "", raw_text).strip()
                    heading_hierarchy[level] = clean_heading
                    for deeper in [k for k in heading_hierarchy if k > level]:
                        del heading_hierarchy[deeper]
                    active_heading = " > ".join(
                        heading_hierarchy[k] for k in sorted(heading_hierarchy)
                    )
                    pending_paragraphs.append(f"{'#' * min(level, 4)} {clean_heading}")
                else:
                    pending_paragraphs.append(raw_text)

            elif isinstance(item, Table):
                flush_text()
                raw_rows: list[list[str]] = []
                banner_text: str | None = None
                for row in item.rows:
                    seen_tc: set[int] = set()
                    dedup_cells: list[str] = []
                    for cell in row.cells:
                        tc_id = id(cell._tc)
                        if tc_id in seen_tc:
                            dedup_cells.append("")
                        else:
                            seen_tc.add(tc_id)
                            dedup_cells.append(clean_cell_text(cell.text))
                    if not any(dedup_cells):
                        continue
                    if not raw_rows and len(seen_tc) == 1 and len(row.cells) > 1:
                        banner_text = dedup_cells[0]
                        continue
                    raw_rows.append(dedup_cells)
                if not raw_rows:
                    if banner_text:
                        pending_paragraphs.append(banner_text)
                    continue
                md_table, _ = build_markdown_table(raw_rows[0], raw_rows[1:])
                if md_table:
                    table_header = active_heading
                    if banner_text:
                        table_header = (
                            f"{active_heading} > {banner_text}" if active_heading else banner_text
                        )
                    raw_sections.append(
                        ParsedSection(
                            text=md_table,
                            page_numbers=[],
                            section_header=table_header,
                            block_type="table",
                        )
                    )

        flush_text()
        sections, _ = finalize_sections(
            raw_sections, coalesce_prose=True, drop_noisy_callouts=drop_noisy_callouts
        )

        doc_id, content_hash, metadata = self.metadata_matcher.build_document_metadata(
            file_path=path,
            relative_file_path=rel_path,
            extraction_method=ExtractionMethod.PYTHON_DOCX,
            extraction_confidence=1.0 if sections else 0.2,
            sharepoint_url=sharepoint_url,
        )
        if metadata.building is None and sections:
            cover_text = "\n".join(s.text for s in sections)[:1200]
            aliases = get_settings().building_aliases or None
            if inferred_bldg := infer_building(cover_text, extra_aliases=aliases):
                bldgs = split_buildings(inferred_bldg)
                metadata.buildings = bldgs
                metadata.building = " - ".join(bldgs) if bldgs else None
                sources = list(metadata.extra.get("resolution_sources", []))
                sources.append("cover_page_building")
                metadata.extra["resolution_sources"] = sources

        return ParsedDoc(
            document_id=doc_id,
            content_hash=content_hash,
            markdown="\n\n".join(s.text for s in sections),
            sections=sections,
            metadata=metadata,
        )
