"""Excel (.xlsx) extractor for engineering schedules (Meetstaten / Toestellenlijsten) and Document Registers."""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import openpyxl

from etl.catalog import is_document_register_file
from etl.extractors.base import (
    BaseExtractor,
    build_markdown_table,
    clean_cell_text,
)
from etl.schemas import (
    ContentBlock,
    DocCategory,
    ExtractedDocument,
    ExtractionMethod,
)

_NUMERIC_CELL_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?%?$")
_TEMPLATE_PLACEHOLDER_RE = re.compile(
    r"^(?:wb\+?\s*perceel\s*volgnummer|0(?:\.0+)?)$",
    re.IGNORECASE,
)


def _is_key_value_metadata_row(row_vals: List[str]) -> bool:
    """Return True when a row is a 2-cell `Label: | Value` metadata line above a table."""
    non_empty = [c for c in row_vals if c]
    if len(non_empty) != 2:
        return False
    return non_empty[0].endswith(":")


def _is_template_placeholder_row(row_vals: List[str]) -> bool:
    """Return True when a row contains only empty/zero/template placeholder cells."""
    non_empty = [c for c in row_vals if c]
    if not non_empty:
        return True
    return all(_TEMPLATE_PLACEHOLDER_RE.match(c) for c in non_empty)


def _detect_table_header_index(non_empty_rows: List[Tuple[int, List[str]]]) -> int:
    """Locate the real column-header row within the top 15 rows, skipping title & key-value rows."""
    window = non_empty_rows[:15]
    best_idx = 0
    best_score = -1

    for idx, (_, row_vals) in enumerate(window):
        non_empty = [c for c in row_vals if c]
        if len(non_empty) < 2 or _is_key_value_metadata_row(row_vals):
            continue
        # Prefer rows with many distinct non-numeric text header labels (and few trailing blanks)
        text_labels = [c for c in non_empty if not _NUMERIC_CELL_RE.match(c)]
        score = len(text_labels) * 10 + len(non_empty)
        if score > best_score:
            best_score = score
            best_idx = idx
        # Early exit when the very first non-title, non-KV row is already a dense header (>=3 labels)
        if idx <= 1 and len(text_labels) >= 3 and len(text_labels) == len(non_empty):
            # Still check if a much wider header appears within the next few rows (e.g. WB sheets)
            continue

    return best_idx


class XlsxExtractor(BaseExtractor):
    """Parses Excel workbooks into per-worksheet Markdown tables with sheet_name anchors.

    Note: Document Register indexing (Layer 1) is handled exclusively during Pass 1
    by `DocumentRegisterIndex` in `catalog.py` to avoid double-parsing workbooks.
    """

    def extract(
        self,
        file_path: Union[str, Path],
        relative_path: Optional[Union[str, Path]] = None,
        sharepoint_url: Optional[str] = None,
    ) -> ExtractedDocument:
        path = Path(file_path)
        rel_path = Path(relative_path) if relative_path is not None else Path(path.name)
        is_register = is_document_register_file(rel_path) or is_document_register_file(path)

        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        blocks: List[ContentBlock] = []
        block_counter = 0

        try:
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                non_empty_rows: List[Tuple[int, List[str]]] = []

                for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
                    cleaned = [clean_cell_text(c) for c in row]
                    # Trim trailing empty cells
                    while cleaned and not cleaned[-1]:
                        cleaned.pop()
                    if any(cleaned):
                        non_empty_rows.append((row_idx, cleaned))

                if not non_empty_rows:
                    continue

                # Separate title & key-value metadata rows at the top from the real column header
                table_start_idx = _detect_table_header_index(non_empty_rows)
                preamble_lines: List[str] = []
                for _, row_vals in non_empty_rows[:table_start_idx]:
                    non_empty = [c for c in row_vals if c]
                    if not non_empty:
                        continue
                    if len(non_empty) == 1:
                        preamble_lines.append(f"**{non_empty[0]}**")
                    elif _is_key_value_metadata_row(row_vals):
                        preamble_lines.append(f"- **{non_empty[0]}** {non_empty[1]}")
                    else:
                        preamble_lines.append(" | ".join(non_empty))

                table_rows = non_empty_rows[table_start_idx:]
                if not table_rows:
                    continue

                header_row_num = table_rows[0][0]
                end_row_num = table_rows[-1][0]

                # Filter out empty/zero template placeholder rows from the data body
                filtered_data_rows = [
                    r for _, r in table_rows[1:] if not _is_template_placeholder_row(r)
                ]

                if preamble_lines:
                    block_counter += 1
                    blocks.append(
                        ContentBlock(
                            block_id=f"xlsx-blk-{block_counter:04d}",
                            page_number=None,
                            section_heading=f"{sheet_name} (Title Block)",
                            sheet_name=sheet_name,
                            block_type="text",
                            markdown_content="\n".join(preamble_lines),
                            table_headers=None,
                        )
                    )

                # Determine maximum column width across header + kept data rows
                all_kept = [table_rows[0][1]] + filtered_data_rows
                max_cols = max(len(r) for r in all_kept)
                header_cells = list(table_rows[0][1])
                if len(header_cells) < max_cols:
                    header_cells.extend(
                        [f"Col_{i + 1}" for i in range(len(header_cells), max_cols)]
                    )

                md_table, clean_headers = build_markdown_table(header_cells, filtered_data_rows)
                if not md_table:
                    continue

                block_counter += 1
                section_heading = f"{sheet_name} (Rows {header_row_num}-{end_row_num})"
                blocks.append(
                    ContentBlock(
                        block_id=f"xlsx-blk-{block_counter:04d}",
                        page_number=None,
                        section_heading=section_heading,
                        sheet_name=sheet_name,
                        block_type="table",
                        markdown_content=md_table,
                        table_headers=clean_headers,
                    )
                )
        finally:
            wb.close()

        metadata = self.metadata_matcher.build_document_metadata(
            file_path=path,
            relative_file_path=rel_path,
            extraction_method=ExtractionMethod.OPENPYXL_TABLE,
            is_scanned=False,
            is_hvac_diagram=False,
            extraction_confidence=1.0 if blocks else 0.2,
            quality_warnings=[],
            sharepoint_url=sharepoint_url,
        )
        if is_register and metadata.doc_category == DocCategory.UNKNOWN:
            metadata.doc_category = DocCategory.DOCUMENT_REGISTER

        return ExtractedDocument(
            metadata=metadata,
            blocks=blocks,
        )
