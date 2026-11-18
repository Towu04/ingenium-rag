"""Excel (.xlsx) extractor for engineering schedules (Meetstaten / Toestellenlijsten) and Document Registers."""

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


class XlsxExtractor(BaseExtractor):
    """Parses Excel workbooks into per-worksheet Markdown tables with sheet_name anchors.

    Note: Document Register indexing (Layer 1) is handled exclusively during Pass 1
    by `DocumentRegisterIndex` in `catalog.py` to avoid double-parsing workbooks.
    """

    def extract(
        self,
        file_path: Union[str, Path],
        relative_path: Optional[Union[str, Path]] = None,
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

                # Separate optional single-cell title header rows at the top from the table header
                preamble_lines: List[str] = []
                table_start_idx = 0
                for idx, (_, row_vals) in enumerate(non_empty_rows):
                    non_blank_count = sum(1 for c in row_vals if c)
                    if non_blank_count >= 2:
                        table_start_idx = idx
                        break
                    preamble_lines.append(" ".join(c for c in row_vals if c))
                else:
                    table_start_idx = 0

                if preamble_lines:
                    block_counter += 1
                    blocks.append(
                        ContentBlock(
                            block_id=f"xlsx-blk-{block_counter:04d}",
                            page_number=None,
                            section_heading=f"{sheet_name} (Title Block)",
                            sheet_name=sheet_name,
                            block_type="text",
                            markdown_content="\n".join(f"**{line}**" for line in preamble_lines),
                            table_headers=None,
                        )
                    )

                table_rows = non_empty_rows[table_start_idx:]
                if not table_rows:
                    continue

                start_row_num = table_rows[0][0]
                end_row_num = table_rows[-1][0]

                # Determine maximum column width across the table
                max_cols = max(len(r) for _, r in table_rows)
                header_cells = list(table_rows[0][1])
                if len(header_cells) < max_cols:
                    header_cells.extend(
                        [f"Col_{i + 1}" for i in range(len(header_cells), max_cols)]
                    )

                data_rows = [r for _, r in table_rows[1:]]
                md_table, clean_headers = build_markdown_table(header_cells, data_rows)
                if not md_table:
                    continue

                block_counter += 1
                section_heading = f"{sheet_name} (Rows {start_row_num}-{end_row_num})"
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
        )
        if is_register and metadata.doc_category == DocCategory.UNKNOWN:
            metadata.doc_category = DocCategory.DOCUMENT_REGISTER

        return ExtractedDocument(
            metadata=metadata,
            blocks=blocks,
        )
