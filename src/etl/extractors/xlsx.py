"""Excel (`.xlsx`) extractor emitting `domain.ParsedDoc` directly."""

import re
from pathlib import Path

import openpyxl

from domain import DocCategory, ExtractionMethod, ParsedDoc, ParsedSection
from etl.catalog import is_document_register_file
from etl.extractors.common import build_markdown_table, clean_cell_text, finalize_sections
from etl.metadata import MetadataMatcher

_NUMERIC_CELL_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?%?$")
_TEMPLATE_PLACEHOLDER_RE = re.compile(
    r"^(?:wb\+?\s*perceel\s*volgnummer|0(?:\.0+)?)$",
    re.IGNORECASE,
)


def _is_unused_wb_template_sheet(
    sheet_name: str, non_empty_rows: list[tuple[int, list[str]]]
) -> bool:
    """Return True if a worksheet is an unfilled Ingenium WB template tab (e.g. `x17`..`x100`)."""
    flat_text = " ".join(c for _, r in non_empty_rows[:12] for c in r if c).lower()
    if "wb+ perceel volgnummer" not in flat_text:
        return False
    if re.match(r"^x\d+$", sheet_name.strip(), re.IGNORECASE):
        return True
    for _, r in non_empty_rows[:12]:
        non_empty = [c for c in r if c]
        if non_empty and non_empty[0].lower().startswith("onderwerp:") and len(non_empty) == 1:
            return True
    return False


def _is_key_value_metadata_row(row_vals: list[str]) -> bool:
    non_empty = [c for c in row_vals if c]
    return len(non_empty) == 2 and non_empty[0].endswith(":")


def _is_template_placeholder_row(row_vals: list[str]) -> bool:
    non_empty = [c for c in row_vals if c]
    return not non_empty or all(_TEMPLATE_PLACEHOLDER_RE.match(c) for c in non_empty)


def _detect_table_header_index(non_empty_rows: list[tuple[int, list[str]]]) -> int:
    best_idx = 0
    best_score = -1
    for idx, (_, row_vals) in enumerate(non_empty_rows[:15]):
        non_empty = [c for c in row_vals if c]
        if len(non_empty) < 2 or _is_key_value_metadata_row(row_vals):
            continue
        text_labels = [c for c in non_empty if not _NUMERIC_CELL_RE.match(c)]
        score = len(text_labels) * 10 + len(non_empty)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


class XlsxExtractor:
    """Parses Excel workbooks into per-worksheet Markdown table sections in `ParsedDoc`."""

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
        is_register = is_document_register_file(rel_path) or is_document_register_file(path)

        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        raw_sections: list[ParsedSection] = []
        try:
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                non_empty_rows: list[tuple[int, list[str]]] = []
                for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
                    cleaned = [clean_cell_text(c) for c in row]
                    while cleaned and not cleaned[-1]:
                        cleaned.pop()
                    if any(cleaned):
                        non_empty_rows.append((row_idx, cleaned))

                if not non_empty_rows or _is_unused_wb_template_sheet(sheet_name, non_empty_rows):
                    continue

                table_start_idx = _detect_table_header_index(non_empty_rows)
                preamble_lines: list[str] = []
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
                filtered_data_rows = [
                    r for _, r in table_rows[1:] if not _is_template_placeholder_row(r)
                ]

                if preamble_lines:
                    raw_sections.append(
                        ParsedSection(
                            text="\n".join(preamble_lines),
                            page_numbers=[],
                            section_header=f"{sheet_name} (Title Block)",
                            block_type="text",
                            sheet_name=sheet_name,
                        )
                    )

                all_kept = [table_rows[0][1]] + filtered_data_rows
                max_cols = max(len(r) for r in all_kept)
                header_cells = list(table_rows[0][1])
                if len(header_cells) < max_cols:
                    header_cells.extend(f"Col_{i + 1}" for i in range(len(header_cells), max_cols))

                md_table, _ = build_markdown_table(header_cells, filtered_data_rows)
                if md_table:
                    raw_sections.append(
                        ParsedSection(
                            text=md_table,
                            page_numbers=[],
                            section_header=f"{sheet_name} (Rows {header_row_num}-{end_row_num})",
                            block_type="table",
                            sheet_name=sheet_name,
                        )
                    )
        finally:
            wb.close()

        sections, _ = finalize_sections(
            raw_sections, coalesce_prose=True, drop_noisy_callouts=drop_noisy_callouts
        )

        doc_id, content_hash, metadata = self.metadata_matcher.build_document_metadata(
            file_path=path,
            relative_file_path=rel_path,
            extraction_method=ExtractionMethod.OPENPYXL_TABLE,
            extraction_confidence=1.0 if sections else 0.2,
            sharepoint_url=sharepoint_url,
        )
        if is_register and metadata.extra.get("doc_category") == DocCategory.UNKNOWN.value:
            metadata.extra["doc_category"] = DocCategory.DOCUMENT_REGISTER.value

        return ParsedDoc(
            document_id=doc_id,
            content_hash=content_hash,
            markdown="\n\n".join(s.text for s in sections),
            sections=sections,
            metadata=metadata,
        )
