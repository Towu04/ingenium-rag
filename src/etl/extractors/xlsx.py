"""Excel (`.xlsx`) extractor emitting `domain.ParsedDoc` directly."""

import re
from pathlib import Path

import openpyxl

from domain import DocCategory, ExtractionMethod, ParsedDoc, ParsedSection
from etl.catalog import is_document_register_file
from etl.extractors.common import build_markdown_table, clean_cell_text, finalize_sections
from etl.metadata import MetadataMatcher

_NUMERIC_CELL_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?%?$")
_PLACEHOLDER_SHEET_RE = re.compile(r"^(?:x\d+|template\d*|sjabloon\d*)$", re.IGNORECASE)
_EMPTY_OR_ZERO_RE = re.compile(r"^(?:0+(?:[.,]0+)?|-|n/?a)$", re.IGNORECASE)
_SUBTOTAL_LABEL_RE = re.compile(r"^(?:sub)?totaal\b", re.IGNORECASE)


def _is_key_value_metadata_row(row_vals: list[str]) -> bool:
    non_empty = [c for c in row_vals if c]
    return len(non_empty) == 2 and non_empty[0].endswith(":")


def _is_placeholder_row(row_vals: list[str]) -> bool:
    non_empty = [c for c in row_vals if c]
    if not non_empty:
        return True
    if all(_EMPTY_OR_ZERO_RE.match(c) for c in non_empty):
        return True
    return (
        len(non_empty) == 2
        and bool(_SUBTOTAL_LABEL_RE.match(non_empty[0]))
        and bool(_EMPTY_OR_ZERO_RE.match(non_empty[1]))
    )


def _is_unused_template_sheet(sheet_name: str, data_rows: list[list[str]]) -> bool:
    """Skip placeholder-named template tabs (e.g. `x17`, `template`) with no populated data rows."""
    if not _PLACEHOLDER_SHEET_RE.match(sheet_name.strip()):
        return False
    return not any(not _is_placeholder_row(r) for r in data_rows)


def _detect_table_header_index(non_empty_rows: list[tuple[int, list[str]]]) -> int:
    best_idx, best_score = 0, -1
    for idx, (_, row_vals) in enumerate(non_empty_rows[:15]):
        non_empty = [c for c in row_vals if c]
        if len(non_empty) < 2 or _is_key_value_metadata_row(row_vals):
            continue
        text_labels = [c for c in non_empty if not _NUMERIC_CELL_RE.match(c)]
        score = len(text_labels) * 10 + len(non_empty)
        if score > best_score:
            best_score, best_idx = score, idx
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
                non_empty_rows: list[tuple[int, list[str]]] = []
                for row_idx, row in enumerate(wb[sheet_name].iter_rows(values_only=True), start=1):
                    cleaned = [clean_cell_text(c) for c in row]
                    while cleaned and not cleaned[-1]:
                        cleaned.pop()
                    if any(cleaned):
                        non_empty_rows.append((row_idx, cleaned))

                if not non_empty_rows:
                    continue

                table_start_idx = _detect_table_header_index(non_empty_rows)
                table_rows = non_empty_rows[table_start_idx:]
                if not table_rows:
                    continue

                raw_data_rows = [r for _, r in table_rows[1:]]
                if _is_unused_template_sheet(sheet_name, raw_data_rows):
                    continue

                preamble_lines: list[str] = []
                for _, row_vals in non_empty_rows[:table_start_idx]:
                    non_empty = [c for c in row_vals if c]
                    if len(non_empty) == 1:
                        preamble_lines.append(f"**{non_empty[0]}**")
                    elif _is_key_value_metadata_row(row_vals):
                        preamble_lines.append(f"- **{non_empty[0]}** {non_empty[1]}")
                    elif non_empty:
                        preamble_lines.append(" | ".join(non_empty))

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

                filtered_data_rows = [r for r in raw_data_rows if not _is_placeholder_row(r)]
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
                            section_header=f"{sheet_name} (Rows {table_rows[0][0]}-{table_rows[-1][0]})",
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
