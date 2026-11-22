"""Shared Markdown table rendering, cell cleaning, and section-coalescing helpers for ETL."""

import re
from collections.abc import Sequence

from config import get_settings
from domain import ParsedDoc, ParsedSection, is_bogus_heading
from etl.quality import score_callout_readability

# Matches numbered engineering headings such as "1. Algemene Voorwaarden", "3.2 Luchtgroepen"
ENGINEERING_HEADING_RE = re.compile(
    r"^(?:#+\s*)?((\d+(?:\.\d+){0,3})\.?)\s+([A-ZÀ-ÖØ-Þ][^\n]{2,100})$"
)


def clean_cell_text(value: object) -> str:
    """Clean a table cell value for Markdown table rendering without breaking pipes or newlines."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    text = re.sub(r"[\r\n\t]+", " ", text)
    return text.replace("|", r"\|").strip()


def build_markdown_table(
    headers: Sequence[object], rows: Sequence[Sequence[object]]
) -> tuple[str, list[str]]:
    """Render headers and data rows as a valid Markdown table and return `(markdown, clean_headers)`."""
    clean_headers = [clean_cell_text(h) for h in headers]
    num_cols = len(clean_headers)
    if num_cols == 0:
        return "", []

    display_headers = [h if h else f"Col_{idx + 1}" for idx, h in enumerate(clean_headers)]
    lines = [
        "| " + " | ".join(display_headers) + " |",
        "| " + " | ".join([":---"] * num_cols) + " |",
    ]
    for row in rows:
        cleaned_row = [clean_cell_text(c) for c in row]
        if len(cleaned_row) < num_cols:
            cleaned_row.extend([""] * (num_cols - len(cleaned_row)))
        elif len(cleaned_row) > num_cols:
            cleaned_row = cleaned_row[:num_cols]
        if any(cleaned_row):
            lines.append("| " + " | ".join(cleaned_row) + " |")
    return "\n".join(lines), display_headers


def finalize_sections(
    raw_sections: Sequence[ParsedSection],
    *,
    coalesce_prose: bool = True,
    drop_noisy_callouts: bool = True,
    callout_readability_threshold: float | None = None,
) -> tuple[list[ParsedSection], int]:
    """Sanitize headings, quarantine unreadable `diagram_callout`s, and coalesce cross-page prose."""
    threshold = (
        callout_readability_threshold
        if callout_readability_threshold is not None
        else get_settings().callout_readability_threshold
    )
    sections: list[ParsedSection] = []
    dropped_callouts = 0

    for sec in raw_sections:
        text = sec.text.strip()
        if not text:
            continue

        if sec.block_type == "diagram_callout" and drop_noisy_callouts:
            if score_callout_readability(text) < threshold:
                dropped_callouts += 1
                continue

        header = sec.section_header.strip() if sec.section_header else None
        if header and is_bogus_heading(header):
            header = None

        if (
            coalesce_prose
            and sec.block_type == "text"
            and sections
            and sections[-1].block_type == "text"
            and sections[-1].section_header == header
        ):
            prev = sections[-1]
            sections[-1] = ParsedSection(
                text=f"{prev.text}\n\n{text}",
                page_numbers=sorted(set(prev.page_numbers) | set(sec.page_numbers)),
                section_header=header,
                block_type="text",
                sheet_name=prev.sheet_name or sec.sheet_name,
            )
            continue

        sections.append(
            ParsedSection(
                text=text,
                page_numbers=sorted(set(sec.page_numbers)),
                section_header=header,
                block_type=sec.block_type,
                sheet_name=sec.sheet_name,
            )
        )

    return sections, dropped_callouts


def render_markdown_export(doc: ParsedDoc) -> str:
    """Render a human-readable `.md` export with provenance header comments for inspection."""
    meta = doc.metadata
    header = (
        f"<!-- DOC: {meta.source_path} | PROJECT: {meta.project or 'N/A'} | "
        f"BUILDING: {meta.building or 'N/A'} | PHASE: {meta.phase or 'N/A'} -->"
    )
    parts: list[str] = [header]
    for sec in doc.sections:
        anchors: list[str] = []
        if sec.page_numbers:
            anchors.append(f"PAGES: {','.join(map(str, sec.page_numbers))}")
        if sec.sheet_name:
            anchors.append(f"SHEET: {sec.sheet_name}")
        anchors.append(f"SECTION: {sec.section_header or sec.sheet_name or 'General'}")
        parts.append(f"<!-- {' | '.join(anchors)} -->\n\n{sec.text}")
    return "\n\n".join(parts) + "\n"
