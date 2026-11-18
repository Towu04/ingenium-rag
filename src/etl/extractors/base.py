"""Abstract base extractor and Markdown/anchor formatting utilities."""

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import List, Optional, Sequence, Union

from etl.metadata_matcher import MetadataMatcher
from etl.schemas import ContentBlock, DocumentMetadata, ExtractedDocument

# Matches numbered engineering headings such as:
# "1. Algemene Voorwaarden", "3.2 Luchtgroepen (Ventilation Units)", "4.2.1 Filters"
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
    # Replace internal newlines/tabs with spaces and escape pipe characters
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = text.replace("|", r"\|")
    return text.strip()


def build_markdown_table(
    headers: Sequence[object], rows: Sequence[Sequence[object]]
) -> tuple[str, List[str]]:
    """Render headers and data rows as a valid Markdown table and return (markdown, clean_headers)."""
    clean_headers = [clean_cell_text(h) for h in headers]
    num_cols = len(clean_headers)
    if num_cols == 0:
        return "", []

    # Ensure every header has at least a placeholder name if blank
    display_headers = [
        h if h else f"Col_{idx + 1}" for idx, h in enumerate(clean_headers)
    ]

    lines: List[str] = [
        "| " + " | ".join(display_headers) + " |",
        "| " + " | ".join([":---"] * num_cols) + " |",
    ]

    for row in rows:
        cleaned_row = [clean_cell_text(c) for c in row]
        # Pad or trim to match header column count
        if len(cleaned_row) < num_cols:
            cleaned_row.extend([""] * (num_cols - len(cleaned_row)))
        elif len(cleaned_row) > num_cols:
            cleaned_row = cleaned_row[:num_cols]

        if any(cleaned_row):
            lines.append("| " + " | ".join(cleaned_row) + " |")

    return "\n".join(lines), display_headers


def build_doc_header_comment(metadata: DocumentMetadata) -> str:
    """Build the top-level document provenance HTML comment required by Section 6.D."""
    return (
        f"<!-- DOC: {metadata.file_name} | PROJECT: {metadata.project_id} | "
        f"BUILDING: {metadata.building} | PHASE: {metadata.phase} -->"
    )


def build_block_anchor_comment(block: ContentBlock) -> str:
    """Build the page/section/sheet HTML anchor comment before a content block."""
    parts: List[str] = []
    if block.page_number is not None:
        parts.append(f"PAGE: {block.page_number}")
    if block.sheet_name:
        parts.append(f"SHEET: {block.sheet_name}")
    section_label = block.section_heading or block.sheet_name or "General"
    parts.append(f"SECTION: {section_label}")
    return f"<!-- {' | '.join(parts)} -->"


def assemble_full_markdown(
    metadata: DocumentMetadata, blocks: Sequence[ContentBlock]
) -> str:
    """Assemble full Markdown string with explicit document and page/section anchor comments."""
    segments: List[str] = [build_doc_header_comment(metadata)]
    last_anchor: Optional[str] = None

    for block in blocks:
        if not block.markdown_content.strip():
            continue
        anchor = build_block_anchor_comment(block)
        if anchor != last_anchor:
            segments.append(anchor)
            last_anchor = anchor
        segments.append(block.markdown_content.strip())

    return "\n\n".join(segments) + "\n"


class BaseExtractor(ABC):
    """Abstract interface for all local engineering document extractors."""

    def __init__(self, metadata_matcher: Optional[MetadataMatcher] = None) -> None:
        self.metadata_matcher = metadata_matcher or MetadataMatcher()

    @abstractmethod
    def extract(
        self,
        file_path: Union[str, Path],
        relative_path: Optional[Union[str, Path]] = None,
    ) -> ExtractedDocument:
        """Extract content blocks and metadata from a local engineering document."""
        raise NotImplementedError
