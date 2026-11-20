"""Canonical Pydantic data models for the EMERGENT x Ingenium ETL pipeline.

This module defines the data contract between ETL (Document Extraction & Metadata Matching)
and downstream Ingestion/Indexing for RAG.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# Legacy placeholder strings written by older ETL runs. Unresolved metadata is now `None`,
# but persisted JSON / manifests may still contain these values, so they are coerced on load.
LEGACY_UNKNOWN_SENTINELS = frozenset({"UNKNOWN_PROJECT", "UNKNOWN_BUILDING", "UNKNOWN_PHASE"})


def coerce_unknown_to_none(value: Any) -> Any:
    """Map legacy `UNKNOWN_*` sentinels and blank strings to `None`."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.upper() in LEGACY_UNKNOWN_SENTINELS:
            return None
        return stripped
    return value


class DocCategory(str, Enum):
    """High-level engineering document categories for AZ Sint-Jan Brugge."""

    SPECIFICATION = "specification"       # Bestek / Technical specification
    CALCULATION = "calculation"           # Berekeningsnota
    SCHEDULE_TABLE = "schedule_table"     # Meetstaat / Equipment schedule / Toestellenlijst
    DOCUMENT_REGISTER = "doc_register"    # Documentenlijst / Index
    DRAWING_DIAGRAM = "drawing_diagram"   # Plan / Schema / HVAC diagram
    REPORT_MEETING = "report_meeting"     # Verslag / Report
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    """Local extraction backend used to parse the document."""

    PYMUPDF_DIGITAL = "pymupdf_digital"
    PYMUPDF_OCR = "pymupdf_ocr"
    PYMUPDF_DIAGRAM = "pymupdf_diagram"
    PYTHON_DOCX = "python_docx"
    OPENPYXL_TABLE = "openpyxl_table"


class DocumentMetadata(BaseModel):
    """Canonical metadata attached to every extracted engineering document."""

    doc_id: str = Field(..., description="Deterministic ID based on relative path or SHA256")
    file_name: str
    file_extension: str
    file_hash_sha256: str
    sharepoint_url: Optional[str] = None
    relative_folder_path: str
    sharepoint_item_id: Optional[str] = Field(
        default=None,
        description="Microsoft Graph DriveItem ID when streamed from SharePoint",
    )
    sharepoint_etag: Optional[str] = Field(
        default=None,
        description="Microsoft Graph DriveItem eTag/cTag for remote change detection",
    )

    # Core Domain Metadata required by Ingenium (None = could not be resolved)
    project_id: Optional[str] = Field(default=None)
    project_name: str = Field(default="AZ Sint-Jan Brugge")
    building: Optional[str] = Field(
        default=None,
        description="Building/Wing/Block/Floor identifier; multiple values joined with ' - '",
    )
    phase: Optional[str] = Field(
        default=None,
        description="Project phase, e.g., Voorontwerp, Aanbesteding, Uitvoering, As-Built",
    )
    doc_category: DocCategory = Field(default=DocCategory.UNKNOWN)
    revision: Optional[str] = None
    doc_code: Optional[str] = None
    resolution_sources: List[str] = Field(
        default_factory=list,
        description="Which matcher layers resolved the metadata (empty for legacy JSON outputs)",
    )

    # Audit & Quality Metadata
    extraction_method: ExtractionMethod = Field(default=ExtractionMethod.PYMUPDF_DIGITAL)
    is_scanned: bool = False
    is_hvac_diagram: bool = False
    needs_ocr: bool = Field(
        default=False,
        description="True when (part of) the file is a raster scan whose text could not be extracted",
    )
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    quality_warnings: List[str] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=_utc_now)

    @field_validator("project_id", "building", "phase", mode="before")
    @classmethod
    def _coerce_sentinels(cls, value: Any) -> Any:
        return coerce_unknown_to_none(value)


class ContentBlock(BaseModel):
    """A single layout-aware block (text, table, or diagram callout) with exact citation anchor."""

    block_id: str
    page_number: Optional[int] = Field(None, description="1-indexed page number for PDFs")
    section_heading: Optional[str] = Field(
        None,
        description="Active hierarchical heading, e.g., '3.2 Ventilation Units'",
    )
    sheet_name: Optional[str] = Field(None, description="Excel sheet name if applicable")
    block_type: str = Field(..., description="'text', 'table', or 'diagram_callout'")
    markdown_content: str = Field(
        ...,
        description="Clean Markdown representation (tables preserved)",
    )
    table_headers: Optional[List[str]] = None


class ExtractedDocument(BaseModel):
    """Complete extracted representation of a single file (.json and .md contract).

    To avoid triple text storage on disk, `blocks` and `metadata` are serialized to `.json`
    while `full_markdown_with_anchors` is dynamically assembled on demand when writing `.md`.
    """

    model_config = ConfigDict(extra="ignore")

    metadata: DocumentMetadata
    blocks: List[ContentBlock] = Field(default_factory=list)

    def render_markdown(self) -> str:
        """Assemble the full Markdown document with explicit HTML citation anchor comments."""
        from etl.extractors.base import assemble_full_markdown

        return assemble_full_markdown(self.metadata, self.blocks)

    @property
    def full_markdown_with_anchors(self) -> str:
        """Dynamically compute full anchored Markdown without duplicating it in JSON output."""
        return self.render_markdown()


class ProjectCatalogEntry(BaseModel):
    """Row from the master Google Sheet project catalog."""

    project_id: str
    project_name: str = Field(default="AZ Sint-Jan Brugge")
    folder_name: Optional[str] = None
    sharepoint_url: Optional[str] = None
    building: Optional[str] = None
    phase: Optional[str] = None
    client: str = Field(default="AZ Sint-Jan Brugge")
    extra: Dict[str, Any] = Field(default_factory=dict)


class DocumentRegisterEntry(BaseModel):
    """Row parsed from an Excel Document Register (Documentenlijst / Borderel / Index)."""

    doc_code: Optional[str] = None
    file_name: Optional[str] = None
    title: Optional[str] = None
    building: Optional[str] = None
    phase: Optional[str] = None
    revision: Optional[str] = None
    discipline: Optional[str] = None
    doc_category: Optional[DocCategory] = None
    source_register_path: Optional[str] = None


class MatchedMetadata(BaseModel):
    """Resolved domain metadata produced by MetadataMatcher before or during extraction."""

    project_id: Optional[str] = None
    project_name: str = Field(default="AZ Sint-Jan Brugge")
    sharepoint_url: Optional[str] = None
    building: Optional[str] = None
    phase: Optional[str] = None
    doc_category: DocCategory = Field(default=DocCategory.UNKNOWN)
    revision: Optional[str] = None
    doc_code: Optional[str] = None
    resolution_sources: List[str] = Field(default_factory=list)
