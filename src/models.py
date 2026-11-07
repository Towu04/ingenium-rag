from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentMetadata(BaseModel):
    """Project-level taxonomy matched from SharePoint paths and headers."""
    model_config = ConfigDict(extra="forbid")

    source_path: str
    sharepoint_url: str | None = None
    project: str | None = None
    building: str | None = None
    phase: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ParsedSection(BaseModel):
    """A layout-aware block (section or table) extracted from a document."""
    text: str
    page_numbers: list[int] = Field(default_factory=list)
    section_header: str | None = None


class ParsedDoc(BaseModel):
    document_id: str
    content_hash: str  # SHA-256 hash for incremental SharePoint delta sync
    markdown: str      # Full document markdown fallback
    sections: list[ParsedSection] = Field(default_factory=list)
    metadata: DocumentMetadata
    parsed_at: datetime = Field(default_factory=utc_now)


class ParentChunk(BaseModel):
    """Context unit sent to the LLM (full section or complete Markdown table)."""
    id: str
    document_id: str
    parent_index: int
    text: str          # Keep full text here so overlaps don't corrupt LLM context!
    page_numbers: list[int] = Field(default_factory=list)
    section_header: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    metadata: DocumentMetadata


class ChildChunk(BaseModel):
    """Search unit embedded via BGE-M3 and indexed in the vector store."""
    id: str
    parent_id: str
    document_id: str
    child_index: int
    text: str
    embedding_text: str | None = None  # Text prefixed with metadata for BGE-M3
    page_numbers: list[int] = Field(default_factory=list)
    section_header: str | None = None
    metadata: DocumentMetadata


class SearchResult(BaseModel):
    """Output of retrieval: returns the deduplicated ParentChunk to the generator."""
    parent_chunk: ParentChunk
    matched_child_id: str
    score: float


class Citation(BaseModel):
    """Verifiable source reference"""
    document_id: str
    source_path: str
    sharepoint_url: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    section_header: str | None = None
    parent_chunk_id: str | None = None


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_results: list[SearchResult] = Field(default_factory=list)  # Crucial for Elisabeth's evaluation!