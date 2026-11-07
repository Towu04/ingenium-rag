from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class RawFile(BaseModel):
    document_id: str
    source_path: str
    content: bytes
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDoc(BaseModel):
    document_id: str
    source_path: str
    markdown: str
    metadata: "DocumentMetadata"
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DocumentMetadata(BaseModel):
    source_path: str
    document_id: str | None = None
    project: str | None = None
    building: str | None = None
    phase: str | None = None
    page_number: int | None = None
    section: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ParentChunk(BaseModel):
    id: str
    document_id: str
    text: str
    metadata: DocumentMetadata


class ChildChunk(BaseModel):
    id: str
    parent_id: str
    document_id: str
    text: str
    metadata: DocumentMetadata


class Citation(BaseModel):
    document_id: str
    source_path: str
    page_number: int | None = None
    section: str | None = None
    chunk_id: str | None = None


class SearchResult(BaseModel):
    chunk: ChildChunk
    score: float


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)


ParsedDoc.model_rebuild()
