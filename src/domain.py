"""Core domain models and data contracts for the Ingenium RAG pipeline.

This module defines the strict data schemas passed between the connectors,
processing (parsing & chunking), storage, retrieval, and generation layers.
"""

from datetime import datetime, timezone
from typing import Any
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Returns the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)

class RawFile(BaseModel):
    """Represents an unparsed file fetched from SharePoint or a local directory.

    Acts as the boundary contract between the `connectors/` layer (ingestion)
    and the `processing/` layer (parsing and metadata extraction).

    Attributes:
        document_id: Unique deterministic ID (e.g., SharePoint driveItem ID or path hash).
        source_path: Relative folder and file path inside SharePoint
            (e.g., `'AZ_Sint-Jan/Blok_Noord/Uitvoering/Bestek_HVAC.pdf'`).
        local_path: Path to the temporary downloaded file on disk ready for parsing.
        content_hash: SHA-256 hash or SharePoint eTag used for incremental delta sync.
        sharepoint_url: Direct web URL to open the original file in SharePoint.
        file_extension: Normalized lowercase extension (e.g., `'.pdf'`, `'.docx'`).
        last_modified: Optional UTC timestamp of the last file modification in SharePoint.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        ...,
        description="Unique deterministic identifier for the file.",
    )
    source_path: str = Field(
        ...,
        description="Relative path in SharePoint used for folder-to-metadata matching.",
    )
    local_path: Path = Field(
        ...,
        description="Local filesystem path where the file is temporarily staged for parsing.",
    )
    content_hash: str = Field(
        ...,
        description="SHA-256 hash or SharePoint eTag for detecting file changes.",
    )
    sharepoint_url: str | None = Field(
        default=None,
        description="Direct web URL to view the file in SharePoint.",
    )
    file_extension: str = Field(
        ...,
        description="Lowercase file extension including dot (e.g., '.pdf', '.docx').",
    )
    last_modified: datetime | None = Field(
        default=None,
        description="Last modified timestamp reported by SharePoint.",
    )

class DocumentMetadata(BaseModel):
    """Project-level taxonomy and provenance extracted from paths and headers.

    Used both for metadata pre-filtering during vector search and for formatting
    source citations in the final response.

    Attributes:
        source_path: Relative file path
            (e.g., `'AZ_Sint-Jan/Blok_Noord/Uitvoering/Bestek_HVAC.pdf'`).
        sharepoint_url: Direct web URL to open the original document in SharePoint.
        project: Matched project identifier or name (e.g., `'PRJ-2024-01'`).
        building: Target hospital building or zone (e.g., `'Blok Noord'`).
        phase: Engineering project phase (e.g., `'Voorontwerp'`, `'Uitvoering'`).
        extra: Arbitrary key-value pairs for domain-specific or custom attributes
            (e.g., discipline, author, revision code).
    """

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(
        ...,
        description="Relative file path inside the repository.",
    )
    sharepoint_url: str | None = Field(
        default=None,
        description="Direct web URL to open the document in SharePoint.",
    )
    project: str | None = Field(
        default=None,
        description="Matched project identifier or name.",
    )
    building: str | None = Field(
        default=None,
        description="Target building or hospital wing.",
    )
    phase: str | None = Field(
        default=None,
        description="Engineering project phase (e.g., design, execution, as-built).",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional unstructured or parser-specific metadata key-value pairs.",
    )


class ParsedSection(BaseModel):
    """A layout-aware structural block (prose section or table) extracted from a document.

    Preserves page boundaries and heading hierarchies prior to parent-child chunking.

    Attributes:
        text: Extracted Markdown content of the section or serialized table.
        page_numbers: 1-based page numbers spanned by this section in the source document.
        section_header: Closest heading or breadcrumb title (e.g., `'3.2 Ventilation Units'`).
    """

    text: str = Field(
        ...,
        description="Extracted Markdown text or serialized Markdown table for this section.",
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description="1-based page numbers spanned by this section.",
    )
    section_header: str | None = Field(
        default=None,
        description="Heading or section breadcrumb associated with this block.",
    )


class ParsedDoc(BaseModel):
    """Standardized output contract produced by the document parsing layer (`processing/parser.py`).

    Consumed by `ParentChildChunker` to generate linked parent and child chunks.

    Attributes:
        document_id: Unique, deterministic identifier for the source document.
        content_hash: SHA-256 hash (or SharePoint eTag) of the raw file for delta sync.
        markdown: Full-document Markdown representation (used as fallback if `sections` is empty).
        sections: Ordered list of layout-aware sections preserving page numbers and headers.
        metadata: Extracted project, building, and phase taxonomy for the document.
    """

    document_id: str = Field(
        ...,
        description="Unique deterministic identifier for the source document.",
    )
    content_hash: str = Field(
        ...,
        description="SHA-256 hash or SharePoint eTag used to detect file modifications.",
    )
    markdown: str = Field(
        ...,
        description="Complete document content serialized as Markdown.",
    )
    sections: list[ParsedSection] = Field(
        default_factory=list,
        description="Layout-aware sections with page and header provenance.",
    )
    metadata: DocumentMetadata = Field(
        ...,
        description="Project-level metadata attached to this document.",
    )


class ParentChunk(BaseModel):
    """Large context unit retrieved and sent to the LLM.

    Parent chunks are stored as plain documents in the database without embeddings
    to preserve surrounding context and prevent overlap artifacts during generation.

    Attributes:
        id: Unique deterministic parent chunk ID (e.g., `'{document_id}:p{parent_index}'`).
        document_id: Foreign key referencing the parent `ParsedDoc.document_id`.
        parent_index: 0-based sequential position of this parent chunk within the document.
        text: Clean, un-overlapped Markdown text or complete table passed to the LLM prompt.
        page_numbers: 1-based page numbers where this parent chunk appears.
        section_header: Document section heading under which this chunk falls.
        child_ids: Ordered list of `ChildChunk.id` strings derived from this parent.
        metadata: Inherited document-level metadata for filtering and citation.
    """

    id: str = Field(
        ...,
        description="Deterministic parent chunk ID ('{document_id}:p{parent_index}').",
    )
    document_id: str = Field(
        ...,
        description="Identifier of the source document this chunk belongs to.",
    )
    parent_index: int = Field(
        ...,
        description="0-based sequential order of this parent chunk within the document.",
    )
    text: str = Field(
        ...,
        description="Full Markdown text of the parent section sent to the LLM.",
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description="1-based source page numbers covered by this parent chunk.",
    )
    section_header: str | None = Field(
        default=None,
        description="Section title or heading breadcrumb for citation.",
    )
    child_ids: list[str] = Field(
        default_factory=list,
        description="Ordered IDs of all ChildChunks split from this parent.",
    )
    metadata: DocumentMetadata = Field(
        ...,
        description="Document metadata inherited from the source ParsedDoc.",
    )


class ChildChunk(BaseModel):
    """Small, overlapping search unit embedded via BGE-M3 and vector-indexed.

    Child chunks provide high-precision semantic and lexical retrieval. When matched,
    their `parent_id` is resolved to return the enclosing `ParentChunk`.

    Attributes:
        id: Unique deterministic child chunk ID (e.g., `'{parent_id}:c{child_index}'`).
        parent_id: Foreign key referencing the enclosing `ParentChunk.id`.
        document_id: Foreign key referencing the source `ParsedDoc.document_id`.
        child_index: 0-based sequential position of this child within its parent chunk.
        text: Raw text slice of the child chunk.
        embedding_text: Context-enriched text (prefixed with project/building/section tags)
            passed to the BGE-M3 embedding model.
        page_numbers: 1-based page numbers inherited from the parent section.
        section_header: Section heading inherited from the parent section.
        metadata: Inherited document-level metadata indexed for pre-filtering.
    """

    id: str = Field(
        ...,
        description="Deterministic child chunk ID ('{parent_id}:c{child_index}').",
    )
    parent_id: str = Field(
        ...,
        description="ID of the enclosing ParentChunk to fetch upon retrieval.",
    )
    document_id: str = Field(
        ...,
        description="ID of the source document.",
    )
    child_index: int = Field(
        ...,
        description="0-based sequential index of this child inside its ParentChunk.",
    )
    text: str = Field(
        ...,
        description="Raw text segment of the child chunk.",
    )
    embedding_text: str | None = Field(
        default=None,
        description="Metadata-prefixed text string fed into the embedding model.",
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description="1-based page numbers where this chunk originates.",
    )
    section_header: str | None = Field(
        default=None,
        description="Section heading breadcrumb for context and citation.",
    )
    metadata: DocumentMetadata = Field(
        ...,
        description="Document metadata used for payload pre-filtering in vector search.",
    )


class SearchResult(BaseModel):
    """Deduplicated output of the retrieval and reranking pipeline.

    Represents a winning `ParentChunk` resolved from the highest-scoring `ChildChunk`.

    Attributes:
        parent_chunk: The full parent context unit to provide to the LLM generator.
        matched_child_id: ID of the specific `ChildChunk` that triggered the match.
        score: Relevance score from hybrid search (RRF) or cross-encoder reranking.
    """

    parent_chunk: ParentChunk = Field(
        ...,
        description="Resolved ParentChunk containing full context for generation.",
    )
    matched_child_id: str = Field(
        ...,
        description="ID of the highest-scoring ChildChunk within this parent.",
    )
    score: float = Field(
        ...,
        description="Relevance score assigned by the retriever or reranker.",
    )


class Citation(BaseModel):
    """Verifiable source reference rendered alongside answers in the LLM.

    Allows an engineer to trace any claim back to the exact document, page, and section[cite: 1, 2].

    Attributes:
        document_id: Identifier of the cited source document.
        source_path: Relative SharePoint file path or filename of the cited document.
        sharepoint_url: Clickable link to open the cited document in SharePoint.
        page_numbers: 1-based page numbers supporting the cited statement[cite: 2].
        section_header: Specific section title or number supporting the claim[cite: 2].
        parent_chunk_id: ID of the `ParentChunk` from which the citation was drawn.
    """

    document_id: str = Field(
        ...,
        description="Identifier of the cited source document.",
    )
    source_path: str = Field(
        ...,
        description="File path or display name of the cited document.",
    )
    sharepoint_url: str | None = Field(
        default=None,
        description="Direct URL to view the source document in SharePoint.",
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description="1-based page numbers referenced by the citation.",
    )
    section_header: str | None = Field(
        default=None,
        description="Document section or heading referenced by the citation.",
    )
    parent_chunk_id: str | None = Field(
        default=None,
        description="ID of the ParentChunk backing this citation.",
    )


class Answer(BaseModel):
    """Final response payload returned by the RAG pipeline to the API or evaluator.

    Attributes:
        text: Natural-language answer synthesized strictly from retrieved context[cite: 2].
        citations: List of verifiable source citations backing the answer[cite: 1, 2].
        retrieved_results: Ordered list of `SearchResult` objects used as context,
            retained for faithfulness and recall evaluation[cite: 2, 3].
    """

    text: str = Field(
        ...,
        description="Grounded natural-language answer generated by the LLM.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Verifiable document, page, and section citations used in the answer.",
    )
    retrieved_results: list[SearchResult] = Field(
        default_factory=list,
        description="Underlying retrieved parent chunks and scores for evaluation.",
    )