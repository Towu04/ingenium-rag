"""Canonical domain models, batch/query dataclasses, and pure metadata normalization.

Zero internal dependencies: every stage (`etl`, `chunking`, `embedding`, `storage`,
`retrieval`, `generation`, `evaluation`) communicates exclusively through the contracts
defined in this module.
"""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def compute_doc_id(relative_file_path: str | Path) -> str:
    """Compute a deterministic 16-character hex document ID from a relative POSIX path."""
    rel_posix = Path(relative_file_path).as_posix()
    return hashlib.sha256(rel_posix.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DocCategory(StrEnum):
    """High-level engineering document categories."""

    SPECIFICATION = "specification"  # Bestek / Technical specification
    CALCULATION = "calculation"  # Berekeningsnota
    SCHEDULE_TABLE = "schedule_table"  # Meetstaat / Equipment schedule / Toestellenlijst
    DOCUMENT_REGISTER = "doc_register"  # Documentenlijst / Index
    DRAWING_DIAGRAM = "drawing_diagram"  # Plan / Schema / HVAC diagram
    REPORT_MEETING = "report_meeting"  # Verslag / Report
    UNKNOWN = "unknown"


class ExtractionMethod(StrEnum):
    """Local extraction backend used to parse the source document."""

    PYMUPDF_DIGITAL = "pymupdf_digital"
    PYMUPDF_OCR = "pymupdf_ocr"
    PYMUPDF_DIAGRAM = "pymupdf_diagram"
    PYTHON_DOCX = "python_docx"
    OPENPYXL_TABLE = "openpyxl_table"


# ---------------------------------------------------------------------------
# Pure Metadata Normalization & Flemish Engineering Taxonomy Helpers
# ---------------------------------------------------------------------------

UNKNOWN_SENTINELS: frozenset[str] = frozenset(
    {"UNKNOWN_PROJECT", "UNKNOWN_BUILDING", "UNKNOWN_PHASE"}
)

_BOGUS_BUILDING_LABELS: frozenset[str] = frozenset(
    {"blok peil", "blok nr", "blok no", "blok plan", "blok schema", "blok datum"}
)
_BUILDING_STOPWORDS: frozenset[str] = frozenset(
    {"NR", "NO", "DE", "EN", "IN", "OP", "TE", "VAN", "PE"}
)

_PHASE_CANONICAL_MAP: dict[str, str] = {
    "vo": "Voorontwerp",
    "voorontwerp": "Voorontwerp",
    "preliminary": "Voorontwerp",
    "preliminary design": "Voorontwerp",
    "do": "Definitief Ontwerp",
    "definitief ontwerp": "Definitief Ontwerp",
    "ontwerp": "Definitief Ontwerp",
    "final design": "Definitief Ontwerp",
    "ao": "Aanbesteding",
    "aanbesteding": "Aanbesteding",
    "prijsvraag": "Aanbesteding",
    "bestek": "Aanbesteding",
    "bestekfase": "Aanbesteding",
    "tender": "Aanbesteding",
    "uo": "Uitvoering",
    "uitvoering": "Uitvoering",
    "uitvoeringsdossier": "Uitvoering",
    "execution": "Uitvoering",
    "ab": "As-Built",
    "as-built": "As-Built",
    "as built": "As-Built",
    "oplevering": "As-Built",
    "opleveringsdossier": "As-Built",
}

_DOSSIER_CODE_RE = re.compile(r"^(\d{4,5})[._\-](\d{3}[A-Z]?)$", re.IGNORECASE)
_POSTAL_CODE_HEADING_RE = re.compile(r"^(\d{4})\.?\s+[A-ZÀ-ÖØ-Þ]")

DOCUMENT_REGISTER_FILENAME_RE = re.compile(
    r"(documentenlijst|doc(?:ument)?[_\-\s]*lijst|borderel|document[_\-\s]*register|"
    r"planlijst|plannenlijst|index(?:[_\-\s]*lijst)?|overzicht(?:slijst)?)",
    re.IGNORECASE,
)

PROJECT_ID_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:^|[_\-\s/\\])((?:PRJ|PROJ|ING)[_\-]\d{2,4}(?:[_\-]\d{1,4})?)(?=$|[_\-\s/\\.,;?!])",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![\d.])(0\d{4}[._\-]\d{3}[A-Z]?|\d{4,5}\.\d{3}[A-Z]?)(?![\d]|[._\-]\d)",
        re.IGNORECASE,
    ),
]

_SHAREPOINT_SITE_SLUG_RE = re.compile(r"/sites/([^/?#]+)", re.IGNORECASE)
_SHAREPOINT_SITE_PROJECT_RE = re.compile(
    r"^(\d{4,5})[._\-]?(\d{3}[A-Z]?)(?:$|[-_][A-Za-z0-9]+$)",
    re.IGNORECASE,
)

_PB = r"(?:^|[_\-\s/\\0-9])"
_PA = r"(?:$|[_\-\s/\\.,;?!])"
_PHASE_RULES: list[tuple[re.Pattern[str], re.Pattern[str], str]] = [
    (
        re.compile(_PB + r"(?:voorontwerp|preliminary(?:\s*design)?)" + _PA, re.IGNORECASE),
        re.compile(_PB + r"VO" + _PA, re.IGNORECASE),
        "Voorontwerp",
    ),
    (
        re.compile(
            _PB
            + r"(?:definitief[_\-\s]*ontwerp|final[_\-\s]*design|03[_\-\s]+ontwerp|ontwerp)"
            + _PA,
            re.IGNORECASE,
        ),
        re.compile(_PB + r"DO" + _PA, re.IGNORECASE),
        "Definitief Ontwerp",
    ),
    (
        re.compile(
            _PB + r"(?:aanbesteding|bestek(?:fase)?|prijsvraag|tender)" + _PA, re.IGNORECASE
        ),
        re.compile(_PB + r"AO" + _PA, re.IGNORECASE),
        "Aanbesteding",
    ),
    (
        re.compile(_PB + r"(?:uitvoering(?:sdossier)?|execution)" + _PA, re.IGNORECASE),
        re.compile(_PB + r"UO" + _PA, re.IGNORECASE),
        "Uitvoering",
    ),
    (
        re.compile(_PB + r"(?:as[_\-\s]*built|oplevering(?:sdossier)?)" + _PA, re.IGNORECASE),
        re.compile(_PB + r"AB" + _PA, re.IGNORECASE),
        "As-Built",
    ),
]

_BUILDING_CODE = r"(Noord|Zuid|Oost|West|[A-Z]{1,2}\d{0,2}|\d{1,3})"
_B_END = r"(?=$|[_\-\s/\\.,;?!\)])"
_STRUCTURAL_BUILDING_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:^|[_\-\s/\\])(Blok)[_\-\s]+" + _BUILDING_CODE + _B_END, re.I), "prefix"),
    (re.compile(r"(?:^|[_\-\s/\\])([A-Z]{1,2}\d{0,2})-blok" + _B_END, re.I), "blok_inv"),
    (re.compile(r"(?:^|[_\-\s/\\])(Vleugel)[_\-\s]+" + _BUILDING_CODE + _B_END, re.I), "prefix"),
    (
        re.compile(r"(?:^|[_\-\s/\\])(Gebouw)[_\-\s]+([A-Z]\d{0,2}|\d{1,3})" + _B_END, re.I),
        "prefix",
    ),
    (re.compile(r"(?:^|[_\-\s/\\])(Campus)[_\-\s]+([A-Za-z0-9]+)" + _B_END, re.I), "prefix"),
    (re.compile(r"(?:^|[_\-\s/\\])(Verdieping)[_\s]*([+-]?\d{1,2})" + _B_END, re.I), "floor"),
    (re.compile(r"(?:^|[_\-\s/\\])Niv(?:eau)?\.?[_\s]*([+-]?\d{1,2})" + _B_END, re.I), "niv"),
]

DOC_CATEGORY_RULES: list[tuple[re.Pattern[str], DocCategory]] = [
    (DOCUMENT_REGISTER_FILENAME_RE, DocCategory.DOCUMENT_REGISTER),
    (
        re.compile(r"(bestek|lastenboek|technische[_\-\s]*specificatie)", re.IGNORECASE),
        DocCategory.SPECIFICATION,
    ),
    (
        re.compile(
            r"(berekeningsnota|berekening|calculatie|warmteverlies|koellast|dimensionering)",
            re.IGNORECASE,
        ),
        DocCategory.CALCULATION,
    ),
    (
        re.compile(
            r"(meetstaat|toestellenlijst|materiaallijst|vorderingsstaat|equipment[_\-\s]*schedule)",
            re.IGNORECASE,
        ),
        DocCategory.SCHEDULE_TABLE,
    ),
    (
        re.compile(
            r"(principeschema|schema|grondplan|plan|pid|p&id|isometrie|tekening|diagram)",
            re.IGNORECASE,
        ),
        DocCategory.DRAWING_DIAGRAM,
    ),
    (
        re.compile(
            r"(werfverslag|verslag|notulen|vergadering|rapport|audit|opname)", re.IGNORECASE
        ),
        DocCategory.REPORT_MEETING,
    ),
    (
        re.compile(r"(ventilatie|luchtgroep(?:en)?|debiet(?:en)?|roosterlijst)", re.IGNORECASE),
        DocCategory.SCHEDULE_TABLE,
    ),
]


def strip_unknown_sentinel(value: str | None) -> str | None:
    """Return None for blank strings or legacy UNKNOWN_* sentinel placeholders."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in UNKNOWN_SENTINELS:
        return None
    return cleaned


def normalize_project(value: str | None) -> str | None:
    """Normalize project identifiers (e.g. `1234.001`, `12345-002A` -> `12345.002A`)."""
    cleaned = strip_unknown_sentinel(value)
    if cleaned is None:
        return None
    m = _DOSSIER_CODE_RE.match(cleaned)
    return f"{m.group(1).zfill(5)}.{m.group(2).upper()}" if m else cleaned


def normalize_phase(value: str | None) -> str | None:
    """Normalize verbose bilingual or abbreviated phase strings to canonical Dutch filter keys."""
    cleaned = strip_unknown_sentinel(value)
    if cleaned is None:
        return None
    base = re.sub(r"\s*\([^)]*\)\s*", " ", cleaned).strip()
    key = re.sub(r"[_\s]+", " ", base.lower()).strip()
    return _PHASE_CANONICAL_MAP.get(key, base or None)


def split_buildings(value: str | None) -> list[str]:
    """Split a single or composite building string into clean, canonicalized building tags."""
    cleaned = strip_unknown_sentinel(value)
    if cleaned is None:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"\s+-\s+|[,;/]+", cleaned):
        item = " ".join(part.split()).strip()
        if not item or item.lower() in _BOGUS_BUILDING_LABELS or item.upper() in UNKNOWN_SENTINELS:
            continue
        canonical = infer_building(item)
        for cand in canonical.split(" - ") if canonical else [item]:
            key = cand.lower()
            if key not in seen:
                seen.add(key)
                result.append(cand)
    return result


def normalize_building(value: str | None) -> str | None:
    """Normalize a building string; returns None if empty or bogus."""
    parts = split_buildings(value)
    return " - ".join(parts) if parts else None


def is_bogus_heading(heading: str | None) -> bool:
    """Return True for regex false-positive headings like Belgian postal codes ('1000. Brussel')."""
    if not heading:
        return False
    m = _POSTAL_CODE_HEADING_RE.match(heading.strip())
    return bool(m and 1000 <= int(m.group(1)) <= 9999)


def infer_project_id_from_sharepoint_url(sharepoint_url: str | None) -> str | None:
    """Extract canonical project ID from a SharePoint site URL (e.g. `/sites/12345001-INT/...`)."""
    if not sharepoint_url:
        return None
    slug_match = _SHAREPOINT_SITE_SLUG_RE.search(sharepoint_url)
    if not slug_match:
        return None
    slug = slug_match.group(1)
    proj_match = _SHAREPOINT_SITE_PROJECT_RE.match(slug)
    if proj_match:
        return f"{proj_match.group(1).zfill(5)}.{proj_match.group(2).upper()}"
    for pattern in PROJECT_ID_PATTERNS:
        m = pattern.search(slug)
        if m:
            return normalize_project(m.group(1).upper())
    return None


def infer_project_id_from_path(text_or_path: str | Path) -> str | None:
    """Extract a project identifier code from a relative path or query string."""
    for part in (p for p in Path(text_or_path).parts if p not in {".", "/", "\\"}):
        for pattern in PROJECT_ID_PATTERNS:
            m = pattern.search(part)
            if m:
                return normalize_project(m.group(1).upper())
    return None


def infer_phase(text: str, *, allow_abbreviations: bool = True) -> str | None:
    """Infer canonical Dutch project phase from folder path, filename, or query text."""
    if not text or not text.strip():
        return None
    for full_pattern, abbr_pattern, canonical in _PHASE_RULES:
        if full_pattern.search(text) or (allow_abbreviations and abbr_pattern.search(text)):
            return canonical
    return None


def infer_building(
    text: str,
    *,
    extra_aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Infer generic structural building/wing/floor tokens (plus optional project aliases)."""
    if not text or not text.strip():
        return None
    matched: list[str] = []
    for pattern, kind in _STRUCTURAL_BUILDING_PATTERNS:
        for m in pattern.finditer(text):
            groups = [g.strip() for g in m.groups()]
            if kind == "niv":
                formatted = f"Niv {groups[0]}"
            elif kind == "floor":
                formatted = f"Verdieping {groups[1]}"
            elif kind == "blok_inv":
                code = groups[0].upper()
                if code in _BUILDING_STOPWORDS:
                    continue
                formatted = f"Blok {code}"
            else:
                prefix = groups[0].title()
                label = re.sub(r"[_\-]+", " ", groups[1]).strip()
                if label.upper() in _BUILDING_STOPWORDS:
                    continue
                label = (
                    label.upper()
                    if (len(label) <= 3 or any(c.isdigit() for c in label))
                    else label.title()
                )
                formatted = f"{prefix} {label}"
            if formatted not in matched:
                matched.append(formatted)

    if extra_aliases:
        for alias, canonical in extra_aliases.items():
            if not alias.strip():
                continue
            pat = re.compile(r"(?:^|[_\-\s/\\])" + re.escape(alias.strip()) + _B_END, re.I)
            if pat.search(text) and canonical not in matched:
                matched.append(canonical)

    return " - ".join(matched) if matched else None


def infer_doc_category(file_path: str | Path) -> DocCategory:
    """Infer document category from filename (primary) and parent path (fallback)."""
    path = Path(file_path)
    for target in (path.name, str(path)):
        for pattern, cat in DOC_CATEGORY_RULES:
            if pattern.search(target):
                return cat
    return DocCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Canonical Pipeline Contracts (Pydantic Models)
# ---------------------------------------------------------------------------


class DocumentMetadata(BaseModel):
    """Project-level taxonomy and provenance used for Qdrant filtering and grounded citations."""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(
        ..., description="Relative POSIX file path inside SharePoint or local folder."
    )
    sharepoint_url: str | None = Field(default=None, description="Direct SharePoint web URL.")
    project: str | None = Field(
        default=None, description="Canonical project code (e.g. '12345.001')."
    )
    building: str | None = Field(
        default=None, description="Building/zone label (joined with ' - ' if multiple)."
    )
    buildings: list[str] = Field(
        default_factory=list,
        description="Normalized multi-value building/zone list stored in Qdrant for exact MatchValue filtering.",
    )
    phase: str | None = Field(
        default=None, description="Canonical project phase (e.g. 'Uitvoering')."
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provenance (doc_category, extraction_method, needs_ocr, sharepoint_etag, etc.).",
    )

    @model_validator(mode="after")
    def _normalize_and_sync(self) -> "DocumentMetadata":
        self.project = normalize_project(self.project)
        self.phase = normalize_phase(self.phase)
        if self.buildings:
            cleaned: list[str] = []
            for item in self.buildings:
                cleaned.extend(split_buildings(item))
            # Deduplicate preserving order
            seen: set[str] = set()
            self.buildings = [
                b
                for b in cleaned
                if not (b.lower() in seen or seen.add(b.lower()))  # type: ignore[func-returns-value]
            ]
            self.building = " - ".join(self.buildings) if self.buildings else None
        elif self.building:
            self.buildings = split_buildings(self.building)
            self.building = " - ".join(self.buildings) if self.buildings else None
        return self


class ParsedSection(BaseModel):
    """Layout-aware structural block (prose, Markdown table, or diagram callout) extracted by ETL."""

    text: str = Field(..., description="Extracted Markdown text or serialized Markdown table.")
    page_numbers: list[int] = Field(
        default_factory=list, description="1-based page numbers spanned."
    )
    section_header: str | None = Field(default=None, description="Heading or section breadcrumb.")
    block_type: Literal["text", "table", "diagram_callout"] = Field(
        default="text",
        description="Structural block type preserved from ETL for table-aware chunking.",
    )
    sheet_name: str | None = Field(default=None, description="Excel worksheet name if applicable.")


class ParsedDoc(BaseModel):
    """Canonical document contract emitted directly by `src/etl/` and consumed by `src/chunking/`."""

    document_id: str = Field(
        ..., description="Deterministic 16-char hex ID from relative POSIX path."
    )
    content_hash: str = Field(
        ..., description="SHA-256 hex digest of raw file bytes for delta sync."
    )
    markdown: str = Field(
        ..., description="Complete clean document Markdown (fallback if sections empty)."
    )
    sections: list[ParsedSection] = Field(
        default_factory=list, description="Ordered layout-aware sections."
    )
    metadata: DocumentMetadata = Field(
        ..., description="Normalized project/building/phase taxonomy & provenance."
    )


class ParentChunk(BaseModel):
    """Large un-overlapped context unit (~1000 tokens) stored in Qdrant and sent to the LLM."""

    id: str = Field(..., description="Deterministic ID '{document_id}:p{parent_index}'.")
    document_id: str
    parent_index: int
    text: str
    page_numbers: list[int] = Field(default_factory=list)
    section_header: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    metadata: DocumentMetadata


class ChildChunk(BaseModel):
    """Small overlapping search unit (~250 tokens) embedded via BGE-M3 dense + sparse vectors."""

    id: str = Field(..., description="Deterministic ID '{parent_id}:c{child_index}'.")
    parent_id: str
    document_id: str
    child_index: int
    text: str
    embedding_text: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    section_header: str | None = None
    metadata: DocumentMetadata


class SearchResult(BaseModel):
    """Deduplicated parent-level hit returned by `storage` and reranked by `retrieval`."""

    parent_chunk: ParentChunk
    matched_child_id: str
    score: float
    matched_child_text: str | None = None
    retrieval_score: float | None = None


class RetrievalTrace(BaseModel):
    """Diagnostic trace of a query through metadata filtering, hybrid search, and reranking."""

    query: str
    explicit_filters: dict[str, str] = Field(default_factory=dict)
    auto_filters: dict[str, str] = Field(default_factory=dict)
    applied_filters: dict[str, str] = Field(default_factory=dict)
    fallback_unfiltered: bool = False
    hybrid_child_hits: int = 0
    deduplicated_parents: int = 0
    reranked_kept: int = 0
    reranked_dropped: int = 0
    min_score: float = 0.0
    results: list[SearchResult] = Field(default_factory=list)


class Citation(BaseModel):
    """Verifiable source reference attached to every generated `Answer`."""

    document_id: str
    source_path: str
    sharepoint_url: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    section_header: str | None = None
    parent_chunk_id: str | None = None


class Answer(BaseModel):
    """Final response returned by `RAGPipeline` to CLI, API, and evaluation."""

    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_results: list[SearchResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Lightweight Batch & Query Dataclasses (Boundary between embedding <-> storage)
# ---------------------------------------------------------------------------


@dataclass
class EmbeddedChildBatch:
    """Dense (N x 1024 float32) and sparse lexical weights produced by `embedding`."""

    dense_vecs: np.ndarray
    sparse_weights: list[dict[int, float]]

    def __len__(self) -> int:
        return len(self.dense_vecs)


@dataclass
class EncodedQuery:
    """Single encoded query (1024-dim float32 dense vector + sparse token weights)."""

    dense_vec: np.ndarray
    sparse_weights: dict[int, float]
