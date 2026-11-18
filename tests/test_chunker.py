"""Unit tests for the ParentChildChunker module using the real BAAI/bge-m3 tokenizer."""

import pytest

from processing.chunker import ParentChildChunker, _load_tokenizer
from domain import DocumentMetadata, ParsedDoc, ParsedSection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def preload_real_tokenizer() -> None:
    """Loads the real BAAI/bge-m3 tokenizer once for the entire test session."""
    _load_tokenizer("BAAI/bge-m3")


@pytest.fixture
def base_metadata() -> DocumentMetadata:
    """Provides standard hospital engineering metadata."""
    return DocumentMetadata(
        source_path="AZ_Sint-Jan/Blok_Noord/Uitvoering/HVAC_Bestek.pdf",
        sharepoint_url="https://ingenium.sharepoint.com/sites/az-sint-jan/HVAC_Bestek.pdf",
        project="PRJ-2026-AZSJ",
        building="Blok Noord",
        phase="Uitvoering",
        extra={"discipline": "HVAC"},
    )


@pytest.fixture
def sample_parsed_doc(base_metadata: DocumentMetadata) -> ParsedDoc:
    """Creates a multi-section ParsedDoc calibrated for real BGE-M3 subword token counts."""
    sentence = "De luchtbehandelingskast in blok noord levert verse lucht aan de operatiezaal."
    sec1_text = "\n\n".join([f"Paragraaf {i}: {sentence}" for i in range(1, 10)])
    sec2_text = "\n\n".join([f"Sectie twee {i}: {sentence}" for i in range(1, 4)])

    return ParsedDoc(
        document_id="doc-hvac-001",
        content_hash="sha256-abc123xyz",
        markdown=f"{sec1_text}\n\n{sec2_text}",
        sections=[
            ParsedSection(
                text=sec1_text,
                page_numbers=[1, 2],
                section_header="3.1 Luchtbehandelingskasten",
            ),
            ParsedSection(
                text=sec2_text,
                page_numbers=[3],
                section_header="3.2 Regeling",
            ),
        ],
        metadata=base_metadata,
    )


# ---------------------------------------------------------------------------
# 1. Initialization & Validation Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parent_size, parent_overlap, child_size, child_overlap, expected_msg",
    [
        (0, 0, 250, 40, "Chunk sizes must be positive"),
        (-100, 0, 250, 40, "Chunk sizes must be positive"),
        (1000, 0, 0, 0, "Chunk sizes must be positive"),
        (1000, 0, -50, 0, "Chunk sizes must be positive"),
        (200, 0, 250, 40, "child_size <= parent_size"),
        (1000, 1000, 250, 40, "Overlap must be strictly smaller"),
        (1000, 1200, 250, 40, "Overlap must be strictly smaller"),
        (1000, 0, 250, 250, "Overlap must be strictly smaller"),
        (1000, 0, 250, 300, "Overlap must be strictly smaller"),
    ],
)
def test_init_invalid_parameters_raises_value_error(
    parent_size: int,
    parent_overlap: int,
    child_size: int,
    child_overlap: int,
    expected_msg: str,
) -> None:
    """Verifies constructor guard clauses reject invalid size/overlap combinations."""
    with pytest.raises(ValueError, match=expected_msg):
        ParentChildChunker(
            parent_size=parent_size,
            parent_overlap=parent_overlap,
            child_size=child_size,
            child_overlap=child_overlap,
        )


def test_real_tokenizer_caching_and_max_length() -> None:
    """Verifies _load_tokenizer returns the real cached BGE-M3 tokenizer with updated max_length."""
    tok1 = _load_tokenizer("BAAI/bge-m3")
    tok2 = _load_tokenizer("BAAI/bge-m3")

    assert tok1 is tok2
    assert tok1.model_max_length == 1_000_000
    assert len(tok1.encode("Ventilatie debiet controle")) > 0


# ---------------------------------------------------------------------------
# 2. Core Parent-Child Splitting & Bidirectional Linking Tests
# ---------------------------------------------------------------------------


def test_chunk_bidirectional_references_and_deterministic_ids(
    sample_parsed_doc: ParsedDoc,
) -> None:
    """Ensures every parent links to its children, every child links to its parent, and IDs are deterministic."""
    chunker = ParentChildChunker(
        parent_size=80,
        parent_overlap=0,
        child_size=35,
        child_overlap=10,
    )

    parents, children = chunker.chunk(sample_parsed_doc)

    assert len(parents) >= 3
    assert [p.parent_index for p in parents] == list(range(len(parents)))
    assert [p.id for p in parents] == [
        f"doc-hvac-001:p{i}" for i in range(len(parents))
    ]

    # Verify bidirectional parent <-> child links
    all_children_by_id = {c.id: c for c in children}
    total_linked_children = 0

    for parent in parents:
        assert len(parent.child_ids) > 0
        for expected_c_idx, cid in enumerate(parent.child_ids):
            assert cid in all_children_by_id
            child = all_children_by_id[cid]
            assert child.parent_id == parent.id
            assert child.document_id == sample_parsed_doc.document_id
            assert child.child_index == expected_c_idx
            assert child.id == f"{parent.id}:c{expected_c_idx}"
            total_linked_children += 1

    assert total_linked_children == len(children)


def test_chunk_idempotency(sample_parsed_doc: ParsedDoc) -> None:
    """Running chunk() twice on the same document must yield identical IDs and payloads."""
    chunker = ParentChildChunker(parent_size=80, child_size=35, child_overlap=10)

    parents_1, children_1 = chunker.chunk(sample_parsed_doc)
    parents_2, children_2 = chunker.chunk(sample_parsed_doc)

    assert parents_1 == parents_2
    assert children_1 == children_2


def test_child_overlap_preserves_boundary_words(base_metadata: DocumentMetadata) -> None:
    """Verifies consecutive child chunks share overlapping words when split with real BGE-M3 tokens."""
    sentences = [
        f"UniekeTerm{i} beschrijft de specifieke drukval in klep {i}."
        for i in range(1, 12)
    ]
    text = " ".join(sentences)
    doc = ParsedDoc(
        document_id="doc-overlap",
        content_hash="hash",
        markdown=text,
        sections=[ParsedSection(text=text, page_numbers=[1])],
        metadata=base_metadata,
    )

    chunker = ParentChildChunker(
        parent_size=300,
        parent_overlap=0,
        child_size=50,
        child_overlap=20,
    )
    parents, children = chunker.chunk(doc)

    assert len(parents) == 1
    assert len(children) >= 2

    first_child_words = set(children[0].text.split())
    second_child_words = set(children[1].text.split())
    overlap_words = first_child_words.intersection(second_child_words)

    assert len(overlap_words) > 0, "Expected overlapping words between consecutive children"


# ---------------------------------------------------------------------------
# 3. Provenance, Fallback & Embedding Text Formatting Tests
# ---------------------------------------------------------------------------


def test_provenance_propagation_from_sections(sample_parsed_doc: ParsedDoc) -> None:
    """Checks that page_numbers and section_header propagate to both parents and children."""
    chunker = ParentChildChunker(parent_size=80, child_size=35, child_overlap=10)
    parents, children = chunker.chunk(sample_parsed_doc)

    sec1_parents = [p for p in parents if p.section_header == "3.1 Luchtbehandelingskasten"]
    sec2_parents = [p for p in parents if p.section_header == "3.2 Regeling"]

    assert len(sec1_parents) >= 2
    assert len(sec2_parents) >= 1
    assert all(p.page_numbers == [1, 2] for p in sec1_parents)
    assert all(p.page_numbers == [3] for p in sec2_parents)

    for child in children:
        parent = next(p for p in parents if p.id == child.parent_id)
        assert child.page_numbers == parent.page_numbers
        assert child.section_header == parent.section_header


def test_fallback_to_full_markdown_when_sections_empty(
    base_metadata: DocumentMetadata,
) -> None:
    """Verifies chunker falls back to document.markdown when document.sections is empty."""
    doc = ParsedDoc(
        document_id="doc-fallback",
        content_hash="hash-fb",
        markdown="Dit is een test tekst zonder expliciete secties.",
        sections=[],
        metadata=base_metadata,
    )

    chunker = ParentChildChunker(parent_size=100, child_size=50, child_overlap=10)
    parents, children = chunker.chunk(doc)

    assert len(parents) == 1
    assert len(children) == 1
    assert parents[0].text == "Dit is een test tekst zonder expliciete secties."
    assert parents[0].page_numbers == []
    assert parents[0].section_header is None


def test_embedding_text_formatting_full_and_partial_metadata() -> None:
    """Tests _build_embedding_text with complete, partial, and empty metadata tags."""
    full_meta = DocumentMetadata(
        source_path="a/b.pdf",
        project="PRJ-1",
        building="Blok A",
        phase="Ontwerp",
    )
    doc_full = ParsedDoc(
        document_id="d1", content_hash="h1", markdown="text", metadata=full_meta
    )

    enriched = ParentChildChunker._build_embedding_text(
        "Debiet is 500 m3/h.", doc_full, "4.1 Ventilatie"
    )
    assert (
        enriched
        == "[Project: PRJ-1 | Building: Blok A | Phase: Ontwerp | Section: 4.1 Ventilatie]\nDebiet is 500 m3/h."
    )

    empty_meta = DocumentMetadata(source_path="a/b.pdf")
    doc_empty = ParsedDoc(
        document_id="d2", content_hash="h2", markdown="text", metadata=empty_meta
    )
    plain = ParentChildChunker._build_embedding_text("Debiet is 500 m3/h.", doc_empty, None)
    assert plain == "Debiet is 500 m3/h."


# ---------------------------------------------------------------------------
# 4. Edge Cases & Mutation Isolation Tests
# ---------------------------------------------------------------------------


def test_metadata_deep_copy_prevents_cross_chunk_mutation(
    sample_parsed_doc: ParsedDoc,
) -> None:
    """Mutating metadata on one chunk must never affect the source doc or other chunks."""
    chunker = ParentChildChunker(parent_size=80, child_size=35, child_overlap=10)
    parents, children = chunker.chunk(sample_parsed_doc)

    children[0].metadata.extra["discipline"] = "MUTATED"
    children[0].page_numbers.append(999)

    assert sample_parsed_doc.metadata.extra["discipline"] == "HVAC"
    assert parents[0].metadata.extra["discipline"] == "HVAC"
    assert 999 not in parents[0].page_numbers
    if len(children) > 1:
        assert children[1].metadata.extra["discipline"] == "HVAC"


def test_empty_or_whitespace_only_document_returns_empty_lists(
    base_metadata: DocumentMetadata,
) -> None:
    """Documents with only whitespace or empty sections should return ([], [])."""
    doc = ParsedDoc(
        document_id="doc-empty",
        content_hash="hash-0",
        markdown="   \r\n\t  \n ",
        sections=[
            ParsedSection(text="   \r\n "),
            ParsedSection(text=""),
        ],
        metadata=base_metadata,
    )

    chunker = ParentChildChunker()
    parents, children = chunker.chunk(doc)

    assert parents == []
    assert children == []


def test_windows_crlf_normalization_and_markdown_header_retention(
    base_metadata: DocumentMetadata,
) -> None:
    """Verifies Windows \\r\\n line endings are normalized and Markdown headers stay attached."""
    raw_markdown = (
        "## Hoofdstuk 1 Ventilatie\r\n\r\n"
        "De eerste paragraaf bevat specifieke technische eisen voor de luchtgroep.\r\n"
        "## Hoofdstuk 2 Verwarming\r\n\r\n"
        "De tweede paragraaf bevat specifieke technische eisen voor de warmtepomp."
    )
    doc = ParsedDoc(
        document_id="doc-crlf",
        content_hash="hash-crlf",
        markdown=raw_markdown,
        metadata=base_metadata,
    )

    # Each chapter block is ~22 BGE-M3 tokens; parent_size=28 forces a split right on '\n## '
    chunker = ParentChildChunker(parent_size=28, child_size=28, child_overlap=0)
    parents, children = chunker.chunk(doc)

    assert len(parents) == 2
    assert "\r" not in parents[0].text
    assert "\r" not in parents[1].text
    assert parents[0].text.startswith("## Hoofdstuk 1")
    assert parents[1].text.startswith("## Hoofdstuk 2")