"""Unit tests for Stage 2 (`chunking.ParentChildChunker`) using the real BAAI/bge-m3 tokenizer."""

import pytest

from chunking import ParentChildChunker, load_tokenizer
from domain import DocumentMetadata, ParsedDoc, ParsedSection


@pytest.fixture(scope="session", autouse=True)
def preload_real_tokenizer() -> None:
    load_tokenizer("BAAI/bge-m3")


@pytest.fixture
def base_metadata() -> DocumentMetadata:
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
    sentence = "De luchtbehandelingskast in blok noord levert verse lucht aan de operatiezaal."
    sec1_text = "\n\n".join([f"Paragraaf {i}: {sentence}" for i in range(1, 10)])
    sec2_text = "\n\n".join([f"Sectie twee {i}: {sentence}" for i in range(1, 4)])
    return ParsedDoc(
        document_id="doc-hvac-001",
        content_hash="sha256-abc123xyz",
        markdown=f"{sec1_text}\n\n{sec2_text}",
        sections=[
            ParsedSection(
                text=sec1_text, page_numbers=[1, 2], section_header="3.1 Luchtbehandelingskasten"
            ),
            ParsedSection(text=sec2_text, page_numbers=[3], section_header="3.2 Regeling"),
        ],
        metadata=base_metadata,
    )


@pytest.mark.parametrize(
    "parent_size, parent_overlap, child_size, child_overlap, expected_msg",
    [
        (0, 0, 250, 40, "Chunk sizes must be positive"),
        (-100, 0, 250, 40, "Chunk sizes must be positive"),
        (1000, 0, 0, 0, "Chunk sizes must be positive"),
        (200, 0, 250, 40, "child_size <= parent_size"),
        (1000, 1000, 250, 40, "Overlap must be strictly smaller"),
        (1000, 0, 250, 250, "Overlap must be strictly smaller"),
    ],
)
def test_init_invalid_parameters_raises_value_error(
    parent_size: int,
    parent_overlap: int,
    child_size: int,
    child_overlap: int,
    expected_msg: str,
) -> None:
    with pytest.raises(ValueError, match=expected_msg):
        ParentChildChunker(
            parent_size=parent_size,
            parent_overlap=parent_overlap,
            child_size=child_size,
            child_overlap=child_overlap,
        )


def test_real_tokenizer_caching_and_max_length() -> None:
    tok1 = load_tokenizer("BAAI/bge-m3")
    tok2 = load_tokenizer("BAAI/bge-m3")
    assert tok1 is tok2
    assert tok1.model_max_length == 1_000_000


def test_chunk_bidirectional_references_and_idempotency(sample_parsed_doc: ParsedDoc) -> None:
    chunker = ParentChildChunker(parent_size=80, parent_overlap=0, child_size=35, child_overlap=10)
    parents, children = chunker.chunk(sample_parsed_doc)
    parents2, children2 = chunker.chunk(sample_parsed_doc)
    assert parents == parents2
    assert children == children2
    assert len(parents) >= 3
    assert [p.id for p in parents] == [f"doc-hvac-001:p{i}" for i in range(len(parents))]

    child_map = {c.id: c for c in children}
    for parent in parents:
        assert parent.child_ids
        for c_idx, cid in enumerate(parent.child_ids):
            child = child_map[cid]
            assert child.parent_id == parent.id
            assert child.child_index == c_idx


def test_metadata_deep_copy_prevents_cross_chunk_mutation(sample_parsed_doc: ParsedDoc) -> None:
    chunker = ParentChildChunker(parent_size=80, child_size=35, child_overlap=10)
    parents, children = chunker.chunk(sample_parsed_doc)

    children[0].metadata.extra["discipline"] = "MUTATED"
    children[0].page_numbers.append(999)
    assert sample_parsed_doc.metadata.extra["discipline"] == "HVAC"
    assert parents[0].metadata.extra["discipline"] == "HVAC"
    assert 999 not in parents[0].page_numbers


def test_empty_document_returns_empty_lists(base_metadata: DocumentMetadata) -> None:
    doc = ParsedDoc(
        document_id="doc-empty",
        content_hash="h",
        markdown="  \n ",
        sections=[ParsedSection(text="   ")],
        metadata=base_metadata,
    )
    assert ParentChildChunker().chunk(doc) == ([], [])


def test_markdown_table_splitting_repeats_headers_in_every_parent_and_child(
    base_metadata: DocumentMetadata,
) -> None:
    header = "| Unit ID | Building | Airflow (m3/h) | Pressure Drop (Pa) |"
    sep = "| :--- | :--- | :--- | :--- |"
    rows = [
        f"| LBK-{i:02d} | Blok Noord | {10000 + i * 250} | {450 + i * 10} |" for i in range(1, 25)
    ]
    table_md = "\n".join([header, sep, *rows])
    doc = ParsedDoc(
        document_id="doc-table",
        content_hash="hash-tbl",
        markdown=table_md,
        sections=[
            ParsedSection(
                text=table_md,
                page_numbers=[4],
                section_header="Toestellenlijst Luchtgroepen",
                block_type="table",
            )
        ],
        metadata=base_metadata,
    )
    chunker = ParentChildChunker(parent_size=220, parent_overlap=0, child_size=90, child_overlap=15)
    parents, children = chunker.chunk(doc)

    assert len(parents) >= 2
    assert len(children) >= 4
    for p in parents:
        assert p.text.splitlines()[:2] == [header, sep]
    for c in children:
        assert c.text.splitlines()[:2] == [header, sep]
