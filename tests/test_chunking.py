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
        source_path="Campus_Noord/Blok_Noord/Uitvoering/HVAC_Bestek.pdf",
        sharepoint_url="https://company.sharepoint.com/sites/campus-noord/HVAC_Bestek.pdf",
        project="PRJ-2026-CN",
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


def test_pack_sections_attaches_orphan_headings_and_packs_short_prose(
    base_metadata: DocumentMetadata,
) -> None:
    table_md = "| Post | Bedrag |\n| :--- | :--- |\n| HVAC | 12000 |"
    doc = ParsedDoc(
        document_id="doc-pack",
        content_hash="h-pack",
        markdown="",
        sections=[
            ParsedSection(text="## Raming (EUR excl. BTW)", page_numbers=[1], block_type="text"),
            ParsedSection(
                text=table_md,
                page_numbers=[2],
                section_header="Budget",
                block_type="table",
            ),
            ParsedSection(
                text="Korte alinea een over brandkleppen.",
                page_numbers=[3],
                section_header="2.1 Brandkleppen",
                block_type="text",
            ),
            ParsedSection(
                text="Korte alinea twee over rookafvoer.",
                page_numbers=[4],
                section_header="2.2 Rookafvoer",
                block_type="text",
            ),
        ],
        metadata=base_metadata,
    )
    chunker = ParentChildChunker(parent_size=400, child_size=150, child_overlap=20)
    parents, _ = chunker.chunk(doc)

    # Orphan heading attached to table header (no tiny orphan parent) + two short prose sections packed
    assert len(parents) == 2
    assert parents[0].text == table_md
    assert "Raming (EUR excl. BTW)" in (parents[0].section_header or "")
    assert parents[0].page_numbers == [1, 2]
    assert "brandkleppen" in parents[1].text and "rookafvoer" in parents[1].text
    assert parents[1].page_numbers == [3, 4]


def test_oversized_single_table_row_is_subsplit_with_repeated_headers(
    base_metadata: DocumentMetadata,
) -> None:
    header = "| Code | Omschrijving |"
    sep = "| :--- | :--- |"
    long_desc = " ".join(
        [f"Technische vereiste {i} voor luchtbehandelingskast en kanaalwerk." for i in range(40)]
    )
    table_md = f"{header}\n{sep}\n| ART-01 | {long_desc} |"
    doc = ParsedDoc(
        document_id="doc-bigrow",
        content_hash="h-bigrow",
        markdown=table_md,
        sections=[ParsedSection(text=table_md, page_numbers=[1], block_type="table")],
        metadata=base_metadata,
    )
    chunker = ParentChildChunker(parent_size=200, child_size=90, child_overlap=15)
    parents, children = chunker.chunk(doc)

    assert len(parents) >= 2
    assert len(children) >= 4
    for c in children:
        assert c.text.splitlines()[:2] == [header, sep]
        assert chunker._token_len(c.text) <= 95


def test_embedding_text_includes_file_stem_and_omits_excessive_buildings() -> None:
    meta_many = DocumentMetadata(
        source_path="General/05_Uitvoering/Campus_Overzicht_HVAC.pdf",
        project="12345.001",
        building="Blok A - Blok B - Blok C - Zone Oost",
        phase="Uitvoering",
    )
    doc = ParsedDoc(
        document_id="doc-many-b",
        content_hash="h-mb",
        markdown="Luchtdebiet test.",
        sections=[ParsedSection(text="Luchtdebiet test.", section_header="1. Overzicht")],
        metadata=meta_many,
    )
    _, children = ParentChildChunker().chunk(doc)
    emb_text = children[0].embedding_text or ""
    assert "File: Campus_Overzicht_HVAC" in emb_text
    assert "Project: 12345.001" in emb_text
    assert "Building:" not in emb_text


def test_wide_markdown_table_omits_child_header_repetition_to_prevent_explosion(
    base_metadata: DocumentMetadata,
) -> None:
    cols = [f"M{i:02d}_EUR" for i in range(1, 27)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join([":---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(f"{r * 10 + c}" for c in range(1, 27)) + " |" for r in range(1, 41)]
    table_md = "\n".join([header, sep, *rows])
    doc = ParsedDoc(
        document_id="doc-wide-xlsx",
        content_hash="h-wide",
        markdown=table_md,
        sections=[
            ParsedSection(
                text=table_md,
                page_numbers=[],
                section_header="Posten (Rows 3-42)",
                block_type="table",
                sheet_name="Posten",
            )
        ],
        metadata=base_metadata,
    )
    chunker = ParentChildChunker(
        parent_size=1000, parent_overlap=0, child_size=250, child_overlap=40
    )
    parents, children = chunker.chunk(doc)

    # Header (~200+ tokens) still fits comfortably in parent chunks (<= 400 tokens)
    assert len(parents) >= 2
    for p in parents:
        assert p.text.splitlines()[:2] == [header, sep]

    # Child chunks skip repeating the wide header so rows pack cleanly instead of exploding
    assert len(children) < 30
    assert any(not c.text.startswith(header) for c in children)


def test_max_child_chunks_per_doc_truncates_cleanly_at_parent_boundary(
    base_metadata: DocumentMetadata, caplog
) -> None:
    import logging

    sections = [
        ParsedSection(
            text=("Brandwerende deuren EI1 30 moeten zelfsluitend zijn volgens het bestek. " * 25),
            page_numbers=[i],
            section_header=f"{i}. Afdeling {i}",
        )
        for i in range(1, 20)
    ]
    doc = ParsedDoc(
        document_id="doc-capped",
        content_hash="h-cap",
        markdown="",
        sections=sections,
        metadata=base_metadata,
    )
    chunker = ParentChildChunker(
        parent_size=250, child_size=80, child_overlap=10, max_child_chunks_per_doc=12
    )
    with caplog.at_level(logging.WARNING):
        parents, children = chunker.chunk(doc)

    assert len(parents) >= 1
    assert len(children) <= 12
    parent_ids = {p.id for p in parents}
    child_ids = {c.id for c in children}
    assert all(c.parent_id in parent_ids for c in children)
    assert all(cid in child_ids for p in parents for cid in p.child_ids)
    assert any("max_child_chunks_per_doc=12" in rec.message for rec in caplog.records)


def test_fast_approx_token_chunking_stays_within_embedding_max_length(
    base_metadata: DocumentMetadata,
) -> None:
    prose = (
        "De luchtgroep LBK-02 levert een nominaal luchtdebiet van 18.500 m3/h met "
        "warmteterugwinning (rendement minimaal 78%) en voldoet aan NBN EN 13779. " * 30
    )
    table = "\n".join(
        [
            "| Unit | Zone | Debiet (m3/h) | Drukval (Pa) |",
            "| :--- | :--- | :--- | :--- |",
            *[
                f"| LBK-{i:02d} | Blok A | {10000 + i * 500} | {350 + i * 10} |"
                for i in range(1, 30)
            ],
        ]
    )
    doc = ParsedDoc(
        document_id="doc-fast-tok",
        content_hash="h-fast",
        markdown="",
        sections=[
            ParsedSection(text=prose, page_numbers=[1], section_header="1. Ventilatie"),
            ParsedSection(
                text=table,
                page_numbers=[2],
                section_header="2. Toestellenlijst",
                block_type="table",
            ),
        ],
        metadata=base_metadata,
    )
    chunker = ParentChildChunker(parent_size=1000, child_size=250, child_overlap=40)
    _, children = chunker.chunk(doc)
    # Chunking itself must not require loading the HuggingFace tokenizer
    assert chunker._tokenizer is None
    assert len(children) >= 4
    # Verify real BGE-M3 token lengths stay comfortably under embedding_max_length=320
    real_lens = [chunker._token_len(c.embedding_text or c.text) for c in children]
    assert max(real_lens) <= 320
