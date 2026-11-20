"""Unit & real-corpus smoke tests for the ETL-to-Domain adapter (`processing/etl_adapter.py`)."""

from pathlib import Path

import pytest

from etl.schemas import (
    ContentBlock,
    DocCategory,
    DocumentMetadata as EtlMetadata,
    ExtractedDocument,
    ExtractionMethod,
)
from processing.etl_adapter import (
    adapt_extracted_document,
    load_corpus_from_etl_dir,
    load_parsed_doc_from_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def _sample_etl_doc(**meta_overrides) -> ExtractedDocument:
    defaults = dict(
        doc_id="1313846fb39c5f58",
        file_name="Bestek_HVAC.pdf",
        file_extension=".pdf",
        file_hash_sha256="abc123",
        sharepoint_url="https://ingeniumnv.sharepoint.com/sites/06038004-INT/Shared%20Documents/General/05_Uitvoering/Bestek_HVAC.pdf",
        relative_folder_path="General/05_Uitvoering",
        project_id="UNKNOWN_PROJECT",
        project_name="AZ Sint-Jan Brugge",
        building="Blok P - Blok L - Spoed",
        phase="Uitvoering (Execution)",
        doc_category=DocCategory.SPECIFICATION,
        extraction_method=ExtractionMethod.PYMUPDF_DIGITAL,
    )
    defaults.update(meta_overrides)
    return ExtractedDocument(
        metadata=EtlMetadata(**defaults),
        blocks=[
            ContentBlock(
                block_id="pdf-blk-0001",
                page_number=1,
                section_heading="3.1 Ventilatie",
                block_type="text",
                markdown_content="Eerste paragraaf over luchtgroepen.",
            ),
            ContentBlock(
                block_id="pdf-blk-0002",
                page_number=2,
                section_heading="3.1 Ventilatie",
                block_type="text",
                markdown_content="Vervolg op pagina twee.",
            ),
            ContentBlock(
                block_id="pdf-blk-0003",
                page_number=2,
                section_heading="3.1 Ventilatie",
                block_type="table",
                markdown_content="| Unit | Debiet |\n| :--- | :--- |\n| LBK-01 | 15000 |",
                table_headers=["Unit", "Debiet"],
            ),
        ],
    )


def test_adapt_normalizes_metadata_coalesces_prose_and_keeps_table_atomic() -> None:
    parsed = adapt_extracted_document(_sample_etl_doc())

    assert parsed.document_id == "1313846fb39c5f58"
    assert parsed.content_hash == "abc123"
    assert parsed.metadata.source_path == "General/05_Uitvoering/Bestek_HVAC.pdf"
    assert parsed.metadata.project == "06038.004"  # Resolved from SharePoint site URL
    assert parsed.metadata.phase == "Uitvoering"  # Stripped parenthetical translation
    assert parsed.metadata.buildings == ["Blok P", "Blok L", "Spoed"]
    assert parsed.metadata.building == "Blok P - Blok L - Spoed"
    assert parsed.metadata.extra["doc_category"] == "specification"

    # Two consecutive text blocks with same heading coalesce into 1; table remains separate
    assert len(parsed.sections) == 2
    assert parsed.sections[0].page_numbers == [1, 2]
    assert parsed.sections[0].section_header == "3.1 Ventilatie"
    assert "Vervolg op pagina twee." in parsed.sections[0].text
    assert parsed.sections[1].text.startswith("| Unit | Debiet |")
    assert "<!--" not in parsed.markdown


def test_adapt_sanitizes_legacy_bugs_and_quarantines_noisy_callouts() -> None:
    noisy_callout = "\n".join(
        ["- 1", "- f", "- 2", "- I", "- ~n", "- 3", "- {", "- 4", "- }", "- %3"]
    )
    clean_callout = "\n".join(
        ["- LBK-01 Luchtgroep Operatiekwartier", "- Debiet 12500 m3/h", "- Filterklasse HEPA H14"]
    )
    doc = ExtractedDocument(
        metadata=EtlMetadata(
            doc_id="diag-01",
            file_name="Schema EB Honeywell.pdf",
            file_extension=".pdf",
            file_hash_sha256="hash-diag",
            relative_folder_path="General/00_Info",
            project_id="UNKNOWN_PROJECT",
            building="Blok Peil",
            phase="Definitief Ontwerp (Final Design)",  # False positive from "DO" in diagram
            is_hvac_diagram=True,
        ),
        blocks=[
            ContentBlock(
                block_id="b1",
                page_number=1,
                section_heading="8000. Brugge",
                block_type="text",
                markdown_content="Brandpreventieverslag tekst.",
            ),
            ContentBlock(
                block_id="b2",
                page_number=2,
                section_heading="HVAC Diagram Callouts",
                block_type="diagram_callout",
                markdown_content=noisy_callout,
            ),
            ContentBlock(
                block_id="b3",
                page_number=3,
                section_heading="HVAC Diagram Callouts",
                block_type="diagram_callout",
                markdown_content=clean_callout,
            ),
        ],
    )

    parsed_filtered = adapt_extracted_document(doc, drop_noisy_callouts=True)
    assert parsed_filtered.metadata.project is None
    assert parsed_filtered.metadata.building is None
    assert parsed_filtered.metadata.buildings == []
    assert parsed_filtered.metadata.phase is None
    assert parsed_filtered.metadata.extra["demoted_raw_phase"] == "Definitief Ontwerp (Final Design)"
    assert parsed_filtered.metadata.extra["dropped_noisy_callouts"] == 1
    assert parsed_filtered.sections[0].section_header is None  # "8000. Brugge" stripped
    assert len(parsed_filtered.sections) == 2  # text + clean callout kept, noisy callout dropped

    # When drop_noisy_callouts=False, all 3 blocks are retained
    parsed_all = adapt_extracted_document(doc, drop_noisy_callouts=False)
    assert len(parsed_all.sections) == 3
    assert parsed_all.metadata.extra["dropped_noisy_callouts"] == 0


def test_load_parsed_doc_skips_empty_orphan_json(tmp_path: Path) -> None:
    empty_doc = ExtractedDocument(
        metadata=EtlMetadata(
            doc_id="orphan-1",
            file_name="Aborted.pdf",
            file_extension=".pdf",
            file_hash_sha256="deadbeef",
            relative_folder_path="General",
        ),
        blocks=[],
    )
    jp = tmp_path / "orphan-1.json"
    jp.write_text(empty_doc.model_dump_json(), encoding="utf-8")
    assert load_parsed_doc_from_json(jp) is None


@pytest.mark.skipif(
    not (REAL_PROCESSED_DIR / "manifest.json").exists(),
    reason="Real data/processed/ corpus not present in this environment",
)
def test_real_corpus_smoke_load_from_data_processed() -> None:
    """Smoke test loading all real ETL JSON files in data/processed/ via manifest."""
    docs = load_corpus_from_etl_dir(REAL_PROCESSED_DIR, dedupe_by_hash=True)
    assert len(docs) >= 20

    # Every loaded document must have resolved project 06038.004 from its SharePoint URL
    # and zero UNKNOWN_* sentinels or bogus postal-code headings.
    for doc in docs:
        assert doc.metadata.project == "06038.004"
        assert doc.metadata.building != "Blok Peil"
        assert doc.metadata.building != "UNKNOWN_BUILDING"
        assert doc.metadata.phase != "UNKNOWN_PHASE"
        for sec in doc.sections:
            assert sec.section_header != "8000. Brugge"

    # Multi-building medical gases schema should have split buildings list containing "Blok L"
    med_gas = next(d for d in docs if d.document_id == "70e1a1e206a814c8")
    assert "Blok L" in med_gas.metadata.buildings
    assert "Spoed" in med_gas.metadata.buildings
