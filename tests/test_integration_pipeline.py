"""End-to-end integration tests for ETL -> Adapter -> IndexingPipeline -> RAGPipeline -> FastAPI.

Uses `LocalHybridRepository(path=":memory:")` and deterministic bag-of-words fake embedder/reranker
so the entire pipeline executes in seconds without downloading or loading GPU weights.
"""

from collections.abc import Sequence
import hashlib
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np

from api.app import app, get_pipeline
from domain import ChildChunk, SearchResult
from etl.schemas import (
    ContentBlock,
    DocCategory,
    DocumentMetadata as EtlMetadata,
    ExtractedDocument,
    ExtractionMethod,
)
from evaluation.run_eval import run_evaluation
from generation.generator import ExtractiveStubLLM, GroundedGenerator
from pipelines.indexing_pipeline import IndexingPipeline
from pipelines.rag_pipeline import RAGPipeline
from processing.chunker import ParentChildChunker
from retrieval.query_filter import extract_metadata_filters
from retrieval.searcher import Searcher
from storage.embedder import EmbeddedChildBatch, EncodedQuery
from storage.repository import LocalHybridRepository


# ---------------------------------------------------------------------------
# Deterministic Lightweight Fakes (Fast, Offline, Zero GPU)
# ---------------------------------------------------------------------------


def _hash_tokens(text: str) -> tuple[np.ndarray, dict[int, float]]:
    vec = np.zeros(1024, dtype=np.float32)
    sparse: dict[int, float] = {}
    for raw_tok in text.lower().split():
        tok = raw_tok.strip(".,:;()[]|")
        if len(tok) < 2:
            continue
        digest = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        dim = digest % 1024
        vec[dim] += 1.0
        tok_id = digest % 50_000
        sparse[tok_id] = sparse.get(tok_id, 0.0) + 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec, sparse


class _FakeEmbedder:
    def embed_children(self, children: Sequence[ChildChunk]) -> EmbeddedChildBatch:
        if not children:
            return EmbeddedChildBatch(
                dense_vecs=np.empty((0, 1024), dtype=np.float32),
                sparse_weights=[],
            )
        dense_rows: list[np.ndarray] = []
        sparse_rows: list[dict[int, float]] = []
        for child in children:
            d_vec, s_dict = _hash_tokens(child.embedding_text or child.text)
            dense_rows.append(d_vec)
            sparse_rows.append(s_dict)
        return EmbeddedChildBatch(dense_vecs=np.vstack(dense_rows), sparse_weights=sparse_rows)

    def embed_query(self, query: str) -> EncodedQuery:
        d_vec, s_dict = _hash_tokens(query)
        return EncodedQuery(dense_vec=d_vec, sparse_weights=s_dict)


class _FakeReranker:
    def rerank(
        self, query: str, candidates: Sequence[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        q_tokens = {t.strip(".,:;?!") for t in query.lower().split() if len(t) >= 2}
        scored = []
        for item in candidates:
            p_tokens = set(item.parent_chunk.text.lower().split())
            overlap = len(q_tokens & p_tokens)
            scored.append(
                SearchResult(
                    parent_chunk=item.parent_chunk,
                    matched_child_id=item.matched_child_id,
                    score=float(overlap) + item.score,
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def _write_etl_fixture_dir(base_dir: Path) -> Path:
    json_dir = base_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    doc_fire = ExtractedDocument(
        metadata=EtlMetadata(
            doc_id="fire-001",
            file_name="Brandpreventieverslag.pdf",
            file_extension=".pdf",
            file_hash_sha256="hash-fire-v1",
            sharepoint_url="https://ingeniumnv.sharepoint.com/sites/06038004-INT/Shared%20Documents/General/05_Uitvoering/Brandpreventieverslag.pdf",
            relative_folder_path="General/05_Uitvoering",
            project_id="UNKNOWN_PROJECT",
            building="Blok P - Blok L - Spoed",
            phase="Uitvoering (Execution)",
            doc_category=DocCategory.REPORT_MEETING,
            extraction_method=ExtractionMethod.PYMUPDF_DIGITAL,
        ),
        blocks=[
            ContentBlock(
                block_id="pdf-blk-0001",
                page_number=5,
                section_heading="4.2.2.3 Brandwerende deuren",
                block_type="text",
                markdown_content=(
                    "Op iedere bouwlaag wordt de verbinding tussen het compartiment en het "
                    "trappenhuis verzekerd door een zelfsluitende brandwerende deur EI1 30."
                ),
            ),
        ],
    )

    doc_hvac = ExtractedDocument(
        metadata=EtlMetadata(
            doc_id="hvac-002",
            file_name="Toestellenlijst_HVAC.xlsx",
            file_extension=".xlsx",
            file_hash_sha256="hash-hvac-v1",
            sharepoint_url="https://ingeniumnv.sharepoint.com/sites/06038004-INT/Shared%20Documents/General/02_Aanbesteding/Toestellenlijst_HVAC.xlsx",
            relative_folder_path="General/02_Aanbesteding",
            project_id="06038.004",
            building="Blok A - Operatiekwartier",
            phase="Aanbesteding (Tender / Specifications)",
            doc_category=DocCategory.SCHEDULE_TABLE,
            extraction_method=ExtractionMethod.OPENPYXL_TABLE,
        ),
        blocks=[
            ContentBlock(
                block_id="xlsx-blk-0001",
                page_number=None,
                section_heading="Luchtgroepen",
                sheet_name="Luchtgroepen",
                block_type="table",
                markdown_content=(
                    "| Unit ID | Luchtdebiet (m3/h) | Externe druk (Pa) |\n"
                    "| :--- | :--- | :--- |\n"
                    "| EG1.5 | 1500 | 1008 |\n"
                    "| LBK-01 | 12500 | 650 |"
                ),
                table_headers=["Unit ID", "Luchtdebiet (m3/h)", "Externe druk (Pa)"],
            ),
        ],
    )

    (json_dir / "fire-001.json").write_text(doc_fire.model_dump_json(indent=2), encoding="utf-8")
    (json_dir / "hvac-002.json").write_text(doc_hvac.model_dump_json(indent=2), encoding="utf-8")
    return base_dir


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


def test_query_filter_extracts_natural_and_explicit_filters() -> None:
    q1 = "Wat is het debiet voor project 06038.004 in Blok L tijdens uitvoering?"
    f1 = extract_metadata_filters(q1)
    assert f1["project"] == "06038.004"
    assert f1["building"] == "Blok L"
    assert f1["phase"] == "Uitvoering"

    q2 = "project:PRJ-2021-104 building:Blok A phase:Aanbesteding luchtgroepen"
    f2 = extract_metadata_filters(q2)
    assert f2 == {"project": "PRJ-2021-104", "building": "Blok A", "phase": "Aanbesteding"}


def test_end_to_end_ingestion_incremental_sync_and_filtered_rag(tmp_path: Path) -> None:
    etl_dir = _write_etl_fixture_dir(tmp_path / "processed")
    repo = LocalHybridRepository(path=":memory:")
    embedder = _FakeEmbedder()
    chunker = ParentChildChunker(
        parent_size=200,
        child_size=80,
        child_overlap=10,
    )

    indexer = IndexingPipeline(repository=repo, embedder=embedder, chunker=chunker)

    # First run indexes both documents
    stats1 = indexer.ingest_from_etl_dir(etl_dir, use_manifest=False)
    assert stats1.total_docs == 2
    assert stats1.indexed_docs == 2
    assert stats1.skipped_unchanged == 0
    assert stats1.total_parents == 2
    assert stats1.total_children >= 2

    # Second run skips both documents via needs_indexing(document_id, content_hash)
    stats2 = indexer.ingest_from_etl_dir(etl_dir, use_manifest=False)
    assert stats2.indexed_docs == 0
    assert stats2.skipped_unchanged == 2

    # Query via RAGPipeline with multi-building filter ("Blok L" inside "Blok P - Blok L - Spoed")
    rag = RAGPipeline(
        searcher=Searcher(embedder=embedder, repository=repo),
        reranker=_FakeReranker(),
        generator=GroundedGenerator(llm=ExtractiveStubLLM()),
        retrieve_k=10,
        rerank_k=3,
    )

    answer = rag.run(
        "Welke brandwerende deur is vereist naar het trappenhuis?",
        project="06038.004",
        building="Blok L",
        phase="Uitvoering",
    )
    assert "EI1 30" in answer.text
    assert len(answer.citations) == 1
    assert answer.citations[0].document_id == "fire-001"
    assert answer.citations[0].page_numbers == [5]
    assert answer.citations[0].section_header == "4.2.2.3 Brandwerende deuren"

    # Query HVAC table filtered to Blok A + Aanbesteding
    hvac_ans = rag.run(
        "Wat is het luchtdebiet van EG1.5?",
        building="Blok A",
        phase="Aanbesteding",
    )
    assert "1500" in hvac_ans.text
    assert hvac_ans.citations[0].document_id == "hvac-002"

    # Evaluation harness smoke test
    eval_out = run_evaluation(
        rag,
        [
            {
                "question": "Welke brandwerende deur EI1 30 is vereist?",
                "building": "Blok L",
                "expected_sources": ["General/05_Uitvoering/Brandpreventieverslag.pdf"],
            }
        ],
    )
    assert len(eval_out) == 1
    assert eval_out[0]["citation_precision"] == 1.0


def test_fastapi_health_and_query_endpoint(tmp_path: Path) -> None:
    etl_dir = _write_etl_fixture_dir(tmp_path / "api_processed")
    repo = LocalHybridRepository(path=":memory:")
    embedder = _FakeEmbedder()
    chunker = ParentChildChunker(
        parent_size=200,
        child_size=80,
        child_overlap=10,
    )
    IndexingPipeline(repository=repo, embedder=embedder, chunker=chunker).ingest_from_etl_dir(
        etl_dir, use_manifest=False
    )

    rag = RAGPipeline(
        searcher=Searcher(embedder=embedder, repository=repo),
        reranker=_FakeReranker(),
        generator=GroundedGenerator(llm=ExtractiveStubLLM()),
    )
    app.dependency_overrides[get_pipeline] = lambda: rag
    try:
        client = TestClient(app)
        assert client.get("/health").json() == {"status": "ok"}

        resp = client.post(
            "/v1/query",
            json={
                "query": "brandwerende deur EI1 30",
                "project": "06038.004",
                "building": "Blok L",
                "top_k": 3,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "EI1 30" in body["answer"]
        assert body["citations"][0]["document_id"] == "fire-001"
    finally:
        app.dependency_overrides.clear()
