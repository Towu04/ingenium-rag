"""End-to-end integration tests: ParsedDoc JSON -> IndexingPipeline -> RAGPipeline -> FastAPI -> Eval."""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api.app import app, get_pipeline
from chunking import ParentChildChunker
from domain import (
    ChildChunk,
    DocumentMetadata,
    EmbeddedChildBatch,
    EncodedQuery,
    ParsedDoc,
    ParsedSection,
    SearchResult,
)
from evaluation import run_evaluation
from generation.generator import ExtractiveStubLLM, GroundedGenerator
from pipeline import IndexingPipeline, RAGPipeline
from retrieval import Searcher
from storage import LocalHybridRepository


def _hash_tokens(text: str) -> tuple[np.ndarray, dict[int, float]]:
    vec = np.zeros(1024, dtype=np.float32)
    sparse: dict[int, float] = {}
    for raw_tok in text.lower().split():
        tok = raw_tok.strip(".,:;()[]|")
        if len(tok) < 2:
            continue
        digest = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        vec[digest % 1024] += 1.0
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
                dense_vecs=np.empty((0, 1024), dtype=np.float32), sparse_weights=[]
            )
        dense_rows, sparse_rows = zip(*(_hash_tokens(c.embedding_text or c.text) for c in children))
        return EmbeddedChildBatch(
            dense_vecs=np.vstack(dense_rows), sparse_weights=list(sparse_rows)
        )

    def embed_query(self, query: str) -> EncodedQuery:
        d_vec, s_dict = _hash_tokens(query)
        return EncodedQuery(dense_vec=d_vec, sparse_weights=s_dict)


class _FakeReranker:
    def rerank(
        self, query: str, candidates: Sequence[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        q_tokens = {t.strip(".,:;?!") for t in query.lower().split() if len(t) >= 2}
        scored = [
            SearchResult(
                parent_chunk=item.parent_chunk,
                matched_child_id=item.matched_child_id,
                matched_child_text=item.matched_child_text,
                retrieval_score=(
                    item.retrieval_score if item.retrieval_score is not None else item.score
                ),
                score=float(len(q_tokens & set(item.parent_chunk.text.lower().split())))
                + item.score,
            )
            for item in candidates
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def _write_parsed_doc_fixtures(base_dir: Path) -> Path:
    json_dir = base_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    fire_text = (
        "Op iedere bouwlaag wordt de verbinding tussen het compartiment en het "
        "trappenhuis verzekerd door een zelfsluitende brandwerende deur EI1 30."
    )
    doc_fire = ParsedDoc(
        document_id="fire-001",
        content_hash="hash-fire-v1",
        markdown=fire_text,
        sections=[
            ParsedSection(
                text=fire_text,
                page_numbers=[5],
                section_header="4.2.2.3 Brandwerende deuren",
                block_type="text",
            )
        ],
        metadata=DocumentMetadata(
            source_path="General/05_Uitvoering/Brandpreventieverslag.pdf",
            sharepoint_url="https://company.sharepoint.com/sites/12345001-INT/Brandpreventieverslag.pdf",
            project="12345.001",
            building="Blok A - Blok C - West Wing",
            phase="Uitvoering",
        ),
    )

    hvac_table = (
        "| Unit ID | Luchtdebiet (m3/h) | Externe druk (Pa) |\n"
        "| :--- | :--- | :--- |\n"
        "| EG1.5 | 1500 | 1008 |\n"
        "| LBK-01 | 12500 | 650 |"
    )
    doc_hvac = ParsedDoc(
        document_id="hvac-002",
        content_hash="hash-hvac-v1",
        markdown=hvac_table,
        sections=[
            ParsedSection(
                text=hvac_table,
                page_numbers=[],
                section_header="Luchtgroepen",
                block_type="table",
                sheet_name="Luchtgroepen",
            )
        ],
        metadata=DocumentMetadata(
            source_path="General/02_Aanbesteding/Toestellenlijst_HVAC.xlsx",
            project="12345.001",
            building="Blok A - Operatiekwartier",
            phase="Aanbesteding",
        ),
    )

    (json_dir / "fire-001.json").write_text(doc_fire.model_dump_json(indent=2), encoding="utf-8")
    (json_dir / "hvac-002.json").write_text(doc_hvac.model_dump_json(indent=2), encoding="utf-8")
    return base_dir


def test_end_to_end_pipeline_and_evaluation(tmp_path: Path) -> None:
    processed_dir = _write_parsed_doc_fixtures(tmp_path / "processed")
    embedder = _FakeEmbedder()
    chunker = ParentChildChunker(parent_size=200, child_size=80, child_overlap=10)

    with LocalHybridRepository(path=":memory:") as repo:
        indexer = IndexingPipeline(repository=repo, embedder=embedder, chunker=chunker)

        stats1 = indexer.ingest_processed_dir(processed_dir, use_manifest=False)
        assert (stats1.total_docs, stats1.indexed_docs, stats1.skipped_unchanged) == (2, 2, 0)

        # Second pass skips unchanged documents
        stats2 = indexer.ingest_processed_dir(processed_dir, use_manifest=False)
        assert (stats2.indexed_docs, stats2.skipped_unchanged) == (0, 2)

        rag = RAGPipeline(
            searcher=Searcher(embedder=embedder, repository=repo),
            reranker=_FakeReranker(),
            generator=GroundedGenerator(llm=ExtractiveStubLLM()),
            retrieve_k=10,
            rerank_k=3,
        )

        ans = rag.run(
            "Welke brandwerende deur is vereist naar het trappenhuis?",
            project="12345.001",
            building="Blok C",
            phase="Uitvoering",
        )
        assert "EI1 30" in ans.text
        assert ans.citations[0].document_id == "fire-001"
        assert ans.citations[0].page_numbers == [5]

        hvac_ans = rag.run(
            "Wat is het luchtdebiet van EG1.5?", building="Blok A", phase="Aanbesteding"
        )
        assert "1500" in hvac_ans.text
        assert hvac_ans.citations[0].document_id == "hvac-002"

        eval_out = run_evaluation(
            rag,
            [
                {
                    "question": "Welke brandwerende deur EI1 30 is vereist?",
                    "building": "Blok C",
                    "expected_sources": ["General/05_Uitvoering/Brandpreventieverslag.pdf"],
                }
            ],
        )
        assert eval_out[0]["citation_precision"] == 1.0
        assert eval_out[0]["source_recall"] == 1.0


def test_fastapi_health_and_query_endpoint(tmp_path: Path) -> None:
    processed_dir = _write_parsed_doc_fixtures(tmp_path / "api_processed")
    embedder = _FakeEmbedder()
    chunker = ParentChildChunker(parent_size=200, child_size=80, child_overlap=10)

    with LocalHybridRepository(path=":memory:") as repo:
        IndexingPipeline(repository=repo, embedder=embedder, chunker=chunker).ingest_processed_dir(
            processed_dir, use_manifest=False
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
                    "project": "12345.001",
                    "building": "Blok C",
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert "EI1 30" in body["answer"]
            assert body["citations"][0]["document_id"] == "fire-001"
        finally:
            app.dependency_overrides.clear()


def test_indexing_pipeline_lazy_embedder_and_length_sorted_batching(
    tmp_path: Path, monkeypatch
) -> None:
    import pipeline as pipeline_module

    class _RecordingEmbedder(_FakeEmbedder):
        def __init__(self) -> None:
            self.calls: list[list[int]] = []

        def embed_children(self, children: Sequence[ChildChunk]) -> EmbeddedChildBatch:
            self.calls.append([len(c.embedding_text or c.text) for c in children])
            return super().embed_children(children)

    processed_dir = _write_parsed_doc_fixtures(tmp_path / "batch_processed")
    rec_embedder = _RecordingEmbedder()
    chunker = ParentChildChunker(parent_size=200, child_size=80, child_overlap=10)

    with LocalHybridRepository(path=":memory:") as repo:
        indexer = IndexingPipeline(
            repository=repo,
            embedder=rec_embedder,
            chunker=chunker,
            embed_batch_buffer_size=256,
        )
        stats1 = indexer.ingest_processed_dir(processed_dir, use_manifest=False)
        assert stats1.indexed_docs == 2
        assert "chunking" in stats1.stage_timings
        assert "embedding_and_indexing" in stats1.stage_timings
        # Both docs batched into a single length-sorted embed_children call
        assert len(rec_embedder.calls) == 1
        assert rec_embedder.calls[0] == sorted(rec_embedder.calls[0])

        # Verify on_progress receives stage milestones and bar events
        events: list[str] = []
        stats_forced = indexer.ingest_processed_dir(
            processed_dir,
            use_manifest=False,
            force=True,
            on_progress=lambda ev, cur, tot, detail: events.append(ev),
        )
        assert stats_forced.indexed_docs == 2
        assert "stage_start" in events and "stage_done" in events
        assert "bar_start" in events and "bar_done" in events

        # Lazy embedder guard: when all docs are unchanged, BGEM3Embedder must never be instantiated
        def _fail_if_loaded(*args, **kwargs):
            raise AssertionError("BGEM3Embedder should not be loaded when all docs are unchanged")

        monkeypatch.setattr(pipeline_module, "BGEM3Embedder", _fail_if_loaded)
        lazy_indexer = IndexingPipeline(repository=repo, chunker=chunker)
        stats2 = lazy_indexer.ingest_processed_dir(processed_dir, use_manifest=False)
        assert (stats2.indexed_docs, stats2.skipped_unchanged) == (0, 2)


def test_retrieve_with_trace_reports_filters_funnel_and_child_text(tmp_path: Path) -> None:
    processed_dir = _write_parsed_doc_fixtures(tmp_path / "trace_processed")
    embedder = _FakeEmbedder()
    chunker = ParentChildChunker(parent_size=200, child_size=80, child_overlap=10)

    with LocalHybridRepository(path=":memory:") as repo:
        IndexingPipeline(repository=repo, embedder=embedder, chunker=chunker).ingest_processed_dir(
            processed_dir, use_manifest=False
        )
        rag = RAGPipeline(
            searcher=Searcher(embedder=embedder, repository=repo),
            reranker=_FakeReranker(),
            generator=GroundedGenerator(llm=ExtractiveStubLLM()),
            retrieve_k=10,
            rerank_k=3,
        )
        trace = rag.retrieve_with_trace(
            "Welke brandwerende deur EI1 30 in Blok C?",
            auto_filter=True,
        )
        assert trace.auto_filters.get("building") == "Blok C"
        assert trace.applied_filters.get("building") == "Blok C"
        assert trace.hybrid_child_hits >= 1
        assert trace.deduplicated_parents >= 1
        assert trace.reranked_kept == len(trace.results)
        assert trace.reranked_dropped == 0
        assert trace.results[0].matched_child_text is not None
        assert "EI1 30" in trace.results[0].matched_child_text


def test_openai_compatible_llm_normalizes_v1_base_url() -> None:
    from generation.generator import OpenAICompatibleLLM

    expected = "http://localhost:11434/v1/chat/completions"
    for url in (
        "http://localhost:11434",
        "http://localhost:11434/",
        "http://localhost:11434/v1",
        "http://localhost:11434/v1/",
    ):
        llm = OpenAICompatibleLLM(base_url=url, model="qwen2.5:14b")
        assert llm.endpoint == expected


def test_fastapi_lifespan_closes_repository(monkeypatch) -> None:
    import api.app as app_module

    closed = {"called": False}

    class _FakeRepo:
        def close(self) -> None:
            closed["called"] = True

    app_module._get_pipeline_and_repo.cache_clear()
    monkeypatch.setattr(app_module, "build_rag_pipeline", lambda: (object(), _FakeRepo()))
    try:
        with TestClient(app_module.app) as client:
            assert client.get("/health").status_code == 200
            app_module.get_pipeline()
        assert closed["called"] is True
    finally:
        app_module._get_pipeline_and_repo.cache_clear()


def test_ingest_sharepoint_indexes_incrementally_per_site(tmp_path: Path, monkeypatch) -> None:
    import pipeline as pipeline_module
    from config import SharePointSiteSpec
    from etl.runner import ETLRunResult

    json_dir = tmp_path / "processed" / "json"
    json_dir.mkdir(parents=True)

    doc1 = ParsedDoc(
        document_id="site1-doc1",
        content_hash="h1",
        markdown="Site 1 specs",
        sections=[
            ParsedSection(
                text="Brandwerende deur EI1 30 in Blok A.",
                page_numbers=[1],
                section_header="1. Deuren",
            )
        ],
        metadata=DocumentMetadata(source_path="S1/Spec.pdf", project="10001-INT"),
    )
    doc2 = ParsedDoc(
        document_id="site2-doc1",
        content_hash="h2",
        markdown="Site 2 specs",
        sections=[
            ParsedSection(
                text="Luchtgroep debiet 4000 m3/h in Blok B.",
                page_numbers=[1],
                section_header="2. HVAC",
            )
        ],
        metadata=DocumentMetadata(source_path="S2/HVAC.pdf", project="10002-INT"),
    )
    p1 = json_dir / "site1-doc1.json"
    p2 = json_dir / "site2-doc1.json"
    p1.write_text(doc1.model_dump_json(), encoding="utf-8")
    p2.write_text(doc2.model_dump_json(), encoding="utf-8")

    class _FakeSP:
        def for_site(self, spec: SharePointSiteSpec):
            return spec

    site1_indexed_before_site2 = {"value": False}
    embedder = _FakeEmbedder()
    chunker = ParentChildChunker(parent_size=200, child_size=80, child_overlap=10)

    with LocalHybridRepository(path=":memory:") as repo:

        def _fake_run_sharepoint(self, *, connector, **kwargs) -> ETLRunResult:
            if connector.site_path == "/sites/10001-INT":
                return ETLRunResult(discovered_files=1, processed_files=1, output_json_paths=[p1])
            # When site 2 begins ETL, site 1 must already be indexed in Qdrant!
            site1_indexed_before_site2["value"] = not repo.needs_indexing("site1-doc1", "h1")
            return ETLRunResult(discovered_files=1, processed_files=1, output_json_paths=[p2])

        monkeypatch.setattr(
            pipeline_module.SharePointConnector,
            "from_env",
            classmethod(lambda cls, **kw: _FakeSP()),
        )
        monkeypatch.setattr(pipeline_module.ETLRunner, "run_sharepoint", _fake_run_sharepoint)

        indexer = IndexingPipeline(repository=repo, embedder=embedder, chunker=chunker)
        stats = indexer.ingest_sharepoint(
            sites=[
                SharePointSiteSpec(
                    hostname="tenant.sharepoint.com",
                    site_path="/sites/10001-INT",
                    project_code="10001-INT",
                ),
                SharePointSiteSpec(
                    hostname="tenant.sharepoint.com",
                    site_path="/sites/10002-INT",
                    project_code="10002-INT",
                ),
            ],
            processed_dir=tmp_path / "processed",
        )
        assert site1_indexed_before_site2["value"] is True
        assert stats.indexed_docs == 2
        assert repo.collection_counts()["documents"] == 2


def test_indexing_pipeline_deduplicates_identical_content_hash_across_paths() -> None:
    doc_a = ParsedDoc(
        document_id="copy-a",
        content_hash="same-sha256-xyz",
        markdown="Identical spec",
        sections=[
            ParsedSection(
                text="Brandwerende deur EI1 30 volgens bestek HVAC.",
                page_numbers=[1],
                section_header="1. Deuren",
            )
        ],
        metadata=DocumentMetadata(source_path="04_Prijsvraag/HVAC/Bestek.pdf", project="10001"),
    )
    doc_b = ParsedDoc(
        document_id="copy-b",
        content_hash="same-sha256-xyz",
        markdown="Identical spec",
        sections=[
            ParsedSection(
                text="Brandwerende deur EI1 30 volgens bestek HVAC.",
                page_numbers=[1],
                section_header="1. Deuren",
            )
        ],
        metadata=DocumentMetadata(
            source_path="04_Prijsvraag/publicatie/Bestek.pdf", project="10001"
        ),
    )
    embedder = _FakeEmbedder()
    chunker = ParentChildChunker(parent_size=200, child_size=80, child_overlap=10)

    with LocalHybridRepository(path=":memory:") as repo:
        indexer = IndexingPipeline(repository=repo, embedder=embedder, chunker=chunker)
        stats = indexer.index_documents([doc_a, doc_b])
        assert stats.indexed_docs == 1
        assert stats.skipped_duplicate == 1
        assert repo.collection_counts()["documents"] == 1

        # A subsequent run with yet another copy of the same hash also skips via Qdrant check
        doc_c = doc_b.model_copy(update={"document_id": "copy-c"})
        stats2 = indexer.index_documents([doc_c])
        assert stats2.indexed_docs == 0
        assert stats2.skipped_duplicate == 1


def test_rag_pipeline_low_memory_releases_embedder_before_loading_reranker(tmp_path: Path) -> None:
    processed_dir = _write_parsed_doc_fixtures(tmp_path / "low_mem_processed")
    embedder = _FakeEmbedder()
    chunker = ParentChildChunker(parent_size=200, child_size=80, child_overlap=10)
    events: list[str] = []

    with LocalHybridRepository(path=":memory:") as repo:
        IndexingPipeline(repository=repo, embedder=embedder, chunker=chunker).ingest_processed_dir(
            processed_dir, use_manifest=False
        )
        searcher = Searcher(embedder=embedder, repository=repo)

        def _make_reranker() -> _FakeReranker:
            # Embedder must already be released before the reranker is constructed!
            assert searcher.embedder is None
            events.append("reranker_loaded_after_embedder_freed")
            return _FakeReranker()

        rag = RAGPipeline(
            searcher=searcher,
            reranker=None,
            reranker_factory=_make_reranker,
            low_memory=True,
            generator=GroundedGenerator(llm=ExtractiveStubLLM()),
        )
        assert rag.reranker is None
        trace = rag.retrieve_with_trace("brandwerende deur EI1 30")
        assert events == ["reranker_loaded_after_embedder_freed"]
        assert len(trace.results) >= 1
