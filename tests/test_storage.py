"""Behavioral unit tests for Stage 4 (`storage.LocalHybridRepository`) using in-memory Qdrant."""

import numpy as np
import pytest

from domain import (
    ChildChunk,
    DocumentMetadata,
    EmbeddedChildBatch,
    EncodedQuery,
    ParentChunk,
    ParsedDoc,
)
from storage import LocalHybridRepository


def make_dense_vec(dominant_index: int = 0) -> np.ndarray:
    vec = np.zeros(1024, dtype=np.float32)
    vec[dominant_index] = 1.0
    return vec


def make_bundle(
    doc_id: str = "doc-1",
    content_hash: str = "hash-v1",
    project: str = "Project-A",
    building: str = "Building-1",
    phase: str = "Phase-1",
    num_parents: int = 1,
    children_per_parent: int = 2,
    dominant_dim_offset: int = 0,
) -> tuple[ParsedDoc, list[ParentChunk], list[ChildChunk], EmbeddedChildBatch]:
    meta = DocumentMetadata(
        source_path=f"/docs/{doc_id}.pdf",
        sharepoint_url=f"https://sp.local/{doc_id}",
        project=project,
        building=building,
        phase=phase,
        extra={},
    )
    doc = ParsedDoc(
        document_id=doc_id, content_hash=content_hash, markdown="# Sample", metadata=meta
    )

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    dense_list: list[np.ndarray] = []
    sparse_list: list[dict[int, float]] = []

    for p_idx in range(num_parents):
        parent_id = f"{doc_id}:p{p_idx}"
        c_ids: list[str] = []
        for c_idx in range(children_per_parent):
            child_id = f"{parent_id}:c{c_idx}"
            c_ids.append(child_id)
            children.append(
                ChildChunk(
                    id=child_id,
                    parent_id=parent_id,
                    document_id=doc_id,
                    child_index=c_idx,
                    text=f"Child text {c_idx} of parent {p_idx}",
                    embedding_text=f"Child text {c_idx} of parent {p_idx}",
                    page_numbers=[1],
                    section_header="Intro",
                    metadata=meta,
                )
            )
            dim = dominant_dim_offset + p_idx
            dense_list.append(make_dense_vec(dim))
            sparse_list.append({100 + dim: 1.0})
        parents.append(
            ParentChunk(
                id=parent_id,
                document_id=doc_id,
                parent_index=p_idx,
                text=f"Full parent context {p_idx} for {doc_id}",
                page_numbers=[1],
                section_header="Intro",
                child_ids=c_ids,
                metadata=meta,
            )
        )

    batch = EmbeddedChildBatch(
        dense_vecs=np.vstack(dense_list) if dense_list else np.empty((0, 1024), dtype=np.float32),
        sparse_weights=sparse_list,
    )
    return doc, parents, children, batch


@pytest.fixture
def repo() -> LocalHybridRepository:
    with LocalHybridRepository(path=":memory:") as r:
        yield r


class TestDocumentSyncLifecycle:
    def test_needs_indexing_and_collection_counts(self, repo: LocalHybridRepository) -> None:
        doc, parents, children, batch = make_bundle(doc_id="doc-sync", content_hash="v1")
        assert repo.needs_indexing("doc-sync", "v1") is True
        repo.upsert_document_bundle(doc, parents, children, batch)
        assert repo.needs_indexing("doc-sync", "v1") is False
        assert repo.needs_indexing("doc-sync", "v2") is True
        assert repo.collection_counts() == {"documents": 1, "parent_chunks": 1, "child_chunks": 2}

    def test_upsert_replaces_stale_chunks_on_update(self, repo: LocalHybridRepository) -> None:
        repo.upsert_document_bundle(*make_bundle(doc_id="doc-u", content_hash="v1", num_parents=2))
        repo.upsert_document_bundle(*make_bundle(doc_id="doc-u", content_hash="v2", num_parents=1))
        results = repo.search_parents(
            EncodedQuery(dense_vec=make_dense_vec(1), sparse_weights={101: 1.0})
        )
        ids = {r.parent_chunk.id for r in results}
        assert "doc-u:p1" not in ids
        assert "doc-u:p0" in ids

    def test_delete_document_removes_all_content(self, repo: LocalHybridRepository) -> None:
        doc, parents, children, batch = make_bundle(doc_id="doc-del")
        repo.upsert_document_bundle(doc, parents, children, batch)
        repo.delete_document("doc-del")
        assert repo.needs_indexing("doc-del", doc.content_hash) is True
        assert (
            repo.search_parents(
                EncodedQuery(dense_vec=make_dense_vec(0), sparse_weights={100: 1.0})
            )
            == []
        )

    def test_upsert_raises_on_mismatched_counts(self, repo: LocalHybridRepository) -> None:
        doc, parents, children, batch = make_bundle(children_per_parent=2)
        short = EmbeddedChildBatch(
            dense_vecs=batch.dense_vecs[:1], sparse_weights=batch.sparse_weights[:1]
        )
        with pytest.raises(ValueError, match="Mismatch"):
            repo.upsert_document_bundle(doc, parents, children, short)


class TestHybridSearchAndFiltering:
    def test_deduplicates_children_to_single_parent(self, repo: LocalHybridRepository) -> None:
        doc, parents, children, batch = make_bundle(
            doc_id="doc-d", num_parents=1, children_per_parent=3, dominant_dim_offset=5
        )
        repo.upsert_document_bundle(doc, parents, children, batch)
        results = repo.search_parents(
            EncodedQuery(dense_vec=make_dense_vec(5), sparse_weights={105: 1.0})
        )
        assert len(results) == 1
        assert results[0].parent_chunk.id == "doc-d:p0"

    def test_respects_scalar_and_multi_building_filters(self, repo: LocalHybridRepository) -> None:
        repo.upsert_document_bundle(
            *make_bundle(doc_id="doc-a", project="Alpha", building="B1", dominant_dim_offset=0)
        )
        repo.upsert_document_bundle(
            *make_bundle(
                doc_id="doc-multi",
                project="06038.004",
                building="Blok P - Blok L - Spoed",
                phase="Uitvoering",
                dominant_dim_offset=0,
            )
        )
        q = EncodedQuery(dense_vec=make_dense_vec(0), sparse_weights={100: 1.0})
        assert [r.parent_chunk.document_id for r in repo.search_parents(q, project="Alpha")] == [
            "doc-a"
        ]
        assert [r.parent_chunk.document_id for r in repo.search_parents(q, building="Blok L")] == [
            "doc-multi"
        ]
        assert [r.parent_chunk.document_id for r in repo.search_parents(q, building="Spoed")] == [
            "doc-multi"
        ]
        assert repo.search_parents(q, project="Alpha", building="Blok L") == []

    def test_stores_child_text_and_supports_inspection_iterators(
        self, repo: LocalHybridRepository
    ) -> None:
        doc, parents, children, batch = make_bundle(
            doc_id="doc-insp", num_parents=1, children_per_parent=2, dominant_dim_offset=3
        )
        repo.upsert_document_bundle(doc, parents, children, batch)

        q = EncodedQuery(dense_vec=make_dense_vec(3), sparse_weights={103: 1.0})
        results, child_hits = repo.search_parents_with_stats(q)
        assert child_hits == 2
        assert len(results) == 1
        assert results[0].matched_child_text is not None
        assert results[0].matched_child_text.startswith("Child text ")
        assert results[0].retrieval_score == results[0].score

        doc_parents, doc_children = repo.get_document_chunks("doc-insp")
        assert len(doc_parents) == 1
        assert len(doc_children) == 2
        assert all("text" in ch for ch in doc_children)
        assert len(list(repo.iter_documents())) == 1
