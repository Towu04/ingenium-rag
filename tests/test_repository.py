"""Behavioral unit tests for LocalHybridRepository using in-memory Qdrant."""

import numpy as np
import pytest

from domain import ChildChunk, DocumentMetadata, ParentChunk, ParsedDoc
from storage.embedder import EmbeddedChildBatch, EncodedQuery
from storage.repository import LocalHybridRepository


# --------------------------------------------------------------------------- #
# Test Helpers & Factories
# --------------------------------------------------------------------------- #

def make_dense_vec(dominant_index: int = 0) -> np.ndarray:
    """Creates a 1024-dim unit vector pointing along `dominant_index` for predictable similarity."""
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
    """Builds a valid, self-consistent (doc, parents, children, embeddings) bundle."""
    meta = DocumentMetadata(
        source_path=f"/docs/{doc_id}.pdf",
        sharepoint_url=f"https://sp.local/{doc_id}",
        project=project,
        building=building,
        phase=phase,
        extra={},
    )
    doc = ParsedDoc(
        document_id=doc_id,
        content_hash=content_hash,
        markdown="# Sample Doc",
        metadata=meta,
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
    """Provides a fresh, isolated in-memory repository for each test."""
    return LocalHybridRepository(path=":memory:")


# --------------------------------------------------------------------------- #
# Behavioral Tests
# --------------------------------------------------------------------------- #

class TestDocumentSyncLifecycle:
    def test_needs_indexing_for_unseen_and_modified_documents(
        self, repo: LocalHybridRepository
    ) -> None:
        doc, parents, children, batch = make_bundle(doc_id="doc-sync", content_hash="v1")

        # 1. Unseen document requires indexing
        assert repo.needs_indexing("doc-sync", "v1") is True

        # 2. After upsert, same content hash does NOT require indexing
        repo.upsert_document_bundle(doc, parents, children, batch)
        assert repo.needs_indexing("doc-sync", "v1") is False

        # 3. Changed content hash requires re-indexing
        assert repo.needs_indexing("doc-sync", "v2") is True

    def test_upsert_replaces_stale_chunks_on_document_update(
        self, repo: LocalHybridRepository
    ) -> None:
        # Initial version has 2 parent chunks
        doc_v1, parents_v1, children_v1, batch_v1 = make_bundle(
            doc_id="doc-update", content_hash="v1", num_parents=2
        )
        repo.upsert_document_bundle(doc_v1, parents_v1, children_v1, batch_v1)

        # Updated version shrinks to 1 parent chunk
        doc_v2, parents_v2, children_v2, batch_v2 = make_bundle(
            doc_id="doc-update", content_hash="v2", num_parents=1
        )
        repo.upsert_document_bundle(doc_v2, parents_v2, children_v2, batch_v2)

        # Searching should only ever surface the surviving parent from v2
        query = EncodedQuery(dense_vec=make_dense_vec(1), sparse_weights={101: 1.0})
        results = repo.search_parents(query)

        returned_parent_ids = {r.parent_chunk.id for r in results}
        assert "doc-update:p1" not in returned_parent_ids
        assert "doc-update:p0" in returned_parent_ids

    def test_delete_document_removes_all_searchable_content(
        self, repo: LocalHybridRepository
    ) -> None:
        doc, parents, children, batch = make_bundle(doc_id="doc-del")
        repo.upsert_document_bundle(doc, parents, children, batch)

        repo.delete_document("doc-del")

        assert repo.needs_indexing("doc-del", doc.content_hash) is True
        query = EncodedQuery(dense_vec=make_dense_vec(0), sparse_weights={100: 1.0})
        assert repo.search_parents(query) == []

    def test_upsert_raises_on_mismatched_children_and_embeddings(
        self, repo: LocalHybridRepository
    ) -> None:
        doc, parents, children, batch = make_bundle(children_per_parent=2)
        short_batch = EmbeddedChildBatch(
            dense_vecs=batch.dense_vecs[:1],
            sparse_weights=batch.sparse_weights[:1],
        )

        with pytest.raises(ValueError, match="Mismatch"):
            repo.upsert_document_bundle(doc, parents, children, short_batch)


class TestHybridSearchAndFiltering:
    def test_search_deduplicates_multiple_matching_children_to_single_parent(
        self, repo: LocalHybridRepository
    ) -> None:
        # 1 parent with 3 children that all share the same vector
        doc, parents, children, batch = make_bundle(
            doc_id="doc-dedupe", num_parents=1, children_per_parent=3, dominant_dim_offset=5
        )
        repo.upsert_document_bundle(doc, parents, children, batch)

        query = EncodedQuery(dense_vec=make_dense_vec(5), sparse_weights={105: 1.0})
        results = repo.search_parents(query, top_k_parents=5)

        assert len(results) == 1
        assert results[0].parent_chunk.id == "doc-dedupe:p0"
        assert results[0].matched_child_id in {c.id for c in children}
        assert results[0].score > 0

    def test_search_ranks_relevant_parent_first(
        self, repo: LocalHybridRepository
    ) -> None:
        # Parent 0 aligns with dim 10, Parent 1 aligns with dim 11
        doc, parents, children, batch = make_bundle(
            doc_id="doc-rank", num_parents=2, children_per_parent=1, dominant_dim_offset=10
        )
        repo.upsert_document_bundle(doc, parents, children, batch)

        # Query specifically targets Parent 1 (dim 11 / token 111)
        query = EncodedQuery(dense_vec=make_dense_vec(11), sparse_weights={111: 1.0})
        results = repo.search_parents(query, top_k_parents=2)

        assert len(results) == 2
        assert results[0].parent_chunk.id == "doc-rank:p1"
        assert results[0].score >= results[1].score

    def test_search_respects_metadata_filters(
        self, repo: LocalHybridRepository
    ) -> None:
        repo.upsert_document_bundle(
            *make_bundle(doc_id="doc-a", project="Alpha", building="B1", dominant_dim_offset=0)
        )
        repo.upsert_document_bundle(
            *make_bundle(doc_id="doc-b", project="Beta", building="B2", dominant_dim_offset=0)
        )

        query = EncodedQuery(dense_vec=make_dense_vec(0), sparse_weights={100: 1.0})

        alpha_results = repo.search_parents(query, project="Alpha")
        assert [r.parent_chunk.document_id for r in alpha_results] == ["doc-a"]

        b2_results = repo.search_parents(query, building="B2")
        assert [r.parent_chunk.document_id for r in b2_results] == ["doc-b"]

        empty_results = repo.search_parents(query, project="Alpha", building="B2")
        assert empty_results == []

    def test_search_matches_individual_building_in_multi_building_document(
        self, repo: LocalHybridRepository
    ) -> None:
        repo.upsert_document_bundle(
            *make_bundle(
                doc_id="doc-multi",
                project="06038.004",
                building="Blok P - Blok L - Spoed",
                phase="Uitvoering",
                dominant_dim_offset=0,
            )
        )
        query = EncodedQuery(dense_vec=make_dense_vec(0), sparse_weights={100: 1.0})

        assert [r.parent_chunk.document_id for r in repo.search_parents(query, building="Blok L")] == [
            "doc-multi"
        ]
        assert [r.parent_chunk.document_id for r in repo.search_parents(query, building="Spoed")] == [
            "doc-multi"
        ]
        assert repo.search_parents(query, building="Blok A") == []

    def test_search_on_empty_repository_returns_empty_list(
        self, repo: LocalHybridRepository
    ) -> None:
        query = EncodedQuery(dense_vec=make_dense_vec(0), sparse_weights={100: 1.0})
        assert repo.search_parents(query) == []
