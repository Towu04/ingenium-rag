"""Serverless, disk-persistent Hybrid Vector Repository using local Qdrant."""

import uuid
from collections.abc import Iterable, Sequence

from qdrant_client import QdrantClient, models

from domain import (
    ChildChunk,
    EmbeddedChildBatch,
    EncodedQuery,
    ParentChunk,
    ParsedDoc,
    SearchResult,
)


def _to_uuid(key: str) -> str:
    """Generate a deterministic UUID5 point ID for Qdrant from any string identifier."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


class LocalHybridRepository:
    """Persists documents, parent chunks, and hybrid (dense + sparse) child vectors locally."""

    DOCS_COL = "documents"
    PARENTS_COL = "parent_chunks"
    CHILDREN_COL = "child_chunks"

    def __init__(self, path: str = "./data/qdrant_poc") -> None:
        self.client = QdrantClient(path=path)
        self._init_collections()

    def close(self) -> None:
        """Close the local Qdrant client cleanly."""
        self.client.close()

    def __enter__(self) -> "LocalHybridRepository":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _init_collections(self) -> None:
        if not self.client.collection_exists(self.DOCS_COL):
            self.client.create_collection(self.DOCS_COL, vectors_config={})
        if not self.client.collection_exists(self.PARENTS_COL):
            self.client.create_collection(self.PARENTS_COL, vectors_config={})
        if not self.client.collection_exists(self.CHILDREN_COL):
            self.client.create_collection(
                collection_name=self.CHILDREN_COL,
                vectors_config={
                    "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )

    def collection_counts(self) -> dict[str, int]:
        """Return point counts across the three collections for observability."""
        return {
            col: self.client.count(col).count
            for col in (self.DOCS_COL, self.PARENTS_COL, self.CHILDREN_COL)
        }

    def needs_indexing(self, document_id: str, new_content_hash: str) -> bool:
        records = self.client.retrieve(self.DOCS_COL, ids=[_to_uuid(document_id)])
        if not records or not records[0].payload:
            return True
        return records[0].payload.get("content_hash") != new_content_hash

    def delete_document(self, document_id: str) -> None:
        doc_filter = models.Filter(
            must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
            ]
        )
        self.client.delete(self.DOCS_COL, points_selector=[_to_uuid(document_id)])
        self.client.delete(self.PARENTS_COL, points_selector=doc_filter)
        self.client.delete(self.CHILDREN_COL, points_selector=doc_filter)

    def upsert_document_bundle(
        self,
        document: ParsedDoc,
        parents: Sequence[ParentChunk],
        children: Sequence[ChildChunk],
        embedded_batch: EmbeddedChildBatch,
    ) -> None:
        if len(children) != len(embedded_batch):
            raise ValueError(
                f"Mismatch: {len(children)} children vs {len(embedded_batch)} embeddings."
            )

        self.delete_document(document.document_id)

        self.client.upsert(
            self.DOCS_COL,
            points=[
                models.PointStruct(
                    id=_to_uuid(document.document_id),
                    vector={},
                    payload={
                        "document_id": document.document_id,
                        "content_hash": document.content_hash,
                    },
                )
            ],
        )

        if parents:
            self.client.upsert(
                self.PARENTS_COL,
                points=[
                    models.PointStruct(
                        id=_to_uuid(p.id),
                        vector={},
                        payload=p.model_dump(mode="json"),
                    )
                    for p in parents
                ],
            )

        if children:
            self.client.upsert(
                self.CHILDREN_COL,
                points=[
                    models.PointStruct(
                        id=_to_uuid(c.id),
                        vector={
                            "dense": dense_vec.tolist(),
                            "sparse": models.SparseVector(
                                indices=list(sparse_w.keys()),
                                values=list(sparse_w.values()),
                            ),
                        },
                        payload={
                            "child_id": c.id,
                            "parent_id": c.parent_id,
                            "document_id": c.document_id,
                            "project": c.metadata.project,
                            "building": (
                                c.metadata.buildings
                                if c.metadata.buildings
                                else ([c.metadata.building] if c.metadata.building else None)
                            ),
                            "phase": c.metadata.phase,
                        },
                    )
                    for c, dense_vec, sparse_w in zip(
                        children, embedded_batch.dense_vecs, embedded_batch.sparse_weights
                    )
                ],
            )

    def search_parents(
        self,
        encoded_query: EncodedQuery,
        top_k_parents: int = 15,
        child_candidate_limit: int = 40,
        project: str | None = None,
        building: str | None = None,
        phase: str | None = None,
    ) -> list[SearchResult]:
        query_filter = self._build_filter(project=project, building=building, phase=phase)
        hits = self._query_hybrid_children(encoded_query, query_filter, child_candidate_limit)
        best_by_parent = self._dedupe_by_parent(hits, top_k_parents)
        if not best_by_parent:
            return []

        parent_map = self._fetch_parents(best_by_parent.keys())
        return [
            SearchResult(
                parent_chunk=parent_map[pid],
                matched_child_id=child_id,
                score=score,
            )
            for pid, (child_id, score) in best_by_parent.items()
            if pid in parent_map
        ]

    @staticmethod
    def _build_filter(**kwargs: str | None) -> models.Filter | None:
        conditions: list[models.Condition] = [
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in kwargs.items()
            if v is not None
        ]
        return models.Filter(must=conditions) if conditions else None

    def _query_hybrid_children(
        self,
        query: EncodedQuery,
        query_filter: models.Filter | None,
        limit: int,
    ) -> list[models.ScoredPoint]:
        sparse_vec = models.SparseVector(
            indices=list(query.sparse_weights.keys()),
            values=list(query.sparse_weights.values()),
        )
        return self.client.query_points(
            collection_name=self.CHILDREN_COL,
            prefetch=[
                models.Prefetch(
                    query=query.dense_vec.tolist(),
                    using="dense",
                    filter=query_filter,
                    limit=limit,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    filter=query_filter,
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        ).points

    @staticmethod
    def _dedupe_by_parent(
        hits: Sequence[models.ScoredPoint],
        top_k_parents: int,
    ) -> dict[str, tuple[str, float]]:
        best_by_parent: dict[str, tuple[str, float]] = {}
        for hit in hits:
            if not hit.payload:
                continue
            p_id = hit.payload["parent_id"]
            if p_id not in best_by_parent:
                best_by_parent[p_id] = (hit.payload["child_id"], float(hit.score))
            if len(best_by_parent) >= top_k_parents:
                break
        return best_by_parent

    def _fetch_parents(self, parent_ids: Iterable[str]) -> dict[str, ParentChunk]:
        points = self.client.retrieve(
            self.PARENTS_COL,
            ids=[_to_uuid(pid) for pid in parent_ids],
        )
        return {
            pt.payload["id"]: ParentChunk.model_validate(pt.payload) for pt in points if pt.payload
        }
