from collections.abc import Sequence

from domain import ChildChunk, ParentChunk, SearchResult


class InMemoryRepository:
    """Development repository implementing the offline and online storage contract."""

    def __init__(self) -> None:
        self._chunks: dict[str, ChildChunk] = {}
        self._parents: dict[str, ParentChunk] = {}

    def upsert(
        self,
        parents: Sequence[ParentChunk],
        chunks: Sequence[ChildChunk],
    ) -> None:
        self._parents.update({parent.id: parent for parent in parents})
        self._chunks.update({chunk.id: chunk for chunk in chunks})

    def delete_by_document(self, document_id: str) -> None:
        self._chunks = {
            key: chunk for key, chunk in self._chunks.items()
            if chunk.document_id != document_id
        }
        self._parents = {
            key: parent for key, parent in self._parents.items()
            if parent.document_id != document_id
        }

    def hybrid_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        terms = set(query.lower().split())
        ranked = sorted(
            self._chunks.values(),
            key=lambda chunk: sum(term in chunk.text.lower() for term in terms),
            reverse=True,
        )
        return [
            SearchResult(
                parent_chunk=self._parents[chunk.parent_id],
                matched_child_id=chunk.id,
                score=float(sum(term in chunk.text.lower() for term in terms)),
            )
            for chunk in ranked[:top_k]
        ]
