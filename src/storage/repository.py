from collections.abc import Sequence

from domain import ChildChunk, SearchResult


class InMemoryRepository:
    """Development repository implementing the offline and online storage contract."""

    def __init__(self) -> None:
        self._chunks: dict[str, ChildChunk] = {}

    def upsert(self, chunks: Sequence[ChildChunk]) -> None:
        self._chunks.update({chunk.id: chunk for chunk in chunks})

    def delete_by_document(self, document_id: str) -> None:
        self._chunks = {
            key: chunk for key, chunk in self._chunks.items()
            if chunk.document_id != document_id
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
                chunk=chunk,
                score=float(sum(term in chunk.text.lower() for term in terms)),
            )
            for chunk in ranked[:top_k]
        ]
