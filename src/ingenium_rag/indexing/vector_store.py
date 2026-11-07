from collections.abc import Sequence

from ingenium_rag.models import ChildChunk, SearchResult


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, ChildChunk] = {}

    def upsert(self, chunks: Sequence[ChildChunk]) -> None:
        self._chunks.update({chunk.id: chunk for chunk in chunks})

    def delete_by_doc_id(self, document_id: str) -> None:
        self._chunks = {key: value for key, value in self._chunks.items()
                        if value.document_id != document_id}

    def hybrid_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        terms = set(query.lower().split())
        ranked = sorted(
            self._chunks.values(),
            key=lambda chunk: sum(term in chunk.text.lower() for term in terms),
            reverse=True,
        )
        return [SearchResult(chunk=chunk, score=float(sum(
            term in chunk.text.lower() for term in terms
        ))) for chunk in ranked[:top_k]]
