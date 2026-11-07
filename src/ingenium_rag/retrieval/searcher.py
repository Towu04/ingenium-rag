from ingenium_rag.domain import SearchResult
from ingenium_rag.storage.repository import InMemoryRepository


class Searcher:
    def __init__(self, repository: InMemoryRepository) -> None:
        self.repository = repository

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        results = self.repository.hybrid_search(query, top_k=top_k * 2)
        seen_parents: set[str] = set()
        deduplicated: list[SearchResult] = []
        for result in results:
            if result.chunk.parent_id not in seen_parents:
                seen_parents.add(result.chunk.parent_id)
                deduplicated.append(result)
        return deduplicated[:top_k]
