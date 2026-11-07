from domain import SearchResult


class Reranker:
    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        terms = set(query.lower().split())
        return sorted(
            results,
            key=lambda result: result.score + sum(
                term in result.chunk.text.lower() for term in terms
            ),
            reverse=True,
        )[:top_k]
