from domain import Answer
from generation.generator import GroundedGenerator
from retrieval.reranker import Reranker
from retrieval.searcher import Searcher


class RAGPipeline:
    def __init__(
        self,
        searcher: Searcher,
        reranker: Reranker | None = None,
        generator: GroundedGenerator | None = None,
    ) -> None:
        self.searcher = searcher
        self.reranker = reranker or Reranker()
        self.generator = generator or GroundedGenerator()

    def run(self, query: str, top_k: int = 5) -> Answer:
        results = self.searcher.search(query, top_k=top_k)
        results = self.reranker.rerank(query, results, top_k=top_k)
        return self.generator.generate(query, results)
