"""Online RAG pipeline: Query -> Filter -> Hybrid Search -> Cross-Encoder Rerank -> Grounded Generation."""

from collections.abc import Sequence
from typing import Protocol

from domain import Answer, SearchResult
from generation.generator import GroundedGenerator
from retrieval.query_filter import extract_metadata_filters
from retrieval.reranker import BGEM3Reranker
from retrieval.searcher import Searcher


class RerankerProtocol(Protocol):
    """Minimal reranker protocol satisfied by `BGEM3Reranker` and test fakes."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]: ...


class RAGPipeline:
    """Two-stage Parent-Child Hybrid RAG orchestrator."""

    def __init__(
        self,
        searcher: Searcher,
        reranker: RerankerProtocol | None = None,
        generator: GroundedGenerator | None = None,
        retrieve_k: int = 15,
        child_candidate_limit: int = 40,
        rerank_k: int = 5,
    ) -> None:
        self.searcher = searcher
        self.reranker: RerankerProtocol = reranker or BGEM3Reranker()
        self.generator = generator or GroundedGenerator()
        self.retrieve_k = retrieve_k
        self.child_candidate_limit = child_candidate_limit
        self.rerank_k = rerank_k

    def run(
        self,
        query: str,
        *,
        top_k: int | None = None,
        retrieve_k: int | None = None,
        project: str | None = None,
        building: str | None = None,
        phase: str | None = None,
        auto_filter: bool = False,
    ) -> Answer:
        """Execute retrieval, reranking, and citation-grounded generation for a query."""
        inferred = extract_metadata_filters(query) if auto_filter else {}
        eff_project = project if project is not None else inferred.get("project")
        eff_building = building if building is not None else inferred.get("building")
        eff_phase = phase if phase is not None else inferred.get("phase")
        used_only_inferred = auto_filter and project is None and building is None and phase is None

        k_final = top_k if top_k is not None else self.rerank_k
        k_retrieve = max(retrieve_k if retrieve_k is not None else self.retrieve_k, k_final)

        candidates = self.searcher.search(
            query,
            top_k_parents=k_retrieve,
            child_candidate_limit=self.child_candidate_limit,
            project=eff_project,
            building=eff_building,
            phase=eff_phase,
            fallback_unfiltered=used_only_inferred,
        )
        reranked = self.reranker.rerank(query, candidates, top_k=k_final)
        return self.generator.generate(query, reranked)
