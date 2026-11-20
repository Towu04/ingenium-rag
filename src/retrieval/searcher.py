"""Hybrid parent searcher using domain-only protocols (zero concrete storage/embedding imports)."""

import logging
from typing import Protocol

from domain import (
    EncodedQuery,
    SearchResult,
    normalize_building,
    normalize_phase,
    normalize_project,
)

logger = logging.getLogger(__name__)


class QueryEncoder(Protocol):
    """Protocol satisfied by `BGEM3Embedder` and lightweight test fakes."""

    def embed_query(self, query: str) -> EncodedQuery: ...


class ParentStore(Protocol):
    """Protocol satisfied by `LocalHybridRepository` and lightweight test fakes."""

    def search_parents(
        self,
        encoded_query: EncodedQuery,
        top_k_parents: int = 15,
        child_candidate_limit: int = 40,
        project: str | None = None,
        building: str | None = None,
        phase: str | None = None,
    ) -> list[SearchResult]: ...


class Searcher:
    """Encodes a user query and executes metadata-filtered hybrid parent retrieval."""

    def __init__(self, embedder: QueryEncoder, repository: ParentStore) -> None:
        self.embedder = embedder
        self.repository = repository

    def search(
        self,
        query: str,
        *,
        top_k_parents: int = 15,
        child_candidate_limit: int = 40,
        project: str | None = None,
        building: str | None = None,
        phase: str | None = None,
        fallback_unfiltered: bool = False,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []

        encoded = self.embedder.embed_query(query)
        norm_project = normalize_project(project)
        norm_building = normalize_building(building)
        norm_phase = normalize_phase(phase)

        results = self.repository.search_parents(
            encoded_query=encoded,
            top_k_parents=top_k_parents,
            child_candidate_limit=child_candidate_limit,
            project=norm_project,
            building=norm_building,
            phase=norm_phase,
        )

        if not results and fallback_unfiltered and any((norm_project, norm_building, norm_phase)):
            logger.warning(
                "Filtered search (project=%s, building=%s, phase=%s) returned 0 results; retrying unfiltered.",
                norm_project,
                norm_building,
                norm_phase,
            )
            results = self.repository.search_parents(
                encoded_query=encoded,
                top_k_parents=top_k_parents,
                child_candidate_limit=child_candidate_limit,
            )

        return results
