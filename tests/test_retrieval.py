"""Unit tests for the BGE cross-encoder reranker wrapper (model weights are faked)."""

from typing import Any

import pytest

import retrieval.reranker as reranker_module
from domain import DocumentMetadata, ParentChunk, SearchResult
from retrieval.reranker import BGEM3Reranker


class _FakeFlagReranker:
    """Scores a pair by how many query tokens appear in the passage."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = kwargs

    def compute_score(self, pairs: list[list[str]], **kwargs: Any) -> list[float] | float:
        scores = [
            float(sum(tok in passage.lower() for tok in query.lower().split()))
            for query, passage in pairs
        ]
        return scores[0] if len(scores) == 1 else scores


@pytest.fixture
def reranker(monkeypatch: pytest.MonkeyPatch) -> BGEM3Reranker:
    monkeypatch.setattr(reranker_module, "FlagReranker", _FakeFlagReranker)
    return BGEM3Reranker(device="cpu")


def _result(pid: str, text: str, score: float = 0.5) -> SearchResult:
    parent = ParentChunk(
        id=pid,
        document_id="d",
        parent_index=0,
        text=text,
        metadata=DocumentMetadata(source_path="a.md", project="A"),
    )
    return SearchResult(parent_chunk=parent, matched_child_id=f"{pid}:c0", score=score)


def test_rerank_single_candidate_handles_scalar_score(reranker: BGEM3Reranker) -> None:
    results = reranker.rerank("answer", [_result("p", "the answer")])
    assert [r.parent_chunk.id for r in results] == ["p"]
    assert results[0].score == 1.0


def test_rerank_reorders_and_truncates(reranker: BGEM3Reranker) -> None:
    candidates = [
        _result("p0", "nothing relevant here", score=0.9),
        _result("p1", "brandwerende deur EI1 30", score=0.1),
        _result("p2", "deur", score=0.5),
    ]
    results = reranker.rerank("brandwerende deur", candidates, top_k=2)
    assert [r.parent_chunk.id for r in results] == ["p1", "p2"]


def test_rerank_empty_candidates(reranker: BGEM3Reranker) -> None:
    assert reranker.rerank("q", []) == []
