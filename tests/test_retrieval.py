"""Unit tests for Stage 5 (`retrieval`: query_filter, Searcher protocols, BGEM3Reranker)."""

from typing import Any

import pytest

import retrieval.reranker as reranker_module
from domain import DocumentMetadata, ParentChunk, SearchResult
from retrieval import BGEM3Reranker, extract_metadata_filters


class _FakeFlagReranker:
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


def test_rerank_single_multiple_and_empty(reranker: BGEM3Reranker) -> None:
    assert reranker.rerank("q", []) == []

    single = reranker.rerank("answer", [_result("p", "the answer")])
    assert [r.parent_chunk.id for r in single] == ["p"]
    assert single[0].score == 1.0

    candidates = [
        _result("p0", "nothing relevant", score=0.9),
        _result("p1", "brandwerende deur EI1 30", score=0.1),
        _result("p2", "deur", score=0.5),
    ]
    top2 = reranker.rerank("brandwerende deur", candidates, top_k=2)
    assert [r.parent_chunk.id for r in top2] == ["p1", "p2"]


def test_query_filter_extracts_natural_and_explicit_filters() -> None:
    f1 = extract_metadata_filters(
        "Wat is het debiet voor project 06038.004 in Blok L tijdens uitvoering?"
    )
    assert f1 == {"project": "06038.004", "building": "Blok L", "phase": "Uitvoering"}

    f2 = extract_metadata_filters(
        "project:PRJ-2021-104 building:Blok A phase:Aanbesteding luchtgroepen"
    )
    assert f2 == {"project": "PRJ-2021-104", "building": "Blok A", "phase": "Aanbesteding"}
