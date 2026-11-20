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


def test_rerank_single_multiple_and_empty(
    reranker: BGEM3Reranker, caplog: pytest.LogCaptureFixture
) -> None:
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

    # Default min_score=0.0 keeps all finite candidates; custom min_score filters low scores
    assert len(reranker.rerank("brandwerende deur", candidates, top_k=5)) == 3
    filtered = reranker.rerank("brandwerende deur", candidates, top_k=5, min_score=0.05)
    assert [r.parent_chunk.id for r in filtered] == ["p1", "p2"]

    # Custom min_score dropping all candidates logs a warning
    with caplog.at_level("WARNING", logger="retrieval.reranker"):
        dropped_all = reranker.rerank("brandwerende deur", candidates, top_k=5, min_score=10.0)
    assert dropped_all == []
    assert "dropped all 3 candidate(s)" in caplog.text


def test_reranker_drops_non_finite_scores_and_preserves_child_context(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _NaNReranker:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def compute_score(self, pairs: list[list[str]], **kwargs: Any) -> list[float]:
            return [float("nan"), 0.42, float("inf")]

    monkeypatch.setattr(reranker_module, "FlagReranker", _NaNReranker)
    rr = BGEM3Reranker(device="cpu")
    cands = [
        SearchResult(
            parent_chunk=_result("p0", "bad").parent_chunk,
            matched_child_id="p0:c0",
            matched_child_text="child 0",
            score=0.9,
        ),
        SearchResult(
            parent_chunk=_result("p1", "good").parent_chunk,
            matched_child_id="p1:c0",
            matched_child_text="child 1 text",
            score=0.33,
        ),
        SearchResult(
            parent_chunk=_result("p2", "inf").parent_chunk,
            matched_child_id="p2:c0",
            matched_child_text="child 2",
            score=0.2,
        ),
    ]
    with caplog.at_level("WARNING", logger="retrieval.reranker"):
        out = rr.rerank("q", cands, top_k=5)
    assert [r.parent_chunk.id for r in out] == ["p1"]
    assert out[0].score == pytest.approx(0.42)
    assert out[0].retrieval_score == pytest.approx(0.33)
    assert out[0].matched_child_text == "child 1 text"
    assert "non-finite score(s)" in caplog.text


def test_reranker_enables_fp16_on_mps_and_formats_file_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reranker_module, "FlagReranker", _FakeFlagReranker)
    mps_reranker = BGEM3Reranker(device="mps")
    assert mps_reranker._reranker.init_kwargs["use_fp16"] is True

    parent = ParentChunk(
        id="p-meta",
        document_id="d",
        parent_index=0,
        text="Spec tekst",
        section_header="3.1 Ventilatie",
        metadata=DocumentMetadata(
            source_path="General/05_Uitvoering/Bestek_HVAC_BlokL.pdf",
            project="06038.004",
            building="Blok A - Blok B - Blok L - Spoed",
        ),
    )
    formatted = BGEM3Reranker._format_passage(
        SearchResult(parent_chunk=parent, matched_child_id="p-meta:c0", score=0.5)
    )
    assert "File: Bestek_HVAC_BlokL" in formatted
    assert "Building:" not in formatted


def test_query_filter_extracts_natural_and_explicit_filters() -> None:
    f1 = extract_metadata_filters(
        "Wat is het debiet voor project 06038.004 in Blok L tijdens uitvoering?"
    )
    assert f1 == {"project": "06038.004", "building": "Blok L", "phase": "Uitvoering"}

    f2 = extract_metadata_filters(
        "project:PRJ-2021-104 building:Blok A phase:Aanbesteding luchtgroepen"
    )
    assert f2 == {"project": "PRJ-2021-104", "building": "Blok A", "phase": "Aanbesteding"}
