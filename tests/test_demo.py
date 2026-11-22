"""Offline unit tests for stakeholder demo script helpers (`scripts/demo.py`)."""

import json
from pathlib import Path

import pytest

from domain import Answer, DocumentMetadata, ParentChunk, RetrievalTrace, SearchResult
from scripts._cli import CLIError
from scripts.demo import (
    format_answer_text,
    format_location,
    format_source_name,
    load_demo_scenarios,
    render_relevance_bar,
)


def _sample_result(
    *,
    source_path: str = "12345001-INT/General/05_Uitvoering/12345.001_Bestek.pdf",
    score: float = 0.91,
    pages: list[int] | None = None,
    section: str | None = "4.1 Brandveiligheid",
) -> SearchResult:
    parent = ParentChunk(
        id="p1",
        document_id="doc1",
        parent_index=0,
        text="Brandwerende deuren dienen te voldoen aan EI1 30.",
        page_numbers=[12, 13] if pages is None else pages,
        section_header=section,
        metadata=DocumentMetadata(
            source_path=source_path,
            project="12345.001",
            buildings=["Blok A"],
            phase="Uitvoering",
        ),
    )
    return SearchResult(
        parent_chunk=parent,
        score=score,
        matched_child_id="c1",
        matched_child_text="Brandwerende deuren dienen te voldoen aan EI1 30.",
    )


def test_load_demo_scenarios_array_and_wrapped_object(tmp_path: Path) -> None:
    arr_file = tmp_path / "scenarios.json"
    arr_file.write_text(
        json.dumps(
            [
                {
                    "title": "Dutch Query",
                    "question": "Wat zijn de eisen?",
                    "project": "12345.001",
                    "top_k": 4,
                }
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_demo_scenarios(arr_file)
    assert len(loaded) == 1
    assert loaded[0].title == "Dutch Query"
    assert loaded[0].project == "12345.001"
    assert loaded[0].top_k == 4

    wrapped_file = tmp_path / "wrapped.json"
    wrapped_file.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "title": "Refusal",
                        "question": "Lunch menu?",
                        "expect_refusal": True,
                        "min_score": 0.25,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    wrapped = load_demo_scenarios(wrapped_file)
    assert len(wrapped) == 1
    assert wrapped[0].expect_refusal is True
    assert wrapped[0].min_score == 0.25


def test_load_demo_scenarios_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(CLIError, match="not found"):
        load_demo_scenarios(tmp_path / "missing.json")

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CLIError, match="Invalid JSON"):
        load_demo_scenarios(bad_json)

    missing_fields = tmp_path / "incomplete.json"
    missing_fields.write_text(json.dumps([{"title": "Only title"}]), encoding="utf-8")
    with pytest.raises(CLIError, match="requires non-empty 'title' and 'question'"):
        load_demo_scenarios(missing_fields)


def test_format_source_name_strips_urls_and_prefixes() -> None:
    url = "https://company.sharepoint.com/sites/12345001-INT/Shared%20Documents/General/01_SO/Nota%20Technieken.pdf"
    assert format_source_name(url) == "Nota Technieken.pdf"

    nested = "12345001-INT/General/05_Uitvoering/sub/Bestek_HVAC.xlsx"
    assert format_source_name(nested) == "Bestek_HVAC.xlsx"

    win_path = r"C:\projects\docs\Report_v2.docx"
    assert format_source_name(win_path) == "Report_v2.docx"

    assert format_source_name("", fallback="doc-42") == "doc-42"


def test_format_location_and_relevance_bar() -> None:
    assert format_location([5], "4.1.2 NOODSTROOM") == "p. 5 · 4.1.2 NOODSTROOM"
    assert format_location([3, 4, 5], None) == "pp. 3-5"
    assert format_location([], "Sheet1 (Rows 1-12)") == "Sheet1 (Rows 1-12)"
    assert format_location([], None) == "—"

    assert "95%" in render_relevance_bar(0.95)
    assert "bold green" in render_relevance_bar(0.95)
    assert "yellow" in render_relevance_bar(0.55)
    assert "dim" in render_relevance_bar(0.20)


def test_format_answer_text_extractive_and_refusal() -> None:
    res = _sample_result()
    trace_hit = RetrievalTrace(query="brandwerende deuren", min_score=0.15, results=[res])
    ans_hit = Answer(text="[Stub LLM — raw]", citations=[], retrieved_results=[res])
    rendered = format_answer_text(trace_hit, ans_hit, using_llm=False)
    assert "[Stub LLM" not in rendered
    assert "12345.001_Bestek.pdf" in rendered
    assert "EI1 30" in rendered

    trace_empty = RetrievalTrace(
        query="lunch menu",
        deduplicated_parents=10,
        reranked_dropped=10,
        min_score=0.15,
        results=[],
    )
    ans_empty = Answer(text="No info.", citations=[], retrieved_results=[])
    refusal = format_answer_text(trace_empty, ans_empty, using_llm=False)
    assert "No supporting evidence" in refusal
    assert "0.15" in refusal
