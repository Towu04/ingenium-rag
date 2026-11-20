"""Dataset evaluation runner (depends on `domain` only — decoupled from API & concrete pipeline)."""

import csv
import json
from pathlib import Path
from typing import Any, Protocol

from domain import Answer
from evaluation.metrics import citation_precision, source_recall


class AnswerRunner(Protocol):
    """Minimal protocol satisfied by `RAGPipeline` and test doubles."""

    def run(
        self,
        query: str,
        *,
        top_k: int | None = None,
        project: str | None = None,
        building: str | None = None,
        phase: str | None = None,
    ) -> Answer: ...


def load_eval_rows(dataset_path: Path | str) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Evaluation JSON must contain a list of question objects.")
        return data
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    raise ValueError(f"Unsupported evaluation dataset format: {path.suffix}")


def run_evaluation(
    runner: AnswerRunner,
    rows: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run each evaluation question and compute citation precision and source recall."""
    outputs: list[dict[str, Any]] = []
    for row in rows:
        question = str(row["question"])
        answer = runner.run(
            question,
            top_k=top_k,
            project=row.get("project") or None,
            building=row.get("building") or None,
            phase=row.get("phase") or None,
        )
        actual_sources = {c.source_path for c in answer.citations}
        expected_raw = row.get("expected_sources") or []
        if isinstance(expected_raw, str):
            expected_sources = {s.strip() for s in expected_raw.split(";") if s.strip()}
        else:
            expected_sources = {str(s) for s in expected_raw}

        outputs.append(
            {
                "question": question,
                "answer": answer.text,
                "citations": [c.model_dump(mode="json") for c in answer.citations],
                "retrieved_parent_ids": [r.parent_chunk.id for r in answer.retrieved_results],
                "citation_precision": (
                    citation_precision(expected_sources, actual_sources)
                    if expected_sources
                    else None
                ),
                "source_recall": (
                    source_recall(expected_sources, actual_sources) if expected_sources else None
                ),
            }
        )
    return outputs
