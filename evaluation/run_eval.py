"""Evaluation runner executing a Q&A JSON/CSV dataset against `RAGPipeline`."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluation.metrics import citation_precision
from pipelines.rag_pipeline import RAGPipeline


def load_eval_rows(dataset_path: Path) -> list[dict[str, Any]]:
    if dataset_path.suffix.lower() == ".json":
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Evaluation JSON must contain a list of question objects.")
        return data
    if dataset_path.suffix.lower() == ".csv":
        with dataset_path.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    raise ValueError(f"Unsupported evaluation dataset format: {dataset_path.suffix}")


def run_evaluation(
    pipeline: RAGPipeline,
    rows: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run each evaluation row through `pipeline` and compute citation precision when labels exist."""
    outputs: list[dict[str, Any]] = []
    for row in rows:
        question = str(row["question"])
        answer = pipeline.run(
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
            }
        )
    return outputs


def main() -> None:
    from api.app import build_default_pipeline

    parser = argparse.ArgumentParser(description="Run Ingenium RAG evaluation dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to eval .json or .csv")
    parser.add_argument("--output", type=Path, default=Path("data/eval_results.json"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    pipeline = build_default_pipeline()
    rows = load_eval_rows(args.dataset)
    results = run_evaluation(pipeline, rows, top_k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluated {len(results)} question(s) -> {args.output}")


if __name__ == "__main__":
    main()
