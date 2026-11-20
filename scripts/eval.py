"""CLI entry point for running an evaluation JSON/CSV dataset (`ingenium-eval`)."""

import argparse
import json
from pathlib import Path

from config import get_settings
from evaluation import load_eval_rows, run_evaluation
from pipeline import build_rag_pipeline


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run Ingenium RAG evaluation dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to eval .json or .csv")
    parser.add_argument("--output", type=Path, default=settings.data_dir / "eval_results.json")
    parser.add_argument("--qdrant-path", type=Path, default=settings.qdrant_path)
    parser.add_argument("--top-k", type=int, default=settings.rerank_k)
    parser.add_argument("--device", type=str, default=settings.device)
    args = parser.parse_args()

    pipeline, repo = build_rag_pipeline(qdrant_path=args.qdrant_path, device=args.device)
    with repo:
        rows = load_eval_rows(args.dataset)
        results = run_evaluation(pipeline, rows, top_k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluated {len(results)} question(s) -> {args.output}")


if __name__ == "__main__":
    main()
