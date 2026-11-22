"""CLI entry point for running an evaluation JSON/CSV dataset (`ingenium-eval`)."""

import json
import sys
import time
from pathlib import Path

from rich.table import Table

from config import get_settings
from evaluation import load_eval_rows, run_evaluation
from scripts._cli import (
    CLIError,
    PipelineProgress,
    add_common_args,
    build_parser,
    clean_preview,
    ensure_qdrant_ready,
    err_console,
    out_console,
    run_cli,
    setup_cli_logging,
    suppress_hf_noise_early,
)


def _mean_metric(results: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in results if r.get(key) is not None]
    return (sum(vals) / len(vals)) if vals else None


def _main() -> None:
    suppress_hf_noise_early()
    settings = get_settings()

    parser = build_parser(
        "Run Ingenium RAG evaluation dataset and report citation precision / source recall.",
        epilog=(
            "Examples:\n"
            "  ingenium-eval --dataset data/eval.json\n"
            "  ingenium-eval --dataset data/eval.csv --out data/eval_results.json --json"
        ),
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to eval .json or .csv.")
    parser.add_argument(
        "--out",
        "--output",
        dest="output",
        type=Path,
        default=settings.data_dir / "eval_results.json",
        help=f"Output JSON path (default: {settings.data_dir / 'eval_results.json'}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.rerank_k,
        help=f"Top-k reranked results per question (default: {settings.rerank_k}).",
    )
    parser.add_argument("--device", type=str, default=settings.device, help="cuda / mps / cpu.")
    add_common_args(parser, default_qdrant_path=settings.qdrant_path)
    args = parser.parse_args()

    setup_cli_logging(verbose=args.verbose, quiet=args.quiet, json_mode=args.as_json)

    ensure_qdrant_ready(args.qdrant_path, require_docs=True)
    rows = load_eval_rows(args.dataset)
    if not rows:
        raise CLIError(f"Evaluation dataset '{args.dataset}' is empty.")

    t0 = time.perf_counter()
    from pipeline import build_rag_pipeline

    with PipelineProgress(quiet=args.quiet, json_mode=args.as_json) as prog:
        pipeline, repo = build_rag_pipeline(
            qdrant_path=args.qdrant_path, device=args.device, on_progress=prog
        )
        with repo:
            results = run_evaluation(pipeline, rows, top_k=args.top_k, on_progress=prog)

    elapsed = time.perf_counter() - t0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    mean_prec = _mean_metric(results, "citation_precision")
    mean_rec = _mean_metric(results, "source_recall")

    if args.as_json:
        sys.stdout.write(
            json.dumps(
                {
                    "questions": len(results),
                    "elapsed_seconds": round(elapsed, 2),
                    "mean_citation_precision": round(mean_prec, 4)
                    if mean_prec is not None
                    else None,
                    "mean_source_recall": round(mean_rec, 4) if mean_rec is not None else None,
                    "output_path": str(args.output),
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        return

    tbl = Table(
        title=f"Evaluation Results ({len(results)} questions in {elapsed:.1f}s)",
        header_style="bold cyan",
    )
    tbl.add_column("#", justify="right", width=3)
    tbl.add_column("Question")
    tbl.add_column("Citations", justify="right", width=9)
    tbl.add_column("Precision", justify="right", style="green", width=10)
    tbl.add_column("Recall", justify="right", style="cyan", width=10)
    for idx, row in enumerate(results, start=1):
        prec = f"{row['citation_precision']:.2f}" if row["citation_precision"] is not None else "-"
        rec = f"{row['source_recall']:.2f}" if row["source_recall"] is not None else "-"
        tbl.add_row(
            str(idx),
            clean_preview(str(row["question"]), 60),
            str(len(row["citations"])),
            prec,
            rec,
        )
    tbl.add_section()
    tbl.add_row(
        "",
        "[bold]Mean[/bold]",
        "",
        f"[bold]{mean_prec:.2f}[/bold]" if mean_prec is not None else "-",
        f"[bold]{mean_rec:.2f}[/bold]" if mean_rec is not None else "-",
    )
    out_console.print(tbl)
    err_console.print(f"[dim]Saved detailed results -> {args.output}[/dim]")


def main() -> None:
    run_cli(_main)


if __name__ == "__main__":
    main()
