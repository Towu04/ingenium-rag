"""CLI entry point for querying the local Parent-Child Hybrid RAG index (`ingenium-query`)."""

import json
import sys
import time
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from config import configure_logging, get_settings
from domain import Answer, RetrievalTrace
from scripts._cli import (
    CLIError,
    add_common_args,
    build_parser,
    clean_preview,
    err_console,
    out_console,
    run_cli,
    suppress_hf_noise_early,
)


class _PassThroughReranker:
    min_score: float = 0.0

    def rerank(self, query, candidates, top_k: int = 5, min_score: float | None = None):
        return list(candidates)[:top_k]


def _fmt_filters(filters: dict[str, str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in filters.items()) if filters else "none"


def _render_sources_table(trace: RetrievalTrace, *, title: str = "Sources & Citations") -> Table:
    table = Table(title=title, show_lines=False, header_style="bold cyan")
    table.add_column("#", justify="right", style="bold", width=3)
    table.add_column("Score", justify="right", style="green", width=7)
    table.add_column("File", style="white")
    table.add_column("Pages", justify="right", style="dim", width=9)
    table.add_column("Section", style="yellow")
    table.add_column("Building / Phase", style="magenta")

    for idx, res in enumerate(trace.results, start=1):
        p = res.parent_chunk
        m = p.metadata
        file_name = Path(m.source_path).name if m.source_path else p.document_id
        pages = ",".join(map(str, p.page_numbers)) if p.page_numbers else "-"
        section = clean_preview(p.section_header or "-", max_chars=48)
        bldg = ", ".join(m.buildings) if m.buildings else "-"
        phase = m.phase or "-"
        table.add_row(
            str(idx),
            f"{res.score:.4f}",
            file_name,
            pages,
            section,
            f"{bldg} / {phase}",
        )
    return table


def _render_debug_trace(trace: RetrievalTrace) -> None:
    summary = Table(title="Retrieval Debug Trace", show_header=False, box=None, padding=(0, 2))
    summary.add_column("Key", style="bold cyan")
    summary.add_column("Value")
    summary.add_row("Query", trace.query)
    summary.add_row("Explicit filters", _fmt_filters(trace.explicit_filters))
    summary.add_row("Auto filters", _fmt_filters(trace.auto_filters))
    applied_str = _fmt_filters(trace.applied_filters)
    if trace.fallback_unfiltered:
        applied_str += " [yellow](fell back to unfiltered)[/yellow]"
    summary.add_row("Applied filters", applied_str)
    summary.add_row(
        "Stage funnel",
        f"{trace.hybrid_child_hits} child hits -> "
        f"{trace.deduplicated_parents} parents -> "
        f"{trace.reranked_kept} kept / {trace.reranked_dropped} dropped "
        f"(min_score={trace.min_score:.4f}) -> top {len(trace.results)}",
    )
    out_console.print(summary)

    if not trace.results:
        out_console.print("[yellow]No results returned.[/yellow]")
        return

    out_console.print()
    out_console.print(_render_sources_table(trace, title="Ranked Results"))

    for idx, res in enumerate(trace.results, start=1):
        p = res.parent_chunk
        m = p.metadata
        rrf_str = f"{res.retrieval_score:.4f}" if res.retrieval_score is not None else "n/a"
        header = (
            f"[{idx}] rerank={res.score:.4f} | rrf={rrf_str} | "
            f"{m.source_path} (doc={p.document_id}, parent={p.id})"
        )
        meta_line = (
            f"project={m.project or '-'} | buildings={', '.join(m.buildings) or '-'} | "
            f"phase={m.phase or '-'} | pages={p.page_numbers or '-'} | "
            f"section={p.section_header or '-'}"
        )
        child_block = (
            clean_preview(res.matched_child_text, max_chars=450)
            if res.matched_child_text
            else "<not stored in payload; re-index to populate child text>"
        )
        body = (
            f"[dim]{meta_line}[/dim]\n\n"
            f"[bold cyan]Matched child ({res.matched_child_id}):[/bold cyan]\n{child_block}\n\n"
            f"[bold cyan]Full parent chunk:[/bold cyan]\n{p.text}"
        )
        out_console.print(Panel(body, title=header, border_style="blue"))


def _render_answer(trace: RetrievalTrace, answer: Answer) -> None:
    out_console.print(Panel(answer.text, title="Answer", border_style="cyan"))
    if trace.results:
        out_console.print(_render_sources_table(trace))


def _main() -> None:
    suppress_hf_noise_early()
    settings = get_settings()

    parser = build_parser(
        "Query the Ingenium local Qdrant hybrid RAG index.",
        epilog=(
            "Examples:\n"
            "  ingenium-query 'Wat zijn de eisen voor brandwerende deuren?'\n"
            "  ingenium-query 'brandwerende deuren' --building 'Blok L' --debug\n"
            "  ingenium-query 'fire-resistant doors' --json | jq '.trace.results[0]'"
        ),
    )
    parser.add_argument(
        "question", nargs="?", default=None, help="Question to ask (prompts if omitted)."
    )
    parser.add_argument("--project", type=str, default=None, help="Filter by project ID.")
    parser.add_argument("--building", type=str, default=None, help="Filter by building/zone.")
    parser.add_argument("--phase", type=str, default=None, help="Filter by project phase.")
    parser.add_argument(
        "--auto-filter", action="store_true", help="Infer metadata filters from query text."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.rerank_k,
        help=f"Final reranked results to return (default: {settings.rerank_k}).",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=settings.retrieve_k,
        help=f"Pre-rerank parent candidates (default: {settings.retrieve_k}).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help=f"Override reranker score cutoff (default: {settings.reranker_min_score}).",
    )
    parser.add_argument("--device", type=str, default=settings.device, help="cuda / mps / cpu.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Show retrieved & reranked chunks without invoking an LLM.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking and return hybrid RRF results directly.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show filters, funnel counts, RRF + rerank scores, matched child, and full parent.",
    )
    add_common_args(parser, default_qdrant_path=settings.qdrant_path)
    args = parser.parse_args()

    configure_logging(verbose=args.verbose, quiet=args.quiet, json_mode=args.as_json)

    question = args.question or ("" if args.as_json else input("Question: ").strip())
    if not question:
        raise CLIError("No question provided. Pass a question argument or use Q='...' with make.")

    # Fail fast if the local Qdrant index is missing or empty before loading BGE-M3 weights.
    from storage import LocalHybridRepository

    if not args.qdrant_path.exists():
        raise CLIError(f"Qdrant index not found at '{args.qdrant_path}'. Run 'make ingest' first.")
    with LocalHybridRepository(path=str(args.qdrant_path)) as check_repo:
        doc_count = check_repo.collection_counts()["documents"]
        if doc_count == 0:
            raise CLIError(
                f"Qdrant index at '{args.qdrant_path}' is empty (0 documents). "
                "Run 'make ingest' first."
            )

    t0 = time.perf_counter()
    if not args.quiet and not args.as_json:
        err_console.print("[dim]Loading retrieval models and querying index...[/dim]")

    from generation.generator import GroundedGenerator
    from pipeline import RAGPipeline, build_llm_client, build_rag_pipeline
    from retrieval import Searcher

    if args.no_rerank:
        from embedding import BGEM3Embedder

        repo = LocalHybridRepository(path=str(args.qdrant_path))
        embedder = BGEM3Embedder(model_name=settings.embedding_model, device=args.device)
        pipeline = RAGPipeline(
            searcher=Searcher(embedder=embedder, repository=repo),
            reranker=_PassThroughReranker(),
            generator=GroundedGenerator(llm=build_llm_client(settings)),
            retrieve_k=args.retrieve_k,
            rerank_k=args.top_k,
        )
    else:
        pipeline, repo = build_rag_pipeline(qdrant_path=args.qdrant_path, device=args.device)

    with repo:
        trace = pipeline.retrieve_with_trace(
            question,
            top_k=args.top_k,
            retrieve_k=args.retrieve_k,
            project=args.project,
            building=args.building,
            phase=args.phase,
            auto_filter=args.auto_filter,
            min_score=args.min_score,
        )
        answer = (
            None if args.retrieval_only else pipeline.generator.generate(question, trace.results)
        )

    elapsed = time.perf_counter() - t0
    if not args.quiet and not args.as_json:
        err_console.print(f"[dim]Completed in {elapsed:.1f}s.[/dim]")

    if args.as_json:
        payload: dict[str, object] = {"trace": trace.model_dump(mode="json")}
        if answer is not None:
            payload["answer"] = answer.text
            payload["citations"] = [c.model_dump(mode="json") for c in answer.citations]
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return

    if args.debug:
        _render_debug_trace(trace)
        if answer is not None:
            out_console.print()
            out_console.print(Panel(answer.text, title="Answer", border_style="cyan"))
        return

    if args.retrieval_only:
        if not trace.results:
            out_console.print("[yellow]No matching chunks found.[/yellow]")
            return
        out_console.print(_render_sources_table(trace, title="Retrieved Parent Chunks"))
        for idx, res in enumerate(trace.results, start=1):
            p = res.parent_chunk
            title = f"[{idx}] score={res.score:.4f} | {Path(p.metadata.source_path).name}"
            out_console.print(Panel(clean_preview(p.text, max_chars=600), title=title))
        return

    assert answer is not None
    _render_answer(trace, answer)


def main() -> None:
    run_cli(_main)


if __name__ == "__main__":
    main()
