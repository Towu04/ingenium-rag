"""Lightweight teammate inspection, RAGAS chunk export, and batch-query CLI (`ingenium-inspect`)."""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table

from config import get_settings
from domain import RetrievalTrace
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
from storage import LocalHybridRepository


def _cmd_stats(args: argparse.Namespace) -> None:
    ensure_qdrant_ready(args.qdrant_path, require_docs=False)
    with LocalHybridRepository(path=str(args.qdrant_path)) as repo:
        counts = repo.collection_counts()

        if args.doc_id:
            parents, children = repo.get_document_chunks(args.doc_id)
            if not parents:
                raise CLIError(f"No chunks found for document_id={args.doc_id!r}.")

            if args.as_json:
                payload = {
                    "document_id": args.doc_id,
                    "source_path": parents[0].metadata.source_path,
                    "metadata": parents[0].metadata.model_dump(mode="json"),
                    "parents": [p.model_dump(mode="json") for p in parents],
                    "children": children,
                }
                sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
                return

            meta = parents[0].metadata
            out_console.print(
                Panel(
                    f"Source:    {meta.source_path}\n"
                    f"Project:   {meta.project or '-'} | Phase: {meta.phase or '-'} | "
                    f"Buildings: {', '.join(meta.buildings) or '-'}\n"
                    f"Chunks:    {len(parents)} parents, {len(children)} children",
                    title=f"Document {args.doc_id}",
                    border_style="cyan",
                )
            )
            child_by_parent: dict[str, list[dict[str, Any]]] = {}
            for ch in children:
                child_by_parent.setdefault(str(ch.get("parent_id", "")), []).append(ch)
            for p in parents[: args.limit]:
                body_lines = [p.text, "", "[bold cyan]Children:[/bold cyan]"]
                for ch in child_by_parent.get(p.id, []):
                    ch_txt = (
                        clean_preview(str(ch["text"]), max_chars=240)
                        if ch.get("text")
                        else "<not stored in payload>"
                    )
                    body_lines.append(f"  [dim]{ch.get('child_id')}:[/dim] {ch_txt}")
                out_console.print(
                    Panel(
                        "\n".join(body_lines),
                        title=f"{p.id} | pages={p.page_numbers or '-'} | section={p.section_header or '-'}",
                        border_style="blue",
                    )
                )
            if len(parents) > args.limit:
                err_console.print(
                    f"[dim]Showing {args.limit}/{len(parents)} parents (use --limit to see more).[/dim]"
                )
            return

        proj_counts: Counter[str] = Counter()
        bldg_counts: Counter[str] = Counter()
        phase_counts: Counter[str] = Counter()
        doc_parents: Counter[str] = Counter()
        doc_paths: dict[str, str] = {}
        seen_docs: set[str] = set()

        for p in repo.iter_parents():
            doc_parents[p.document_id] += 1
            doc_paths.setdefault(p.document_id, p.metadata.source_path)
            if p.document_id not in seen_docs:
                seen_docs.add(p.document_id)
                proj_counts[p.metadata.project or "<None>"] += 1
                phase_counts[p.metadata.phase or "<None>"] += 1
                if p.metadata.buildings:
                    for b in p.metadata.buildings:
                        bldg_counts[b] += 1
                else:
                    bldg_counts["<None>"] += 1

    if args.as_json:
        payload = {
            "qdrant_path": str(args.qdrant_path),
            "collections": counts,
            "metadata_distribution": {
                "projects": dict(proj_counts.most_common()),
                "phases": dict(phase_counts.most_common()),
                "buildings": dict(bldg_counts.most_common()),
            },
            "top_documents": [
                {"document_id": doc_id, "parent_chunks": n_p, "source_path": doc_paths[doc_id]}
                for doc_id, n_p in doc_parents.most_common(args.limit)
            ],
        }
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return

    overview = Table(title=f"Qdrant Index Overview ({args.qdrant_path})", header_style="bold cyan")
    overview.add_column("Collection", style="cyan")
    overview.add_column("Points", justify="right", style="green")
    for col in ("documents", "parent_chunks", "child_chunks"):
        overview.add_row(col, str(counts[col]))
    out_console.print(overview)

    meta_tbl = Table(title="Metadata Distribution (by Document)", header_style="bold cyan")
    meta_tbl.add_column("Field", style="cyan")
    meta_tbl.add_column("Value")
    meta_tbl.add_column("Docs", justify="right", style="green")
    for field_name, counter in (
        ("Project", proj_counts),
        ("Phase", phase_counts),
        ("Building", bldg_counts),
    ):
        for idx, (val, cnt) in enumerate(counter.most_common(10)):
            meta_tbl.add_row(field_name if idx == 0 else "", val, str(cnt))
    out_console.print(meta_tbl)

    docs_tbl = Table(title=f"Top {args.limit} Documents by Parent Chunks", header_style="bold cyan")
    docs_tbl.add_column("Doc ID", style="cyan", width=16)
    docs_tbl.add_column("Parents", justify="right", style="green", width=8)
    docs_tbl.add_column("Source Path")
    for doc_id, n_p in doc_parents.most_common(args.limit):
        docs_tbl.add_row(doc_id, str(n_p), doc_paths.get(doc_id, ""))
    out_console.print(docs_tbl)


def _cmd_export(args: argparse.Namespace) -> None:
    ensure_qdrant_ready(args.qdrant_path, require_docs=False)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_filter = args.doc_id

    with LocalHybridRepository(path=str(args.qdrant_path)) as repo:
        parents = [p.model_dump(mode="json") for p in repo.iter_parents(document_id=doc_filter)]
        children = list(repo.iter_children(document_id=doc_filter))

    if args.format == "json":
        p_path = out_dir / "parent_chunks.json"
        c_path = out_dir / "child_chunks.json"
        p_path.write_text(json.dumps(parents, indent=2, ensure_ascii=False), encoding="utf-8")
        c_path.write_text(json.dumps(children, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        p_path = out_dir / "parent_chunks.jsonl"
        c_path = out_dir / "child_chunks.jsonl"
        with p_path.open("w", encoding="utf-8") as fh:
            for row in parents:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with c_path.open("w", encoding="utf-8") as fh:
            for row in children:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.as_json:
        sys.stdout.write(
            json.dumps(
                {
                    "parent_chunks": len(parents),
                    "child_chunks": len(children),
                    "parent_file": str(p_path),
                    "child_file": str(c_path),
                },
                indent=2,
            )
            + "\n"
        )
        return

    tbl = Table(title="Chunk Export Complete", header_style="bold cyan")
    tbl.add_column("Collection", style="cyan")
    tbl.add_column("Exported", justify="right", style="green")
    tbl.add_column("Output Path")
    tbl.add_row("parent_chunks", str(len(parents)), str(p_path))
    tbl.add_row("child_chunks", str(len(children)), str(c_path))
    out_console.print(tbl)


def _load_questions(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Questions JSON must be a list of strings or objects.")
        out: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, str):
                out.append({"question": item})
            elif isinstance(item, dict) and "question" in item:
                out.append(item)
        return out
    return [
        {"question": line.strip()}
        for line in raw.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _render_markdown_report(traces: list[RetrievalTrace]) -> str:
    lines: list[str] = ["# Batch Retrieval Inspection Report", ""]
    for q_idx, trace in enumerate(traces, start=1):
        lines.append(f"## Q{q_idx}: {trace.query}")
        lines.append(
            f"- **Filters**: applied=`{trace.applied_filters or {}}` "
            f"(explicit=`{trace.explicit_filters or {}}`, auto=`{trace.auto_filters or {}}`)"
        )
        lines.append(
            f"- **Funnel**: `{trace.hybrid_child_hits}` child hits -> "
            f"`{trace.deduplicated_parents}` parents -> "
            f"`{trace.reranked_kept}` kept / `{trace.reranked_dropped}` dropped "
            f"(min_score=`{trace.min_score:.4f}`)"
        )
        lines.append("")
        if not trace.results:
            lines.append("_No results returned._\n")
            continue
        for rank, res in enumerate(trace.results, start=1):
            p = res.parent_chunk
            m = p.metadata
            rrf = f"{res.retrieval_score:.4f}" if res.retrieval_score is not None else "n/a"
            lines.append(
                f"### [{rank}] rerank=`{res.score:.4f}` | rrf=`{rrf}` | "
                f"`{m.source_path}` (pages `{p.page_numbers}`, section: `{p.section_header}`)"
            )
            lines.append(
                f"- Metadata: project=`{m.project}`, buildings=`{m.buildings}`, phase=`{m.phase}`"
            )
            lines.append(f"- **Matched child (`{res.matched_child_id}`)**:")
            child_txt = res.matched_child_text or "<not stored in payload>"
            lines.append(f"```text\n{child_txt}\n```")
            lines.append(f"- **Full parent (`{p.id}`)**:")
            lines.append(f"```text\n{p.text}\n```")
            lines.append("")
    return "\n".join(lines)


def _cmd_batch_query(args: argparse.Namespace) -> None:
    ensure_qdrant_ready(args.qdrant_path, require_docs=True)
    questions = _load_questions(args.questions)
    if not questions:
        raise CLIError(f"No questions found in '{args.questions}'.")

    # Lazy import keeps `stats` and `export` instant without importing torch/FlagEmbedding.
    from pipeline import build_rag_pipeline

    t0 = time.perf_counter()
    traces: list[RetrievalTrace] = []
    with PipelineProgress(quiet=args.quiet, json_mode=args.as_json) as prog:
        pipeline, repo = build_rag_pipeline(
            qdrant_path=args.qdrant_path, device=args.device, on_progress=prog
        )
        with repo:
            prog("bar_start", 0, len(questions), "Running batch queries")
            for idx, item in enumerate(questions, start=1):
                trace = pipeline.retrieve_with_trace(
                    item["question"],
                    top_k=args.top_k,
                    retrieve_k=args.retrieve_k,
                    project=item.get("project"),
                    building=item.get("building"),
                    phase=item.get("phase"),
                    auto_filter=args.auto_filter,
                    min_score=args.min_score,
                )
                traces.append(trace)
                prog("bar_advance", idx, len(questions), item["question"][:48])
            prog("bar_done", len(questions), len(questions), "Running batch queries")

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".json":
        out_path.write_text(
            json.dumps([t.model_dump(mode="json") for t in traces], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        out_path.write_text(_render_markdown_report(traces), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    if args.as_json:
        sys.stdout.write(
            json.dumps(
                {
                    "questions": len(traces),
                    "report_path": str(out_path),
                    "elapsed_seconds": round(elapsed, 2),
                    "traces": [t.model_dump(mode="json") for t in traces],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        return

    tbl = Table(
        title=f"Batch Query Summary ({len(traces)} questions in {elapsed:.1f}s)",
        header_style="bold cyan",
    )
    tbl.add_column("#", justify="right", width=3)
    tbl.add_column("Question")
    tbl.add_column("Kept", justify="right", style="green", width=5)
    tbl.add_column("Top-1 Score", justify="right", style="cyan", width=11)
    tbl.add_column("Top-1 Source")
    for idx, tr in enumerate(traces, start=1):
        top_score = f"{tr.results[0].score:.4f}" if tr.results else "-"
        top_src = Path(tr.results[0].parent_chunk.metadata.source_path).name if tr.results else "-"
        tbl.add_row(str(idx), clean_preview(tr.query, 55), str(len(tr.results)), top_score, top_src)
    out_console.print(tbl)
    err_console.print(f"[dim]Saved full report -> {out_path}[/dim]")


def _main() -> None:
    suppress_hf_noise_early()
    settings = get_settings()

    parser = build_parser(
        "Inspect Qdrant chunks, export for RAGAS, or run batch retrieval reports.",
        epilog=(
            "Examples:\n"
            "  ingenium-inspect stats\n"
            "  ingenium-inspect stats --doc-id 1313846fb39c5f58\n"
            "  ingenium-inspect export --out data/exports\n"
            "  ingenium-inspect batch-query --questions questions.txt --out report.md"
        ),
    )
    add_common_args(parser, default_qdrant_path=settings.qdrant_path)
    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="Print DB stats, metadata distributions, or doc chunks.")
    p_stats.add_argument("--doc-id", type=str, default=None, help="Inspect chunks for one doc ID.")
    p_stats.add_argument("--limit", type=int, default=15, help="Max docs or parents (default: 15).")
    p_stats.set_defaults(func=_cmd_stats)

    p_exp = sub.add_parser("export", help="Export parent & child chunks to JSONL/JSON for RAGAS.")
    p_exp.add_argument(
        "--out",
        type=Path,
        default=Path("data/exports"),
        help="Output directory (default: data/exports).",
    )
    p_exp.add_argument(
        "--format",
        choices=("jsonl", "json"),
        default="jsonl",
        help="Export format (default: jsonl).",
    )
    p_exp.add_argument("--doc-id", type=str, default=None, help="Filter export to one document ID.")
    p_exp.set_defaults(func=_cmd_export)

    p_batch = sub.add_parser("batch-query", help="Run questions from .txt/.json and save a report.")
    p_batch.add_argument("--questions", type=Path, required=True, help="Input .txt or .json file.")
    p_batch.add_argument(
        "--out",
        type=Path,
        default=Path("data/exports/batch_query_report.md"),
        help="Output .md or .json report path.",
    )
    p_batch.add_argument("--top-k", type=int, default=settings.rerank_k)
    p_batch.add_argument("--retrieve-k", type=int, default=settings.retrieve_k)
    p_batch.add_argument("--min-score", type=float, default=None)
    p_batch.add_argument("--auto-filter", action="store_true")
    p_batch.add_argument("--device", type=str, default=settings.device)
    p_batch.set_defaults(func=_cmd_batch_query)

    args = parser.parse_args()
    setup_cli_logging(verbose=args.verbose, quiet=args.quiet, json_mode=args.as_json)
    args.func(args)


def main() -> None:
    run_cli(_main)


if __name__ == "__main__":
    main()
