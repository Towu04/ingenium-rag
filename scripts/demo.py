"""Stakeholder-ready live presentation & interactive Q&A CLI (`make demo`)."""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from config import detect_device, get_settings
from domain import Answer, RetrievalTrace
from scripts._cli import (
    CLIError,
    build_parser,
    clean_preview,
    ensure_qdrant_ready,
    format_duration,
    format_page_ranges,
    run_cli,
    suppress_hf_noise_early,
)

DEFAULT_SCENARIOS_PATH = Path("data/demo_scenarios.json")
DEFAULT_DEMO_MIN_SCORE = 0.15


@dataclass(frozen=True)
class DemoScenario:
    """Single scripted walkthrough scenario loaded from an external JSON file."""

    title: str
    question: str
    description: str = ""
    project: str | None = None
    building: str | None = None
    phase: str | None = None
    top_k: int = 3
    min_score: float | None = None
    expect_refusal: bool = False


def load_demo_scenarios(path: Path) -> list[DemoScenario]:
    """Load and validate demo scenarios from a JSON list or `{\"scenarios\": [...]}` file."""
    if not path.exists():
        raise CLIError(
            f"Demo scenarios file not found: {path}. "
            "Create it or pass --scenarios <path> (or run with --live for interactive Q&A)."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CLIError(f"Invalid JSON in demo scenarios file '{path}': {exc}") from exc

    items = raw.get("scenarios") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise CLIError(f"Scenarios file '{path}' must contain a non-empty JSON array of scenarios.")

    scenarios: list[DemoScenario] = []
    for idx, entry in enumerate(items, start=1):
        if not isinstance(entry, dict):
            raise CLIError(f"Scenario #{idx} in '{path}' must be a JSON object.")
        title = str(entry.get("title") or "").strip()
        question = str(entry.get("question") or "").strip()
        if not title or not question:
            raise CLIError(
                f"Scenario #{idx} in '{path}' requires non-empty 'title' and 'question'."
            )
        scenarios.append(
            DemoScenario(
                title=title,
                question=question,
                description=str(entry.get("description") or "").strip(),
                project=str(entry["project"]).strip() if entry.get("project") else None,
                building=str(entry["building"]).strip() if entry.get("building") else None,
                phase=str(entry["phase"]).strip() if entry.get("phase") else None,
                top_k=max(1, int(entry.get("top_k", 3))),
                min_score=(
                    float(entry["min_score"]) if entry.get("min_score") is not None else None
                ),
                expect_refusal=bool(entry.get("expect_refusal", False)),
            )
        )
    return scenarios


def format_source_name(source_path: str, fallback: str = "") -> str:
    """Extract a clean, human-readable filename from a SharePoint URL or nested path."""
    raw = (source_path or "").strip()
    if not raw:
        return fallback or "Unknown document"

    if "://" in raw:
        parsed = urlparse(raw)
        raw = parsed.path or raw

    decoded = unquote(raw).replace("\\", "/").rstrip("/")
    name = decoded.rsplit("/", 1)[-1].strip()
    return name or fallback or "Unknown document"


def format_location(page_numbers: Sequence[int], section_header: str | None) -> str:
    """Format PDF page ranges or Excel sheet references into a compact location string."""
    pages = sorted({int(p) for p in page_numbers})
    sec = clean_preview(section_header, max_chars=28) if section_header else ""
    if pages:
        page_label = f"p. {pages[0]}" if len(pages) == 1 else f"pp. {format_page_ranges(pages)}"
        return f"{page_label} · {sec}" if sec else page_label
    return sec or "—"


def format_metadata_badges(
    project: str | None,
    buildings: Sequence[str],
    phase: str | None,
) -> str:
    """Render concise Project / Building / Phase tags for the sources table."""
    parts: list[str] = []
    if project:
        parts.append(f"[cyan]{project}[/cyan]")
    if buildings:
        parts.append(f"[magenta]{', '.join(buildings[:2])}[/magenta]")
    if phase:
        parts.append(f"[yellow]{phase}[/yellow]")
    return " · ".join(parts) if parts else "[dim]—[/dim]"


def render_relevance_bar(score: float) -> str:
    """Render a 5-block relevance meter + percentage for non-technical viewers."""
    clamped = max(0.0, min(1.0, float(score)))
    filled = max(1, round(clamped * 5)) if clamped > 0 else 0
    bar = "█" * filled + "░" * (5 - filled)
    pct = f"{clamped * 100:3.0f}%"
    if clamped >= 0.70:
        return f"[bold green]{bar} {pct}[/bold green]"
    if clamped >= 0.40:
        return f"[yellow]{bar} {pct}[/yellow]"
    return f"[dim]{bar} {pct}[/dim]"


def format_answer_text(
    trace: RetrievalTrace,
    answer: Answer,
    *,
    using_llm: bool,
    max_excerpts: int = 3,
    max_chars_per_excerpt: int = 340,
) -> str:
    """Produce a clean stakeholder-facing answer for either Local LLM or Extractive mode."""
    if not trace.results:
        return (
            "No supporting evidence was found in the indexed project documents for this question.\n"
            f"[dim](Guardrail active: {trace.deduplicated_parents} candidate passages checked; "
            f"none met the minimum relevance threshold of {trace.min_score:.2f}.)[/dim]"
        )

    if using_llm:
        return answer.text.strip()

    # In grounded extractive mode (no local LLM configured), present clean numbered passages
    # without developer stub tags so stakeholders can read the exact cited evidence directly.
    blocks: list[str] = []
    for idx, res in enumerate(trace.results[:max_excerpts], start=1):
        p = res.parent_chunk
        fname = format_source_name(p.metadata.source_path, fallback=p.document_id)
        loc = format_location(p.page_numbers, p.section_header)
        excerpt_src = res.matched_child_text or p.text
        snippet = clean_preview(excerpt_src, max_chars=max_chars_per_excerpt)
        blocks.append(
            f"[bold cyan][{idx}][/bold cyan] [bold]{fname}[/bold] [dim]({loc})[/dim]\n{snippet}"
        )
    return "\n\n".join(blocks)


def _silence_library_noise(*, verbose: bool) -> None:
    """Keep the presentation terminal free of HF/Qdrant/tqdm/tokenizer chatter."""
    suppress_hf_noise_early()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TQDM_DISABLE"] = "1"
    if verbose:
        return

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    for noisy in (
        "FlagEmbedding",
        "transformers",
        "sentence_transformers",
        "huggingface_hub",
        "datasets",
        "qdrant_client",
        "httpx",
        "httpcore",
        "torch",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    try:
        import tqdm
        import tqdm.auto

        _orig_init = tqdm.tqdm.__init__

        def _quiet_tqdm_init(self: Any, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            _orig_init(self, *args, **kwargs)

        tqdm.tqdm.__init__ = _quiet_tqdm_init  # type: ignore[method-assign]
        tqdm.auto.tqdm = tqdm.tqdm  # type: ignore[assignment]
    except Exception:
        pass


def _release_mps_cache() -> None:
    """Free transient MPS/Python buffers between demo queries on 8 GB Apple Silicon."""
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _collect_kb_overview(repo: Any) -> dict[str, Any]:
    counts = repo.collection_counts()
    projects: Counter[str] = Counter()
    seen_docs: set[str] = set()
    for p in repo.iter_parents():
        if p.document_id not in seen_docs:
            seen_docs.add(p.document_id)
            if p.metadata.project:
                projects[p.metadata.project] += 1
    return {
        "documents": counts["documents"],
        "parents": counts["parent_chunks"],
        "children": counts["child_chunks"],
        "projects": dict(sorted(projects.items())),
    }


def _render_banner(
    console: Console,
    overview: dict[str, Any],
    *,
    device: str,
    answer_mode: str,
    qdrant_path: Path,
) -> None:
    proj_str = (
        ", ".join(f"{proj} ({cnt:,} docs)" for proj, cnt in overview["projects"].items())
        if overview["projects"]
        else "Unscoped local collection"
    )
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column()
    grid.add_row("Indexed Projects", proj_str)
    grid.add_row(
        "Knowledge Base",
        f"[bold green]{overview['documents']:,}[/bold green] documents  ·  "
        f"[bold]{overview['parents']:,}[/bold] sections  ·  "
        f"[bold]{overview['children']:,}[/bold] searchable passages",
    )
    grid.add_row("Answer Mode", answer_mode)
    grid.add_row("Engine", f"Hybrid Dense + Sparse (BGE-M3) + Reranker on [bold]{device}[/bold]")
    try:
        display_path = str(qdrant_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        display_path = str(qdrant_path)
    grid.add_row("Index Path", f"[dim]{display_path}[/dim]")

    console.print()
    console.print(
        Panel(
            grid,
            title="[bold white]Ingenium Engineering Knowledge Assistant — Live Demo[/bold white]",
            subtitle="[dim]Local Parent-Child Hybrid RAG · Zero External Cloud Calls[/dim]",
            border_style="bright_cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def _render_sources_table(trace: RetrievalTrace) -> Table:
    table = Table(
        title="Verified Sources",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=False,
        expand=False,
    )
    table.add_column("#", justify="right", style="bold cyan", no_wrap=True, min_width=1)
    table.add_column("Document", style="white", no_wrap=True, overflow="ellipsis", max_width=25)
    table.add_column("Scope", no_wrap=True, overflow="ellipsis", min_width=9, max_width=16)
    table.add_column("Page / Sheet", style="dim", no_wrap=True, overflow="ellipsis", max_width=16)
    table.add_column("Relevance", justify="right", no_wrap=True, min_width=10)

    for idx, res in enumerate(trace.results, start=1):
        p = res.parent_chunk
        m = p.metadata
        table.add_row(
            str(idx),
            format_source_name(m.source_path, fallback=p.document_id),
            format_metadata_badges(m.project, m.buildings, m.phase),
            format_location(p.page_numbers, p.section_header),
            render_relevance_bar(res.score),
        )
    return table


def _render_query_result(
    console: Console,
    *,
    question: str,
    filters: dict[str, str],
    trace: RetrievalTrace,
    answer: Answer,
    elapsed: float,
    using_llm: bool,
    header_label: str | None = None,
    description: str = "",
) -> None:
    if header_label:
        console.print()
        console.print(Rule(f"[bold cyan]{header_label}[/bold cyan]", style="cyan"))
        if description:
            console.print(f"[dim]{description}[/dim]")

    flt_parts = [f"{k}=[bold]{v}[/bold]" for k, v in filters.items()]
    flt_text = " · ".join(flt_parts) if flt_parts else "All indexed projects"
    q_line = Text()
    q_line.append("Question: ", style="bold white")
    q_line.append(question, style="bold bright_white")
    console.print()
    console.print(q_line)
    console.print(f"[dim]Scope:[/dim] {flt_text}")

    has_hits = bool(trace.results)
    panel_title = (
        "Answer (Synthesized)"
        if (has_hits and using_llm)
        else ("Grounded Evidence from Project Documents" if has_hits else "Grounded Refusal")
    )
    border = "green" if has_hits else "yellow"
    body = format_answer_text(trace, answer, using_llm=using_llm)

    console.print()
    console.print(
        Panel(body, title=panel_title, border_style=border, box=box.ROUNDED, padding=(1, 2))
    )

    if has_hits:
        console.print()
        console.print(_render_sources_table(trace))

    footer = (
        f"Found {len(trace.results)} verified source(s) in {format_duration(elapsed)}"
        if has_hits
        else f"Verified no matching evidence in {format_duration(elapsed)}"
    )
    console.print(f"[dim]{footer}[/dim]")


class _DemoEngine:
    """Manages pipeline lifecycle cleanly across warm-session and `--low-memory` modes."""

    def __init__(
        self,
        *,
        qdrant_path: Path,
        device: str,
        low_memory: bool,
        default_min_score: float,
    ) -> None:
        self.qdrant_path = qdrant_path
        self.device = device
        self.low_memory = low_memory
        self.default_min_score = default_min_score
        self._pipeline: Any = None
        self._repo: Any = None

    def open(self, console: Console) -> dict[str, Any]:
        from storage import LocalHybridRepository

        if self.low_memory:
            with LocalHybridRepository(path=str(self.qdrant_path)) as repo:
                return _collect_kb_overview(repo)

        from pipeline import build_rag_pipeline

        with console.status(
            f"[bold cyan]Loading search & reranking models on {self.device}...[/bold cyan]",
            spinner="dots",
        ):
            self._pipeline, self._repo = build_rag_pipeline(
                qdrant_path=self.qdrant_path,
                device=self.device,
                low_memory=False,
            )
            overview = _collect_kb_overview(self._repo)
        return overview

    def close(self) -> None:
        if self._repo is not None:
            self._repo.close()
            self._repo = None
            self._pipeline = None

    def ask(
        self,
        question: str,
        *,
        project: str | None = None,
        building: str | None = None,
        phase: str | None = None,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> tuple[RetrievalTrace, Answer, float]:
        eff_min = self.default_min_score if min_score is None else min_score
        t0 = time.perf_counter()

        if self.low_memory:
            from pipeline import build_rag_pipeline

            pipeline, repo = build_rag_pipeline(
                qdrant_path=self.qdrant_path,
                device=self.device,
                low_memory=True,
            )
            with repo:
                trace = pipeline.retrieve_with_trace(
                    question,
                    top_k=top_k,
                    project=project,
                    building=building,
                    phase=phase,
                    min_score=eff_min,
                )
                answer = pipeline.generator.generate(question, trace.results)
            del pipeline
            _release_mps_cache()
            return trace, answer, time.perf_counter() - t0

        assert self._pipeline is not None
        trace = self._pipeline.retrieve_with_trace(
            question,
            top_k=top_k,
            project=project,
            building=building,
            phase=phase,
            min_score=eff_min,
        )
        answer = self._pipeline.generator.generate(question, trace.results)
        _release_mps_cache()
        return trace, answer, time.perf_counter() - t0


def _run_interactive_loop(
    console: Console,
    engine: _DemoEngine,
    *,
    known_projects: list[str],
    using_llm: bool,
    top_k: int,
) -> None:
    console.print()
    console.print(Rule("[bold cyan]Live Interactive Q&A[/bold cyan]", style="cyan"))
    console.print(
        "[dim]Ask any question in Dutch or English. Commands: "
        "[bold]/project <id>[/bold] · [bold]/building <name>[/bold] · "
        "[bold]/phase <name>[/bold] · [bold]/clear[/bold] · "
        "[bold]/projects[/bold] · [bold]/quit[/bold][/dim]"
    )

    filters: dict[str, str] = {}
    while True:
        scope_tag = f" [{', '.join(f'{k}={v}' for k, v in filters.items())}]" if filters else ""
        try:
            console.print()
            raw = console.input(f"[bold cyan]Ask{scope_tag} › [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Ending live Q&A session.[/dim]")
            return

        if not raw:
            continue
        low = raw.lower()
        if low in {"/quit", "/exit", "quit", "exit"}:
            console.print("[dim]Demo finished. Goodbye![/dim]")
            return
        if low == "/clear":
            filters.clear()
            console.print("[green]✓[/green] Cleared all active filters.")
            continue
        if low == "/projects":
            proj_list = ", ".join(known_projects) if known_projects else "(none)"
            console.print(f"[cyan]Indexed projects:[/cyan] {proj_list}")
            continue
        if low.startswith(("/project ", "/building ", "/phase ")):
            cmd, _, val = raw.partition(" ")
            key = cmd.lstrip("/").lower()
            val = val.strip()
            if val:
                filters[key] = val
                console.print(f"[green]✓[/green] Filter set: {key}=[bold]{val}[/bold]")
            else:
                filters.pop(key, None)
                console.print(f"[green]✓[/green] Cleared {key} filter.")
            continue

        with console.status(
            "[bold cyan]Searching project documents...[/bold cyan]", spinner="dots"
        ):
            trace, answer, elapsed = engine.ask(
                raw,
                project=filters.get("project"),
                building=filters.get("building"),
                phase=filters.get("phase"),
                top_k=top_k,
            )

        _render_query_result(
            console,
            question=raw,
            filters=dict(filters),
            trace=trace,
            answer=answer,
            elapsed=elapsed,
            using_llm=using_llm,
        )


def _main() -> None:
    parser = build_parser(
        "Stakeholder-ready live presentation & interactive Q&A demo for Ingenium RAG.",
        epilog=(
            "Examples:\n"
            "  make demo                                  # slide-style walkthrough + live Q&A\n"
            "  make demo AUTO=1                           # run scripted scenarios non-stop\n"
            "  make demo LIVE=1                           # jump straight to interactive Q&A\n"
            "  make demo SCENARIOS=data/my_demo.json      # custom scenario deck"
        ),
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=DEFAULT_SCENARIOS_PATH,
        help=f"JSON file of scripted demo scenarios (default: {DEFAULT_SCENARIOS_PATH}).",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=None,
        help="Local Qdrant storage directory (defaults to Settings.qdrant_path).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device override (cuda / mps / cpu).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_DEMO_MIN_SCORE,
        help=f"Relevance cutoff for grounded refusal guardrail (default: {DEFAULT_DEMO_MIN_SCORE}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Default number of verified sources per interactive question (default: 3).",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--pause",
        action="store_true",
        default=True,
        help="Wait for Enter between scripted scenarios (default).",
    )
    mode_group.add_argument(
        "--auto",
        action="store_true",
        help="Run all scripted scenarios straight through without pausing or interactive loop.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Skip scripted scenarios and jump directly into interactive Q&A mode.",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Reload embedder/reranker sequentially per query to minimize peak RAM.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Keep third-party logs enabled and show full tracebacks on error.",
    )
    args = parser.parse_args()

    _silence_library_noise(verbose=args.verbose)
    settings = get_settings()
    qdrant_path = args.qdrant_path or settings.qdrant_path
    ensure_qdrant_ready(qdrant_path, require_docs=True)

    scenarios: list[DemoScenario] = []
    if not args.live:
        scenarios = load_demo_scenarios(args.scenarios)

    console = Console(file=sys.__stdout__, highlight=False)
    device = detect_device(args.device or settings.device)
    using_llm = bool(settings.llm_base_url and settings.llm_model)
    answer_mode = (
        f"[bold green]Local LLM[/bold green] ({settings.llm_model})"
        if using_llm
        else "[bold cyan]Grounded Extractive Mode[/bold cyan] [dim](verbatim cited passages)[/dim]"
    )

    engine = _DemoEngine(
        qdrant_path=qdrant_path,
        device=device,
        low_memory=args.low_memory,
        default_min_score=args.min_score,
    )

    try:
        t_warm = time.perf_counter()
        overview = engine.open(console)
        warm_elapsed = time.perf_counter() - t_warm

        _render_banner(
            console,
            overview,
            device=device,
            answer_mode=answer_mode,
            qdrant_path=qdrant_path,
        )
        if not args.low_memory:
            console.print(f"[dim]Search engine warmed up in {format_duration(warm_elapsed)}.[/dim]")

        interactive_tty = sys.stdin.isatty() and not args.auto
        for idx, sc in enumerate(scenarios, start=1):
            if interactive_tty:
                try:
                    console.print()
                    console.input(
                        f"[dim]Press Enter to run Demo {idx}/{len(scenarios)} ({sc.title})...[/dim]"
                    )
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[dim]Demo ended early.[/dim]")
                    return

            filters = {
                k: v
                for k, v in (
                    ("project", sc.project),
                    ("building", sc.building),
                    ("phase", sc.phase),
                )
                if v
            }
            with console.status(
                "[bold cyan]Searching project documents...[/bold cyan]", spinner="dots"
            ):
                trace, answer, elapsed = engine.ask(
                    sc.question,
                    project=sc.project,
                    building=sc.building,
                    phase=sc.phase,
                    top_k=sc.top_k,
                    min_score=sc.min_score,
                )

            _render_query_result(
                console,
                question=sc.question,
                filters=filters,
                trace=trace,
                answer=answer,
                elapsed=elapsed,
                using_llm=using_llm,
                header_label=f"Demo {idx}/{len(scenarios)}: {sc.title}",
                description=sc.description,
            )

        if args.live or (not args.auto and sys.stdin.isatty()):
            _run_interactive_loop(
                console,
                engine,
                known_projects=list(overview["projects"].keys()),
                using_llm=using_llm,
                top_k=args.top_k,
            )
    finally:
        engine.close()


def main() -> None:
    run_cli(_main)


if __name__ == "__main__":
    main()
