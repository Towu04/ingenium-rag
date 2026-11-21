"""Shared CLI formatting, stdout/stderr consoles, common flags, and friendly error handling."""

import argparse
import logging
import os
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from config import configure_logging

# Strict stdout vs stderr separation: primary output -> stdout, logs/status/errors -> stderr.
out_console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)


class CLIError(RuntimeError):
    """Expected user or environment error with an actionable CLI hint."""


def build_parser(description: str, *, epilog: str | None = None) -> argparse.ArgumentParser:
    """Create an `ArgumentParser` that preserves epilog formatting and shows defaults."""
    return argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_qdrant_path: Path | None = None,
    include_qdrant: bool = True,
    include_json: bool = True,
) -> None:
    """Register standardized verbosity, JSON output, and Qdrant path flags."""
    if include_qdrant and default_qdrant_path is not None:
        parser.add_argument(
            "--qdrant-path",
            type=Path,
            default=default_qdrant_path,
            help=f"Local Qdrant storage directory (default: {default_qdrant_path}).",
        )
    if include_json:
        parser.add_argument(
            "--json",
            dest="as_json",
            action="store_true",
            help="Emit machine-readable JSON to stdout (silences logs/progress on stderr).",
        )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logs, third-party diagnostics, and full error tracebacks.",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress informational output and non-error logs.",
    )


def clean_preview(text: str, max_chars: int = 360) -> str:
    """Normalize whitespace and truncate text cleanly on a word/line boundary."""
    cleaned = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{clipped or cleaned[:max_chars]}..."


def format_friendly_error(exc: BaseException) -> str:
    """Map expected user/environment exceptions to concise, actionable messages."""
    if isinstance(exc, CLIError):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        target = exc.filename or str(exc)
        return f"File or directory not found: {target}"

    msg = str(exc)
    lower = msg.lower()

    if (
        "already accessed by another instance" in lower
        or "resource temporarily unavailable" in lower
    ):
        return (
            "Local Qdrant storage is locked by another process. "
            "Close any running API/CLI command using the same --qdrant-path and retry."
        )
    if "missing required sharepoint/entra credentials" in lower:
        return f"{msg}\nAdd the missing keys to your .env file (see .env.example)."
    if (
        "localentrynotfounderror" in type(exc).__name__.lower()
        or "cannot find the requested files in the local cache" in lower
        or ("offline" in lower and "huggingface" in lower)
    ):
        return (
            "HuggingFace model weights are not in the local cache. "
            "Run 'make download-models' (or unset HF_HUB_OFFLINE) and retry."
        )
    if isinstance(exc, ValueError):
        return f"Invalid input: {msg}"
    return f"{type(exc).__name__}: {msg}"


def suppress_hf_noise_early() -> None:
    """Disable noisy tqdm/HF progress bars unless verbose mode is requested on sys.argv."""
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    if not verbose:
        os.environ.setdefault("TQDM_DISABLE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def print_kv_table(title: str, rows: Sequence[tuple[str, object]]) -> None:
    """Print a compact two-column summary table to stdout."""
    tbl = Table(title=title, header_style="bold cyan")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right", style="green")
    for k, v in rows:
        tbl.add_row(k, str(v))
    out_console.print(tbl)


def ensure_qdrant_ready(qdrant_path: Path, *, require_docs: bool = True) -> None:
    """Fail fast with a friendly hint if the local Qdrant index is missing or empty."""
    if not qdrant_path.exists():
        raise CLIError(f"Qdrant index not found at '{qdrant_path}'. Run 'make ingest' first.")
    if not require_docs:
        return
    from storage import LocalHybridRepository

    with LocalHybridRepository(path=str(qdrant_path)) as repo:
        if repo.collection_counts()["documents"] == 0:
            raise CLIError(
                f"Qdrant index at '{qdrant_path}' is empty (0 documents). Run 'make ingest' first."
            )


def setup_cli_logging(
    *,
    verbose: bool = False,
    quiet: bool = False,
    json_mode: bool = False,
    default_level: int = logging.WARNING,
    disable_progress: bool | None = None,
) -> None:
    """Configure stderr logging routed through `err_console` so logs never tear progress bars."""
    handler: logging.Handler | None = None
    if not quiet and not json_mode:
        handler = RichHandler(
            console=err_console,
            show_path=False,
            markup=False,
            rich_tracebacks=False,
            log_time_format="[%H:%M:%S]",
        )
    configure_logging(
        verbose=verbose,
        quiet=quiet,
        json_mode=json_mode,
        default_level=default_level,
        disable_progress=disable_progress,
        handler=handler,
    )


class PipelineProgress:
    """Stderr-only stage milestone and live progress-bar reporter (auto-disabled on non-TTY/quiet/JSON)."""

    def __init__(self, *, quiet: bool = False, json_mode: bool = False) -> None:
        self.enabled = not (quiet or json_mode)
        self.show_bars = self.enabled and sys.stderr.isatty()
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._base_label: str = ""

    def __enter__(self) -> "PipelineProgress":
        if self.show_bars:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=err_console,
                transient=True,
            )
            self._progress.start()
        return self

    def __exit__(self, *args: object) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    def __call__(self, event: str, current: int, total: int, detail: str) -> None:
        if not self.enabled:
            return
        if event == "stage_start":
            err_console.print(f"[bold cyan][{current}/{total}][/bold cyan] {detail}")
        elif event == "stage_done":
            err_console.print(f"[green]✓[/green] {detail}")
        elif event == "model_load_start":
            err_console.print(f"[dim]{detail}[/dim]")
        elif event == "model_load_done":
            err_console.print(f"[green]✓[/green] [dim]{detail}[/dim]")
        elif event == "bar_start" and self._progress is not None:
            self._base_label = detail
            self._task_id = self._progress.add_task(detail, total=max(total, 1))
        elif event == "bar_advance" and self._progress is not None and self._task_id is not None:
            short = detail if len(detail) <= 38 else f"{detail[:35]}..."
            desc = f"{self._base_label} · {short}" if short else self._base_label
            self._progress.update(self._task_id, completed=current, description=desc)
        elif event == "bar_done" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=total)
            self._progress.remove_task(self._task_id)
            self._task_id = None


def run_cli(main_fn: Callable[[], None]) -> None:
    """Execute a CLI entry point with friendly error handling (full traceback with -v/--verbose)."""
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    try:
        main_fn()
    except KeyboardInterrupt:
        err_console.print("[yellow]Interrupted.[/yellow]")
        raise SystemExit(130) from None
    except SystemExit:
        raise
    except Exception as exc:
        if verbose:
            raise
        err_console.print(f"[bold red]Error:[/bold red] {format_friendly_error(exc)}")
        err_console.print("[dim]Re-run with -v/--verbose for the full traceback.[/dim]")
        raise SystemExit(1) from None
