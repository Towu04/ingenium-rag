"""Shared CLI formatting, stdout/stderr consoles, common flags, and friendly error handling."""

import argparse
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

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
