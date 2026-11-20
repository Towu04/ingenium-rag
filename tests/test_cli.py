"""Unit tests for centralized logging hygiene and shared CLI ergonomics."""

import logging
import sys

import pytest

from config import configure_logging
from scripts._cli import CLIError, clean_preview, format_friendly_error, run_cli


def test_configure_logging_uses_stderr_no_duplicates_and_silences_third_party() -> None:
    configure_logging(verbose=False, quiet=False)
    configure_logging(verbose=False, quiet=False)
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)
    assert root.handlers[0].stream is sys.stderr
    assert logging.getLogger("transformers").level == logging.ERROR
    assert logging.getLogger("httpx").level == logging.ERROR

    configure_logging(verbose=True)
    assert root.level == logging.DEBUG
    assert logging.getLogger("transformers").level == logging.INFO

    configure_logging(json_mode=True)
    assert root.level == logging.ERROR


def test_clean_preview_truncates_on_word_boundary() -> None:
    text = "Brandwerende   deuren\n\n\nmoeten voldoen aan EI1 30 volgens KB 07.12.2016."
    short = clean_preview(text, max_chars=36)
    assert "\n\n\n" not in short
    assert short.endswith("...")
    assert len(short) <= 40


def test_format_friendly_error_translates_common_environment_errors() -> None:
    assert "make ingest" in format_friendly_error(CLIError("Run 'make ingest' first."))
    assert "not found" in format_friendly_error(
        FileNotFoundError(2, "No such file", "missing.json")
    )
    assert "locked by another process" in format_friendly_error(
        RuntimeError("Storage folder data/qdrant_poc is already accessed by another instance")
    )
    assert "make download-models" in format_friendly_error(
        OSError("Cannot find the requested files in the local cache")
    )
    assert ".env" in format_friendly_error(
        ValueError(
            "Missing required SharePoint/Entra credentials in .env or environment: AZURE_TENANT_ID"
        )
    )


def test_run_cli_exits_cleanly_without_traceback_unless_verbose(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom() -> None:
        raise CLIError("Index is empty. Run 'make ingest'.")

    monkeypatch.setattr(sys, "argv", ["ingenium-query"])
    with pytest.raises(SystemExit) as exc_info:
        run_cli(_boom)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Index is empty" in captured.err

    monkeypatch.setattr(sys, "argv", ["ingenium-query", "--verbose"])
    with pytest.raises(CLIError, match="Index is empty"):
        run_cli(_boom)
