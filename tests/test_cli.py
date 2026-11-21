"""Unit tests for centralized logging hygiene and shared CLI ergonomics."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from config import configure_logging
from domain import ChildChunk, DocumentMetadata, EmbeddedChildBatch, ParentChunk, ParsedDoc
from scripts._cli import (
    CLIError,
    PipelineProgress,
    clean_preview,
    format_friendly_error,
    run_cli,
)
from storage import LocalHybridRepository


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


def _cli_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src:."
    env["HF_HUB_OFFLINE"] = "1"
    return env


def test_cli_json_stdout_is_pure_and_parseable(tmp_path: Path) -> None:
    import numpy as np

    qdrant_dir = tmp_path / "qdrant"
    meta = DocumentMetadata(
        source_path="General/05_Uitvoering/Spec.pdf",
        project="PRJ-100",
        building="Blok A",
        phase="Uitvoering",
    )
    doc = ParsedDoc(document_id="doc1", content_hash="h1", markdown="Test", metadata=meta)
    parent = ParentChunk(
        id="doc1:p0",
        document_id="doc1",
        parent_index=0,
        text="Test parent",
        child_ids=["doc1:p0:c0"],
        metadata=meta,
    )
    child = ChildChunk(
        id="doc1:p0:c0",
        parent_id="doc1:p0",
        document_id="doc1",
        child_index=0,
        text="Test child",
        metadata=meta,
    )
    batch = EmbeddedChildBatch(
        dense_vecs=np.ones((1, 1024), dtype=np.float32),
        sparse_weights=[{1: 1.0}],
    )
    with LocalHybridRepository(path=str(qdrant_dir)) as repo:
        repo.upsert_document_bundle(doc, [parent], [child], batch)

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_rag.py",
            "--qdrant-path",
            str(qdrant_dir),
            "--json",
            "stats",
        ],
        capture_output=True,
        text=True,
        env=_cli_env(),
        check=False,
    )
    assert proc.returncode == 0
    parsed = json.loads(proc.stdout)
    assert parsed["collections"] == {"documents": 1, "parent_chunks": 1, "child_chunks": 1}
    assert parsed["metadata_distribution"]["projects"] == {"PRJ-100": 1}


def test_cli_friendly_errors_return_nonzero_without_traceback(tmp_path: Path) -> None:
    missing_qdrant = tmp_path / "does_not_exist"
    proc = subprocess.run(
        [sys.executable, "scripts/query.py", "--qdrant-path", str(missing_qdrant), "test query"],
        capture_output=True,
        text=True,
        env=_cli_env(),
        check=False,
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "make ingest" in proc.stderr
    assert "Traceback" not in proc.stderr

    proc_eval = subprocess.run(
        [
            sys.executable,
            "scripts/eval.py",
            "--qdrant-path",
            str(missing_qdrant),
            "--dataset",
            str(tmp_path / "missing.json"),
        ],
        capture_output=True,
        text=True,
        env=_cli_env(),
        check=False,
    )
    assert proc_eval.returncode == 1
    assert proc_eval.stdout == ""
    assert "Traceback" not in proc_eval.stderr


def test_pipeline_progress_disables_bars_on_non_tty_quiet_and_json() -> None:
    with PipelineProgress(quiet=True) as p_quiet:
        assert p_quiet.enabled is False
        assert p_quiet.show_bars is False
        p_quiet("stage_start", 1, 4, "Test")

    with PipelineProgress(json_mode=True) as p_json:
        assert p_json.enabled is False
        assert p_json.show_bars is False
        p_json("stage_done", 1, 4, "Done")
