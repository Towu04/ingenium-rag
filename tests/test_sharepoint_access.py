"""Unit and End-to-End tests for Ephemeral SharePoint Streaming (`sharepoint_access.py`).

Verifies:
  1. MSAL authentication and Microsoft Graph recursive directory traversal (metadata only).
  2. Single-file download and seeding of `data/processed/json/<doc_id>.json` with SharePoint provenance.
  3. Two-pass streaming in `ETLPipeline.run_from_sharepoint`:
     - Pass 1 loads Document Registers first, enriches JSON in-place, and deletes the raw `.xlsx`.
     - Pass 2 downloads one file at a time, updates the seeded `.json` in-place, writes `.md`,
       and deletes the local raw `.pdf`/`.docx`/`.xlsx` immediately after processing.
     - Pre-download manifest check skips unchanged files by `eTag` / `quickXorHash` without downloading.
  4. `try ... finally` cleanup guarantees corrupted files never leak local disk space.
  5. CLI `inventory --from-sharepoint` and `run --from-sharepoint` commands.
"""

import io
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import docx
import openpyxl
import pymupdf as fitz
import pytest
from typer.testing import CliRunner

from etl.cli import app
from etl.metadata_matcher import compute_doc_id
from etl.pipeline import ETLPipeline
from etl.schemas import DocCategory, ExtractedDocument
from etl.sharepoint_access import (
    SharePointConfig,
    SharePointConnector,
)

runner = CliRunner()


# ===========================================================================
# Synthetic Binary Builders for Mocked Graph Downloads
# ===========================================================================


def _build_register_xlsx_bytes() -> bytes:
    """Create an in-memory Excel Document Register (Documentenlijst) mapping sibling files."""
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Documentenlijst"
    ws.append(["Doc_Nr", "Bestandsnaam", "Omschrijving", "Gebouw", "Fase", "Revisie", "Type"])
    ws.append([
        "ING-104-HVAC-01",
        "Bestek_HVAC_Ventilatie.pdf",
        "Technisch Bestek Luchtbehandeling OK",
        "Blok C - Intensieve Zorgen",
        "Definitief Ontwerp",
        "Rev.B",
        "Bestek",
    ])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _build_pdf_bytes() -> bytes:
    """Create a 2-page digital PDF specification with a table in memory."""
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 70),
        "1. Algemene Voorwaarden\n"
        "Technische specificatie voor de luchtbehandeling van AZ Sint-Jan Brugge.",
        fontsize=11,
    )
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text(
        (50, 60),
        "3.2 Luchtgroepen (Ventilation Units)\n"
        "Overzicht van de luchtgroepen en debieten:",
        fontsize=11,
    )
    x_cols = [50, 160, 290, 430]
    y_rows = [110, 135, 160]
    for y in y_rows:
        p2.draw_line((x_cols[0], y), (x_cols[-1], y))
    for x in x_cols:
        p2.draw_line((x, y_rows[0]), (x, y_rows[-1]))
    rows = [
        ["Unit ID", "Building", "Airflow (m3/h)"],
        ["LBK-IZ-01", "Blok C", "14500"],
    ]
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            p2.insert_text((x_cols[c_i] + 5, y_rows[r_i] + 16), val, fontsize=10)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _build_docx_bytes() -> bytes:
    """Create an in-memory Word (.docx) meeting report."""
    d = docx.Document()
    d.add_heading("1. Werfvergadering HVAC", level=1)
    d.add_paragraph("Bespreking vordering kanaalmontage in Blok B Operatiekwartier.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


class _FakeGraphResponse:
    """Mock HTTP response for Microsoft Graph API calls."""

    def __init__(self, json_data: Any = None, content_bytes: bytes = b"", status_code: int = 200) -> None:
        self._json_data = json_data
        self._content_bytes = content_bytes
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._json_data

    def iter_content(self, chunk_size: int = 65536):
        stream = io.BytesIO(self._content_bytes)
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            yield chunk


def _create_mock_graph_session(
    file_payloads: Dict[str, bytes],
    download_history: List[str],
    staging_dir: Path,
    max_concurrent_files_tracker: List[int],
) -> MagicMock:
    """Build a mocked `requests.Session` simulating Microsoft Graph site/drive traversal and file streaming."""
    session = MagicMock()

    def fake_get(url: str, headers: Any = None, stream: bool = False, timeout: int = 60) -> _FakeGraphResponse:
        # 1. Site resolution
        if "/sites/ingenium.sharepoint.com:/sites/AZSintJan" in url:
            return _FakeGraphResponse({"id": "site-az-sint-jan-001"})

        # 2. Drives listing
        if url.endswith("/sites/site-az-sint-jan-001/drives"):
            return _FakeGraphResponse({
                "value": [{"id": "drive-docs-001", "name": "Documents"}]
            })

        # 3. Root children listing
        if url.endswith("/drives/drive-docs-001/root/children"):
            return _FakeGraphResponse({
                "value": [
                    {
                        "id": "folder-prj-104",
                        "name": "PRJ_104_AZ_Sint_Jan",
                        "folder": {"childCount": 3},
                    }
                ]
            })

        # 4. Project subfolder children listing
        if url.endswith("/drives/drive-docs-001/items/folder-prj-104/children"):
            items = [
                {
                    "id": "item-reg-001",
                    "name": "00_Documentenlijst_HVAC.xlsx",
                    "size": len(file_payloads.get("item-reg-001", b"")),
                    "eTag": "etag-reg-v1",
                    "cTag": "ctag-reg-v1",
                    "webUrl": "https://ingenium.sharepoint.com/sites/AZSintJan/PRJ_104_AZ_Sint_Jan/00_Documentenlijst_HVAC.xlsx",
                    "parentReference": {
                        "path": "/drives/drive-docs-001/root:/PRJ_104_AZ_Sint_Jan"
                    },
                    "file": {"hashes": {"quickXorHash": "qxor-reg-001"}},
                },
                {
                    "id": "item-pdf-002",
                    "name": "Bestek_HVAC_Ventilatie.pdf",
                    "size": len(file_payloads.get("item-pdf-002", b"")),
                    "eTag": "etag-pdf-v1",
                    "cTag": "ctag-pdf-v1",
                    "webUrl": "https://ingenium.sharepoint.com/sites/AZSintJan/PRJ_104_AZ_Sint_Jan/02_Aanbesteding/Blok_B_Operatiekwartier/Bestek_HVAC_Ventilatie.pdf",
                    "parentReference": {
                        "path": "/drives/drive-docs-001/root:/PRJ_104_AZ_Sint_Jan/02_Aanbesteding/Blok_B_Operatiekwartier"
                    },
                    "file": {"hashes": {"quickXorHash": "qxor-pdf-002"}},
                },
                {
                    "id": "item-docx-003",
                    "name": "Werfverslag_01_HVAC.docx",
                    "size": len(file_payloads.get("item-docx-003", b"")),
                    "eTag": "etag-docx-v1",
                    "cTag": "ctag-docx-v1",
                    "webUrl": "https://ingenium.sharepoint.com/sites/AZSintJan/PRJ_104_AZ_Sint_Jan/04_Uitvoering/Blok_B_Operatiekwartier/Werfverslag_01_HVAC.docx",
                    "parentReference": {
                        "path": "/drives/drive-docs-001/root:/PRJ_104_AZ_Sint_Jan/04_Uitvoering/Blok_B_Operatiekwartier"
                    },
                    "file": {"hashes": {"quickXorHash": "qxor-docx-003"}},
                },
            ]
            if "item-corrupt-004" in file_payloads:
                items.append({
                    "id": "item-corrupt-004",
                    "name": "Corrupt_Schema.pdf",
                    "size": len(file_payloads["item-corrupt-004"]),
                    "eTag": "etag-corrupt-v1",
                    "webUrl": "https://ingenium.sharepoint.com/sites/AZSintJan/PRJ_104_AZ_Sint_Jan/Corrupt_Schema.pdf",
                    "parentReference": {
                        "path": "/drives/drive-docs-001/root:/PRJ_104_AZ_Sint_Jan"
                    },
                    "file": {"hashes": {"quickXorHash": "qxor-corrupt-004"}},
                })
            return _FakeGraphResponse({"value": items})

        # 5. Single-file content download
        for item_id, payload in file_payloads.items():
            if url.endswith(f"/items/{item_id}/content"):
                # Record how many raw files currently exist on disk BEFORE this download starts
                if staging_dir.exists():
                    existing_files = [p for p in staging_dir.rglob("*") if p.is_file()]
                    max_concurrent_files_tracker.append(len(existing_files))
                download_history.append(item_id)
                return _FakeGraphResponse(content_bytes=payload)

        raise AssertionError(f"Unexpected Graph API URL: {url}")

    session.get.side_effect = fake_get
    return session


# ===========================================================================
# Unit & E2E SharePoint Streaming Tests
# ===========================================================================


def _mock_msal_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock msal.ConfidentialClientApplication so unit tests never make live OIDC discovery requests."""
    mock_app = MagicMock()
    mock_app.acquire_token_for_client.return_value = {"access_token": "mock-graph-token"}
    mock_cls = MagicMock(return_value=mock_app)
    monkeypatch.setattr("msal.ConfidentialClientApplication", mock_cls)
    return mock_cls


def test_sharepoint_connector_seeds_json_and_deletes_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify SharePointConnector downloads one file, seeds <doc_id>.json with SharePoint metadata, and deletes the raw file."""
    _mock_msal_client(monkeypatch)

    staging_dir = tmp_path / "raw_sharepoint"
    json_dir = tmp_path / "processed" / "json"
    pdf_bytes = _build_pdf_bytes()

    download_history: List[str] = []
    concurrent_tracker: List[int] = []
    session = _create_mock_graph_session(
        file_payloads={
            "item-reg-001": _build_register_xlsx_bytes(),
            "item-pdf-002": pdf_bytes,
            "item-docx-003": _build_docx_bytes(),
        },
        download_history=download_history,
        staging_dir=staging_dir,
        max_concurrent_files_tracker=concurrent_tracker,
    )

    sp_config = SharePointConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret-789",
        hostname="ingenium.sharepoint.com",
        site_path="/sites/AZSintJan",
        drive_name="Documents",
        staging_dir=staging_dir,
        json_out_dir=json_dir,
    )
    connector = SharePointConnector(sp_config=sp_config, session=session)

    # 1. Remote Tree Discovery (Metadata Only — 0 files downloaded)
    remote_files = connector.list_remote_files(supported_only=True)
    assert len(remote_files) == 3
    assert len(download_history) == 0

    pdf_item = next(i for i in remote_files if i.item_id == "item-pdf-002")
    assert (
        pdf_item.relative_path
        == "PRJ_104_AZ_Sint_Jan/02_Aanbesteding/Blok_B_Operatiekwartier/Bestek_HVAC_Ventilatie.pdf"
    )
    assert pdf_item.etag == "etag-pdf-v1"
    assert pdf_item.quick_xor_hash == "qxor-pdf-002"

    # 2. Download single file & seed target JSON
    local_path, seeded_json_path, file_hash = connector.download_and_seed(pdf_item)
    assert local_path.exists()
    assert seeded_json_path.exists()
    expected_doc_id = compute_doc_id(pdf_item.relative_path)
    assert seeded_json_path.name == f"{expected_doc_id}.json"

    # Verify seeded JSON contains SharePoint location/provenance metadata and empty blocks
    raw_seeded_text = seeded_json_path.read_text(encoding="utf-8")
    assert "full_markdown_with_anchors" not in raw_seeded_text
    seeded_doc = ExtractedDocument.model_validate_json(raw_seeded_text)
    assert seeded_doc.metadata.doc_id == expected_doc_id
    assert seeded_doc.metadata.file_name == "Bestek_HVAC_Ventilatie.pdf"
    assert seeded_doc.metadata.file_hash_sha256 == file_hash
    assert seeded_doc.metadata.sharepoint_item_id == "item-pdf-002"
    assert seeded_doc.metadata.sharepoint_etag == "etag-pdf-v1"
    assert (
        seeded_doc.metadata.sharepoint_url
        == "https://ingenium.sharepoint.com/sites/AZSintJan/PRJ_104_AZ_Sint_Jan/02_Aanbesteding/Blok_B_Operatiekwartier/Bestek_HVAC_Ventilatie.pdf"
    )
    assert (
        seeded_doc.metadata.relative_folder_path
        == "PRJ_104_AZ_Sint_Jan/02_Aanbesteding/Blok_B_Operatiekwartier"
    )
    assert seeded_doc.blocks == []

    # 3. Cleanup local raw file and verify empty staging directories are pruned
    connector.cleanup_local_file(local_path)
    assert not local_path.exists()
    assert not any(p.is_file() for p in staging_dir.rglob("*"))


def test_pipeline_sharepoint_streaming_in_place_json_and_ephemeral_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify full two-pass SharePoint streaming, in-place JSON enrichment, and immediate file deletion."""
    _mock_msal_client(monkeypatch)

    staging_dir = tmp_path / "raw_sharepoint"
    md_dir = tmp_path / "processed" / "markdown"
    json_dir = tmp_path / "processed" / "json"
    manifest_path = tmp_path / "processed" / "manifest.json"
    limitations_path = tmp_path / "logs" / "limitations_report.jsonl"

    catalog_csv = tmp_path / "projects_catalog.csv"
    catalog_csv.write_text(
        "project_id;project_name;folder_name;sharepoint_url\n"
        "PRJ-2021-104;AZ Sint-Jan Brugge - Renovatie OK;PRJ_104_AZ_Sint_Jan;"
        "https://ingenium.sharepoint.com/sites/AZSintJan/PRJ_104_AZ_Sint_Jan\n",
        encoding="utf-8",
    )

    download_history: List[str] = []
    concurrent_tracker: List[int] = []
    session = _create_mock_graph_session(
        file_payloads={
            "item-reg-001": _build_register_xlsx_bytes(),
            "item-pdf-002": _build_pdf_bytes(),
            "item-docx-003": _build_docx_bytes(),
        },
        download_history=download_history,
        staging_dir=staging_dir,
        max_concurrent_files_tracker=concurrent_tracker,
    )

    sp_config = SharePointConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret-789",
        hostname="ingenium.sharepoint.com",
        site_path="/sites/AZSintJan",
        drive_name="Documents",
        staging_dir=staging_dir,
        json_out_dir=json_dir,
    )
    connector = SharePointConnector(sp_config=sp_config, session=session)

    pipeline = ETLPipeline(
        catalog_path=catalog_csv,
        markdown_out_dir=md_dir,
        json_out_dir=json_dir,
        manifest_path=manifest_path,
        limitations_report_path=limitations_path,
    )

    # First run: streams all 3 files one by one
    run1 = pipeline.run_from_sharepoint(connector=connector)
    assert run1.discovered_files == 3
    assert run1.processed_files == 3
    assert run1.skipped_unchanged_files == 0
    assert run1.failed_files == 0

    # Pass 1 must have downloaded the Excel Document Register first!
    assert download_history == ["item-reg-001", "item-pdf-002", "item-docx-003"]

    # Every previous raw file was deleted BEFORE the next file was downloaded
    assert all(count == 0 for count in concurrent_tracker)
    # And after completion, zero raw files exist on local disk
    assert not staging_dir.exists() or not any(staging_dir.rglob("*"))

    # Verify exactly 3 .json and 3 .md files exist (no duplicate seeded vs enriched JSON files)
    json_files = sorted(json_dir.glob("*.json"))
    md_files = sorted(md_dir.glob("*.md"))
    assert len(json_files) == 3
    assert len(md_files) == 3

    # Verify the PDF's seeded JSON was updated IN-PLACE with both SharePoint provenance,
    # Layer 1 Register overrides (Blok C - Intensieve Zorgen & Definitief Ontwerp), and extracted blocks!
    pdf_rel_path = (
        "PRJ_104_AZ_Sint_Jan/02_Aanbesteding/Blok_B_Operatiekwartier/Bestek_HVAC_Ventilatie.pdf"
    )
    pdf_doc_id = compute_doc_id(pdf_rel_path)
    pdf_json_path = json_dir / f"{pdf_doc_id}.json"
    assert pdf_json_path.exists()

    raw_pdf_json = pdf_json_path.read_text(encoding="utf-8")
    assert "full_markdown_with_anchors" not in raw_pdf_json
    enriched_pdf_doc = ExtractedDocument.model_validate_json(raw_pdf_json)

    # Preserved from Step A (SharePoint seed):
    assert enriched_pdf_doc.metadata.doc_id == pdf_doc_id
    assert enriched_pdf_doc.metadata.sharepoint_item_id == "item-pdf-002"
    assert enriched_pdf_doc.metadata.sharepoint_etag == "etag-pdf-v1"
    assert (
        enriched_pdf_doc.metadata.sharepoint_url
        == "https://ingenium.sharepoint.com/sites/AZSintJan/PRJ_104_AZ_Sint_Jan/02_Aanbesteding/Blok_B_Operatiekwartier/Bestek_HVAC_Ventilatie.pdf"
    )
    # Enriched in Step B (Layer 2 Catalog + Layer 1 Document Register + PdfExtractor blocks):
    assert enriched_pdf_doc.metadata.project_id == "PRJ-2021-104"
    assert enriched_pdf_doc.metadata.building == "Blok C - Intensieve Zorgen"
    assert enriched_pdf_doc.metadata.phase == "Definitief Ontwerp (Final Design)"
    assert enriched_pdf_doc.metadata.doc_category == DocCategory.SPECIFICATION
    assert len(enriched_pdf_doc.blocks) >= 2
    assert any(b.block_type == "table" for b in enriched_pdf_doc.blocks)

    # Second run: Pre-download manifest check skips all 3 unchanged files by eTag/quickXorHash
    download_history.clear()
    run2 = pipeline.run_from_sharepoint(connector=connector)
    assert run2.discovered_files == 3
    assert run2.processed_files == 0
    assert run2.skipped_unchanged_files == 3
    # Regular files (PDF and DOCX) were NOT downloaded at all during Run 2!
    assert "item-pdf-002" not in download_history
    assert "item-docx-003" not in download_history


def test_sharepoint_corrupted_file_still_deletes_local_raw_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that even if an extractor fails on a corrupted file, the local raw binary is deleted."""
    _mock_msal_client(monkeypatch)

    staging_dir = tmp_path / "raw_sharepoint"
    md_dir = tmp_path / "processed" / "markdown"
    json_dir = tmp_path / "processed" / "json"
    manifest_path = tmp_path / "processed" / "manifest.json"
    limitations_path = tmp_path / "logs" / "limitations_report.jsonl"

    download_history: List[str] = []
    concurrent_tracker: List[int] = []
    session = _create_mock_graph_session(
        file_payloads={
            "item-reg-001": _build_register_xlsx_bytes(),
            "item-pdf-002": _build_pdf_bytes(),
            "item-docx-003": _build_docx_bytes(),
            "item-corrupt-004": b"NOT_A_VALID_PDF_BINARY_HEADER",
        },
        download_history=download_history,
        staging_dir=staging_dir,
        max_concurrent_files_tracker=concurrent_tracker,
    )

    sp_config = SharePointConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret-789",
        hostname="ingenium.sharepoint.com",
        site_path="/sites/AZSintJan",
        drive_name="Documents",
        staging_dir=staging_dir,
        json_out_dir=json_dir,
    )
    connector = SharePointConnector(sp_config=sp_config, session=session)
    pipeline = ETLPipeline(
        markdown_out_dir=md_dir,
        json_out_dir=json_dir,
        manifest_path=manifest_path,
        limitations_report_path=limitations_path,
    )

    result = pipeline.run_from_sharepoint(connector=connector)
    assert result.discovered_files == 4
    assert result.processed_files == 3
    assert result.failed_files == 1
    assert result.limitations_logged >= 1

    # Verify corrupted file was logged to limitations_report.jsonl
    records = pipeline.quality_logger.read_records()
    assert any(r.file_name == "Corrupt_Schema.pdf" for r in records)

    # Verify the corrupted raw binary AND its incomplete seeded JSON were cleaned up
    assert not staging_dir.exists() or not any(staging_dir.rglob("*"))
    corrupt_doc_id = compute_doc_id("PRJ_104_AZ_Sint_Jan/Corrupt_Schema.pdf")
    assert not (json_dir / f"{corrupt_doc_id}.json").exists()


def test_cli_from_sharepoint_inventory_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CLI `inventory --from-sharepoint` and `run --from-sharepoint` flags."""
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant-123")
    monkeypatch.setenv("AZURE_CLIENT_ID", "client-456")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret-789")
    monkeypatch.setenv("SHAREPOINT_HOSTNAME", "ingenium.sharepoint.com")
    monkeypatch.setenv("SHAREPOINT_SITE_PATH", "/sites/AZSintJan")
    monkeypatch.setenv("SHAREPOINT_DRIVE_NAME", "Documents")

    staging_dir = tmp_path / "raw_sharepoint"
    md_dir = tmp_path / "processed" / "markdown"
    json_dir = tmp_path / "processed" / "json"
    manifest_path = tmp_path / "processed" / "manifest.json"
    limitations_path = tmp_path / "logs" / "limitations_report.jsonl"

    download_history: List[str] = []
    concurrent_tracker: List[int] = []
    mock_session = _create_mock_graph_session(
        file_payloads={
            "item-reg-001": _build_register_xlsx_bytes(),
            "item-pdf-002": _build_pdf_bytes(),
            "item-docx-003": _build_docx_bytes(),
        },
        download_history=download_history,
        staging_dir=staging_dir,
        max_concurrent_files_tracker=concurrent_tracker,
    )

    _mock_msal_client(monkeypatch)
    monkeypatch.setattr("requests.Session", lambda: mock_session)

    res_inv = runner.invoke(
        app,
        ["inventory", str(staging_dir), "--from-sharepoint"],
    )
    assert res_inv.exit_code == 0
    assert "PRJ_104" in res_inv.stdout
    assert "Intensieve" in res_inv.stdout
    assert "specification" in res_inv.stdout
    # Heavy PDF/DOCX binaries were not downloaded during inventory!
    assert "item-pdf-002" not in download_history
    assert "item-docx-003" not in download_history

    res_run = runner.invoke(
        app,
        [
            "run",
            str(staging_dir),
            "--from-sharepoint",
            "--markdown-out",
            str(md_dir),
            "--json-out",
            str(json_dir),
            "--manifest",
            str(manifest_path),
            "--limitations-log",
            str(limitations_path),
        ],
    )
    assert res_run.exit_code == 0
    assert "Newly Processed Files" in res_run.stdout
    assert len(list(json_dir.glob("*.json"))) == 3
    assert len(list(md_dir.glob("*.md"))) == 3
    assert not staging_dir.exists() or not any(staging_dir.rglob("*"))


def test_central_credentials_file_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify SharePointConfig & SharePointConnector pull Entra ID & SharePoint settings from central credentials.txt."""
    for key in (
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "TEMP_BEARER_TOKEN",
        "SHAREPOINT_HOSTNAME",
        "SHAREPOINT_SITE_PATH",
        "SHAREPOINT_DRIVE_NAME",
    ):
        monkeypatch.delenv(key, raising=False)

    cred_file = tmp_path / "credentials.txt"
    cred_file.write_text(
        "# Central credentials.txt test\n"
        "TENANT_ID=central-tenant-001\n"
        "CLIENT_ID=central-client-002\n"
        "CLIENT_SECRET=central-secret-003\n"
        "SHAREPOINT_HOSTNAME=ingenium.sharepoint.com\n"
        "SHAREPOINT_SITE_PATH=/sites/AZSintJan\n"
        "SHAREPOINT_DRIVE_NAME=Documents\n"
        "TEMP_BEARER_TOKEN=temp-explorer-token-xyz\n",
        encoding="utf-8",
    )

    connector = SharePointConnector.from_env(
        staging_dir=tmp_path / "staging",
        json_out_dir=tmp_path / "json",
        credentials_path=cred_file,
    )
    assert connector.config.tenant_id == "central-tenant-001"
    assert connector.config.client_id == "central-client-002"
    assert connector.config.client_secret == "central-secret-003"
    assert connector.config.hostname == "ingenium.sharepoint.com"
    assert connector.config.site_path == "/sites/AZSintJan"
    assert connector.config.drive_name == "Documents"
    assert connector.authenticate() == "temp-explorer-token-xyz"
