"""Unit & E2E tests for Stage 1 runner (`ETLRunner`, incremental manifest, and SharePoint streaming)."""

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import docx
import openpyxl
import pymupdf as fitz
import pytest

from domain import DocCategory, ParsedDoc, compute_doc_id
from etl import ETLRunner, SharePointConfig, SharePointConnector, load_processed_docs


def _build_register_xlsx_bytes() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Documentenlijst"
    ws.append(["Doc_Nr", "Bestandsnaam", "Omschrijving", "Gebouw", "Fase", "Revisie", "Type"])
    ws.append(
        [
            "ING-104-HVAC-01",
            "Bestek_HVAC_Ventilatie.pdf",
            "Technisch Bestek",
            "Blok C - Intensieve Zorgen",
            "Definitief Ontwerp",
            "Rev.B",
            "Bestek",
        ]
    )
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _build_pdf_bytes() -> bytes:
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 70), "1. Algemene Voorwaarden\nTechnische specificatie HVAC AZ Sint-Jan.", fontsize=11
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _build_docx_bytes() -> bytes:
    d = docx.Document()
    d.add_heading("1. Werfvergadering HVAC", level=1)
    d.add_paragraph("Bespreking vordering kanaalmontage in Blok B Operatiekwartier.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_local_incremental_etl_and_load_processed_docs(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    proj_dir = raw_root / "PRJ_104_AZ_Sint_Jan" / "02_Aanbesteding" / "Blok_B_Operatiekwartier"
    proj_dir.mkdir(parents=True)
    (proj_dir / "Bestek_HVAC.pdf").write_bytes(_build_pdf_bytes())
    (proj_dir / "Werfverslag_01.docx").write_bytes(_build_docx_bytes())

    processed = tmp_path / "processed"
    runner = ETLRunner(
        markdown_out_dir=processed / "markdown",
        json_out_dir=processed / "json",
        manifest_path=processed / "manifest.json",
        limitations_report_path=tmp_path / "logs" / "limitations.jsonl",
    )

    run1 = runner.run_local(raw_root)
    assert (
        run1.discovered_files,
        run1.processed_files,
        run1.skipped_unchanged_files,
        run1.failed_files,
    ) == (2, 2, 0, 0)
    assert all(isinstance(d, ParsedDoc) for d in run1.documents)

    # Second run skips unchanged files via SHA-256 manifest
    run2 = runner.run_local(raw_root)
    assert (run2.processed_files, run2.skipped_unchanged_files) == (0, 2)

    # load_processed_docs loads canonical ParsedDoc JSONs directly
    loaded = load_processed_docs(processed)
    assert len(loaded) == 2
    assert all(d.metadata.project == "PRJ_104" for d in loaded)
    assert all(d.metadata.buildings == ["Blok B", "Operatiekwartier"] for d in loaded)
    assert all(d.metadata.phase == "Aanbesteding" for d in loaded)


class _FakeGraphResponse:
    def __init__(
        self, json_data: Any = None, content_bytes: bytes = b"", status_code: int = 200
    ) -> None:
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
        while chunk := stream.read(chunk_size):
            yield chunk


def test_sharepoint_streaming_two_pass_and_ephemeral_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_app = MagicMock()
    mock_app.acquire_token_for_client.return_value = {"access_token": "mock-token"}
    monkeypatch.setattr("msal.ConfidentialClientApplication", MagicMock(return_value=mock_app))

    staging_dir = tmp_path / "staging"
    processed = tmp_path / "processed"
    payloads = {
        "item-reg-1": _build_register_xlsx_bytes(),
        "item-pdf-2": _build_pdf_bytes(),
        "item-bad-3": b"NOT_A_VALID_PDF",
    }
    downloads: list[str] = []

    def fake_get(
        url: str, headers: Any = None, stream: bool = False, timeout: int = 60
    ) -> _FakeGraphResponse:
        if "/sites/ingenium.sharepoint.com:/sites/06038004-INT" in url:
            return _FakeGraphResponse({"id": "site-1"})
        if url.endswith("/sites/site-1/drives"):
            return _FakeGraphResponse({"value": [{"id": "drive-1", "name": "Documents"}]})
        if url.endswith("/drives/drive-1/root/children"):
            return _FakeGraphResponse(
                {
                    "value": [
                        {
                            "id": "item-reg-1",
                            "name": "00_Documentenlijst_HVAC.xlsx",
                            "eTag": "etag-reg",
                            "webUrl": "https://ingenium.sharepoint.com/sites/06038004-INT/00_Documentenlijst_HVAC.xlsx",
                            "parentReference": {"path": "/drives/drive-1/root:"},
                            "file": {"hashes": {"quickXorHash": "qx-reg"}},
                        },
                        {
                            "id": "item-pdf-2",
                            "name": "Bestek_HVAC_Ventilatie.pdf",
                            "eTag": "etag-pdf",
                            "webUrl": "https://ingenium.sharepoint.com/sites/06038004-INT/Bestek_HVAC_Ventilatie.pdf",
                            "parentReference": {"path": "/drives/drive-1/root:"},
                            "file": {"hashes": {"quickXorHash": "qx-pdf"}},
                        },
                        {
                            "id": "item-bad-3",
                            "name": "Corrupt.pdf",
                            "eTag": "etag-bad",
                            "webUrl": "https://ingenium.sharepoint.com/sites/06038004-INT/Corrupt.pdf",
                            "parentReference": {"path": "/drives/drive-1/root:"},
                            "file": {"hashes": {"quickXorHash": "qx-bad"}},
                        },
                    ]
                }
            )
        for iid, data in payloads.items():
            if url.endswith(f"/items/{iid}/content"):
                downloads.append(iid)
                return _FakeGraphResponse(content_bytes=data)
        raise AssertionError(f"Unexpected URL: {url}")

    session = MagicMock()
    session.get.side_effect = fake_get

    connector = SharePointConnector(
        sp_config=SharePointConfig(
            tenant_id="t",
            client_id="c",
            client_secret="s",
            hostname="ingenium.sharepoint.com",
            site_path="/sites/06038004-INT",
            staging_dir=staging_dir,
            json_out_dir=processed / "json",
        ),
        session=session,
    )
    runner = ETLRunner(
        markdown_out_dir=processed / "markdown",
        json_out_dir=processed / "json",
        manifest_path=processed / "manifest.json",
        limitations_report_path=tmp_path / "logs" / "limitations.jsonl",
    )

    res = runner.run_sharepoint(connector=connector)
    assert (res.discovered_files, res.processed_files, res.failed_files) == (3, 2, 1)
    assert downloads[0] == "item-reg-1"  # Register downloaded in Pass 1
    assert not staging_dir.exists() or not any(staging_dir.rglob("*"))  # Ephemeral cleanup
    assert not (processed / "json" / f"{compute_doc_id('Corrupt.pdf')}.json").exists()

    pdf_doc = ParsedDoc.model_validate_json(
        (processed / "json" / f"{compute_doc_id('Bestek_HVAC_Ventilatie.pdf')}.json").read_text()
    )
    assert pdf_doc.metadata.project == "06038.004"
    assert pdf_doc.metadata.buildings == ["Blok C", "Intensieve Zorgen"]
    assert pdf_doc.metadata.phase == "Definitief Ontwerp"
    assert pdf_doc.metadata.extra["doc_category"] == DocCategory.SPECIFICATION.value

    # Second run skips unchanged remote files without re-downloading PDF
    downloads.clear()
    res2 = runner.run_sharepoint(connector=connector)
    assert "item-pdf-2" not in downloads
    assert res2.skipped_unchanged_files == 2
