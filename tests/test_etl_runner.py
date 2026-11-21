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
    assert all(d.metadata.buildings == ["Blok B"] for d in loaded)
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


def test_load_sharepoint_sites_tsv_dedup_and_blank_urls(tmp_path: Path) -> None:
    from config import Settings, load_sharepoint_sites

    tsv = tmp_path / "sites.tsv"
    tsv.write_text(
        "# comment line\n"
        "project_code\turl\n"
        "22071.002A\thttps://company.sharepoint.com/sites/22071002A-INT/Shared Documents/General\n"
        "EMPTY_URL\t\n"
        "22071.002A\thttps://company.sharepoint.com/sites/22071002A-INT/Shared%20Documents/General\n"
        "24041.001B\t/sites/24041001B-INT\n",
        encoding="utf-8",
    )
    cfg = Settings(sharepoint_hostname="company.sharepoint.com")
    specs = load_sharepoint_sites(sites_file=tsv, settings=cfg)
    assert len(specs) == 2
    assert specs[0].project_code == "22071.002A"
    assert specs[0].site_path == "/sites/22071002A-INT"
    assert specs[0].folder_prefix == "General"
    assert specs[1].project_code == "24041.001B"
    assert specs[1].hostname == "company.sharepoint.com"

    override = load_sharepoint_sites(site_override="/sites/99999001-INT", settings=cfg)
    assert len(override) == 1
    assert override[0].site_slug == "99999001-INT"


def test_sharepoint_401_refresh_403_failfast_and_429_backoff(tmp_path: Path, monkeypatch) -> None:
    from etl import SharePointAuthError

    class _Resp:
        def __init__(self, status: int, data: Any = None, headers: dict[str, str] | None = None):
            self.status_code = status
            self._data = data or {}
            self.headers = headers or {}

        def json(self) -> Any:
            return self._data

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    # 1. 401 triggers single MSAL refresh -> succeeds; 429 retries -> succeeds
    calls: list[str] = []
    refresh_count = 0

    class _RefreshSession:
        def get(self, url: str, **kwargs: Any) -> _Resp:
            calls.append(url)
            if len(calls) == 1:
                return _Resp(401)
            if len(calls) == 2:
                return _Resp(429, headers={"Retry-After": "0"})
            return _Resp(200, {"id": "site-ok"})

    cfg = SharePointConfig(
        tenant_id="t",
        client_id="c",
        client_secret="s",
        hostname="company.sharepoint.com",
        site_path="/sites/12345001-INT",
        staging_dir=tmp_path / "st",
        json_out_dir=tmp_path / "js",
    )
    sp = SharePointConnector(sp_config=cfg, session=_RefreshSession())  # type: ignore[arg-type]

    def _fake_auth(*, force_refresh: bool = False) -> str:
        nonlocal refresh_count
        refresh_count += 1
        sp._access_token = f"tok-{refresh_count}"
        return sp._access_token

    monkeypatch.setattr(sp, "authenticate", _fake_auth)
    data = sp._graph_get("https://graph.microsoft.com/v1.0/sites/test")
    assert data == {"id": "site-ok"}
    assert refresh_count == 2  # initial auth + 1 refresh on 401

    # 2. Site-level 403 fails fast (1 call, 0 retries) and records unauthorized without crashing
    forbidden_calls = 0

    class _ForbiddenSession:
        def get(self, url: str, **kwargs: Any) -> _Resp:
            nonlocal forbidden_calls
            forbidden_calls += 1
            return _Resp(403)

    sp_403 = SharePointConnector(
        sp_config=cfg,
        session=_ForbiddenSession(),  # type: ignore[arg-type]
    )
    sp_403._access_token = "tok"
    processed = tmp_path / "proc_403"
    runner = ETLRunner(
        markdown_out_dir=processed / "markdown",
        json_out_dir=processed / "json",
        manifest_path=processed / "manifest.json",
        limitations_report_path=tmp_path / "lim_403.jsonl",
    )
    res = runner.run_sharepoint(connector=sp_403)
    assert forbidden_calls == 1
    assert res.unauthorized_count == 1
    assert res.failed_files == 0
    assert any("403" in t for t in res.unauthorized_targets)
    try:
        sp_403._graph_get("https://graph.microsoft.com/v1.0/sites/test")
        raise AssertionError("Expected SharePointAuthError")
    except SharePointAuthError as err:
        assert err.status_code == 403


def test_multi_site_namespacing_prevents_doc_id_collision_and_skips_pre_download(
    tmp_path: Path,
) -> None:
    def _pdf_bytes(text: str) -> bytes:
        doc = fitz.open()
        doc.new_page().insert_text((50, 70), text)
        data = doc.tobytes()
        doc.close()
        return data

    pdf_a = _pdf_bytes("Specificatie project A Blok A.")
    pdf_b = _pdf_bytes("Specificatie project B Blok B.")
    downloads: list[str] = []

    def _make_site_connector(site_slug: str, pdf_bytes: bytes) -> SharePointConnector:
        class _SiteSession:
            def get(self, url: str, **kwargs: Any) -> _FakeGraphResponse:
                if f"/sites/{site_slug}" in url and "/drives" not in url:
                    return _FakeGraphResponse({"id": f"site-{site_slug}"})
                if url.endswith("/drives"):
                    return _FakeGraphResponse(
                        {"value": [{"id": f"drv-{site_slug}", "name": "Documents"}]}
                    )
                if url.endswith("/root/children"):
                    return _FakeGraphResponse(
                        {
                            "value": [
                                {
                                    "id": f"item-{site_slug}",
                                    "name": "Spec.pdf",
                                    "file": {"hashes": {"quickXorHash": f"qxh-{site_slug}"}},
                                    "eTag": f"etag-{site_slug}",
                                    "lastModifiedDateTime": "2026-01-01T00:00:00Z",
                                    "size": len(pdf_bytes),
                                    "webUrl": f"https://company.sharepoint.com/sites/{site_slug}/Shared%20Documents/Spec.pdf",
                                    "@microsoft.graph.downloadUrl": f"https://dl.local/{site_slug}/Spec.pdf",
                                    "parentReference": {"path": f"/drives/drv-{site_slug}/root:"},
                                }
                            ]
                        }
                    )
                if f"dl.local/{site_slug}/Spec.pdf" in url:
                    downloads.append(site_slug)
                    return _FakeGraphResponse(content_bytes=pdf_bytes)
                return _FakeGraphResponse(status_code=404)

        cfg = SharePointConfig(
            tenant_id="t",
            client_id="c",
            client_secret="s",
            hostname="company.sharepoint.com",
            site_path=f"/sites/{site_slug}",
            temp_bearer_token="tok",
            staging_dir=tmp_path / "staging",
            json_out_dir=tmp_path / "proc" / "json",
        )
        return SharePointConnector(sp_config=cfg, session=_SiteSession())  # type: ignore[arg-type]

    processed = tmp_path / "proc"
    runner = ETLRunner(
        markdown_out_dir=processed / "markdown",
        json_out_dir=processed / "json",
        manifest_path=processed / "manifest.json",
        limitations_report_path=tmp_path / "lim_multi.jsonl",
    )
    sp_a = _make_site_connector("22071002A-INT", pdf_a)
    sp_b = _make_site_connector("24041001B-INT", pdf_b)

    res_a = runner.run_sharepoint(
        connector=sp_a, workers=2, project_override="22071.002A", namespace_by_site=True
    )
    res_b = runner.run_sharepoint(
        connector=sp_b, workers=2, project_override="24041.001B", namespace_by_site=True
    )
    assert res_a.processed_files == 1 and res_b.processed_files == 1
    loaded = load_processed_docs(processed)
    assert len(loaded) == 2
    assert {d.metadata.project for d in loaded} == {"22071.002A", "24041.001B"}
    assert len({d.document_id for d in loaded}) == 2

    # Re-run skips both before downloading any bytes
    downloads.clear()
    res_a2 = runner.run_sharepoint(
        connector=sp_a, workers=2, project_override="22071.002A", namespace_by_site=True
    )
    assert res_a2.skipped_unchanged_files == 1
    assert downloads == []
