"""Ephemeral SharePoint connector via Microsoft Entra ID (`msal`) and Microsoft Graph API."""

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import msal
import requests
from pydantic import BaseModel, Field

from config import (
    GRAPH_API_BASE_URL,
    GRAPH_DEFAULT_SCOPE,
    SUPPORTED_EXTRACTION_EXTENSIONS,
    get_settings,
    load_credentials_file,
)
from domain import DocumentMetadata, ExtractionMethod, ParsedDoc, compute_doc_id, utc_now

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a local file."""
    hasher = hashlib.sha256()
    with Path(file_path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _default_method_for_ext(ext: str) -> ExtractionMethod:
    ext_lower = ext.lower()
    if ext_lower == ".docx":
        return ExtractionMethod.PYTHON_DOCX
    if ext_lower in {".xlsx", ".xls"}:
        return ExtractionMethod.OPENPYXL_TABLE
    return ExtractionMethod.PYMUPDF_DIGITAL


@dataclass
class SharePointConfig:
    """Microsoft Entra ID and SharePoint site/drive configuration."""

    tenant_id: str
    client_id: str
    client_secret: str
    hostname: str
    site_path: str
    drive_name: str = "Documents"
    temp_bearer_token: str = ""
    staging_dir: Path = get_settings().raw_sharepoint_dir
    json_out_dir: Path = get_settings().json_out_dir

    @classmethod
    def from_env(
        cls,
        staging_dir: str | Path | None = None,
        json_out_dir: str | Path | None = None,
        credentials_path: str | Path | None = None,
    ) -> "SharePointConfig":
        settings = get_settings()
        cred_file = Path(credentials_path) if credentials_path is not None else None
        file_creds = load_credentials_file(
            credentials_path=cred_file,
            override=credentials_path is not None,
        )

        def _pick(key: str, default: str = "") -> str:
            if credentials_path is not None and key in file_creds:
                return file_creds[key]
            return os.environ.get(key) or file_creds.get(key) or default

        return cls(
            tenant_id=_pick("AZURE_TENANT_ID", settings.azure_tenant_id).strip(),
            client_id=_pick("AZURE_CLIENT_ID", settings.azure_client_id).strip(),
            client_secret=_pick("AZURE_CLIENT_SECRET", settings.azure_client_secret).strip(),
            temp_bearer_token=_pick("TEMP_BEARER_TOKEN", settings.temp_bearer_token).strip(),
            hostname=_pick("SHAREPOINT_HOSTNAME", settings.sharepoint_hostname).strip(),
            site_path=_pick("SHAREPOINT_SITE_PATH", settings.sharepoint_site_path).strip(),
            drive_name=_pick(
                "SHAREPOINT_DRIVE_NAME", settings.sharepoint_drive_name or "Documents"
            ).strip(),
            staging_dir=Path(staging_dir)
            if staging_dir is not None
            else settings.raw_sharepoint_dir,
            json_out_dir=Path(json_out_dir) if json_out_dir is not None else settings.json_out_dir,
        )

    def validate_credentials(self) -> None:
        required = [
            ("SHAREPOINT_HOSTNAME", self.hostname),
            ("SHAREPOINT_SITE_PATH", self.site_path),
        ]
        if not self.temp_bearer_token:
            required = [
                ("AZURE_TENANT_ID", self.tenant_id),
                ("AZURE_CLIENT_ID", self.client_id),
                ("AZURE_CLIENT_SECRET", self.client_secret),
            ] + required
        missing = [name for name, val in required if not val]
        if missing:
            raise ValueError(
                f"Missing required SharePoint/Entra credentials in .env or environment: {', '.join(missing)}"
            )


class RemoteDriveItem(BaseModel):
    """Metadata for a single remote SharePoint file discovered via Microsoft Graph."""

    item_id: str
    name: str
    relative_path: str
    web_url: str | None = None
    parent_path: str = ""
    etag: str | None = None
    ctag: str | None = None
    quick_xor_hash: str | None = None
    sha256_hash: str | None = None
    size_bytes: int = 0
    download_url: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def extension(self) -> str:
        return Path(self.name).suffix.lower()

    @property
    def relative_folder_path(self) -> str:
        parent = Path(self.relative_path).parent
        return parent.as_posix() if parent != Path(".") else ""


class SharePointConnector:
    """Streams files from SharePoint via Microsoft Graph API, seeds `ParsedDoc` JSONs, and cleans up binaries."""

    def __init__(
        self,
        sp_config: SharePointConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = sp_config or SharePointConfig.from_env()
        self.session = session or requests.Session()
        self._access_token: str | None = None
        self._site_id: str | None = None
        self._drive_id: str | None = None

    @classmethod
    def from_env(
        cls,
        staging_dir: str | Path | None = None,
        json_out_dir: str | Path | None = None,
        session: requests.Session | None = None,
        credentials_path: str | Path | None = None,
    ) -> "SharePointConnector":
        return cls(
            sp_config=SharePointConfig.from_env(
                staging_dir=staging_dir,
                json_out_dir=json_out_dir,
                credentials_path=credentials_path,
            ),
            session=session,
        )

    def authenticate(self) -> str:
        self.config.validate_credentials()
        if self.config.temp_bearer_token:
            self._access_token = self.config.temp_bearer_token
            return self._access_token

        authority = f"https://login.microsoftonline.com/{self.config.tenant_id}"
        app = msal.ConfidentialClientApplication(
            client_id=self.config.client_id,
            authority=authority,
            client_credential=self.config.client_secret,
        )
        token_result = app.acquire_token_for_client(scopes=[GRAPH_DEFAULT_SCOPE])
        if not isinstance(token_result, dict) or "access_token" not in token_result:
            err = (
                token_result.get("error_description") or token_result.get("error")
                if isinstance(token_result, dict)
                else "No token response returned"
            )
            raise RuntimeError(f"Microsoft Entra authentication failed: {err}")

        self._access_token = str(token_result["access_token"])
        return self._access_token

    def _headers(self) -> dict[str, str]:
        if not self._access_token:
            self.authenticate()
        return {"Authorization": f"Bearer {self._access_token}", "Accept": "application/json"}

    def _graph_get(self, url: str) -> dict[str, Any]:
        resp = self.session.get(url, headers=self._headers(), timeout=60)
        resp.raise_for_status()
        return resp.json()

    def resolve_site_and_drive(self) -> tuple[str, str]:
        if self._site_id and self._drive_id:
            return self._site_id, self._drive_id

        clean_site = "/" + self.config.site_path.strip("/")
        site_url = f"{GRAPH_API_BASE_URL}/sites/{self.config.hostname}:{clean_site}"
        site_data = self._graph_get(site_url)
        site_id = site_data.get("id")
        if not site_id:
            raise RuntimeError(f"Could not resolve SharePoint site ID from {site_url}")
        self._site_id = str(site_id)

        drives = self._graph_get(f"{GRAPH_API_BASE_URL}/sites/{self._site_id}/drives").get(
            "value", []
        )
        target = self.config.drive_name.strip().lower()
        selected: str | None = None
        for drv in drives:
            drv_name = str(drv.get("name", "")).strip().lower()
            if drv_name == target or (
                target in {"documents", "shared documents", "gedeelde documenten"}
                and drv_name in {"documents", "shared documents", "gedeelde documenten"}
            ):
                selected = str(drv["id"])
                break
        if not selected and drives:
            selected = str(drives[0]["id"])
        if not selected:
            raise RuntimeError(
                f"Could not find SharePoint drive '{self.config.drive_name}' on site {self._site_id}"
            )

        self._drive_id = selected
        return self._site_id, self._drive_id

    def list_remote_files(
        self,
        *,
        supported_only: bool = True,
        folder_prefix: str | None = None,
    ) -> list[RemoteDriveItem]:
        """Traverse SharePoint drive (or a subfolder) and return remote file metadata."""
        logger.info(
            "Connecting to SharePoint %s%s ...", self.config.hostname, self.config.site_path
        )
        _, drive_id = self.resolve_site_and_drive()
        clean_prefix = (folder_prefix or "").strip("/")
        if clean_prefix:
            encoded = quote(clean_prefix, safe="/")
            start_url = f"{GRAPH_API_BASE_URL}/drives/{drive_id}/root:/{encoded}:/children"
            logger.info(
                "Scanning SharePoint subfolder '%s' (drive=%s)...",
                clean_prefix,
                self.config.drive_name,
            )
        else:
            start_url = f"{GRAPH_API_BASE_URL}/drives/{drive_id}/root/children"
            logger.info("Scanning SharePoint drive '%s' from root...", self.config.drive_name)

        collected: list[RemoteDriveItem] = []
        self._traverse_folder(drive_id, start_url, clean_prefix, collected, supported_only)
        collected.sort(key=lambda item: item.relative_path)
        logger.info("SharePoint discovery complete: %d matching file(s) found.", len(collected))
        return collected

    def _traverse_folder(
        self,
        drive_id: str,
        children_url: str,
        parent_rel_path: str,
        collected: list[RemoteDriveItem],
        supported_only: bool,
    ) -> None:
        next_url: str | None = children_url
        while next_url:
            page = self._graph_get(next_url)
            for raw_item in page.get("value", []):
                name = str(raw_item.get("name", "")).strip()
                item_id = str(raw_item.get("id", "")).strip()
                if not name or not item_id or name.startswith("."):
                    continue
                if "folder" in raw_item:
                    folder_rel = f"{parent_rel_path}/{name}".strip("/") if parent_rel_path else name
                    if folder_rel.count("/") <= 1:
                        logger.info(
                            "Scanning folder: %s (%d files found so far)",
                            folder_rel,
                            len(collected),
                        )
                    sub_url = f"{GRAPH_API_BASE_URL}/drives/{drive_id}/items/{item_id}/children"
                    self._traverse_folder(drive_id, sub_url, folder_rel, collected, supported_only)
                elif "file" in raw_item:
                    parsed = self._parse_drive_item(raw_item, fallback_parent=parent_rel_path)
                    if supported_only and parsed.extension not in SUPPORTED_EXTRACTION_EXTENSIONS:
                        continue
                    collected.append(parsed)
            next_url = page.get("@odata.nextLink")

    @staticmethod
    def _parse_drive_item(raw_item: dict[str, Any], fallback_parent: str = "") -> RemoteDriveItem:
        name = str(raw_item["name"]).strip()
        raw_parent = str((raw_item.get("parentReference") or {}).get("path") or "")
        cleaned_parent = re.sub(r"^/drives?/[^/]+/root:?|^/drive/root:?", "", raw_parent).strip("/")
        if not cleaned_parent and fallback_parent:
            cleaned_parent = fallback_parent.strip("/")
        rel_path = f"{cleaned_parent}/{name}".strip("/") if cleaned_parent else name
        hashes = (raw_item.get("file") or {}).get("hashes") or {}

        return RemoteDriveItem(
            item_id=str(raw_item["id"]),
            name=name,
            relative_path=rel_path,
            web_url=raw_item.get("webUrl"),
            parent_path=raw_parent or cleaned_parent,
            etag=raw_item.get("eTag"),
            ctag=raw_item.get("cTag"),
            quick_xor_hash=hashes.get("quickXorHash"),
            sha256_hash=hashes.get("sha256Hash") or hashes.get("sha1Hash"),
            size_bytes=int(raw_item.get("size") or 0),
            download_url=raw_item.get("@microsoft.graph.downloadUrl"),
        )

    def download_file(
        self,
        item: RemoteDriveItem,
        staging_dir: str | Path | None = None,
    ) -> Path:
        base_staging = Path(staging_dir) if staging_dir is not None else self.config.staging_dir
        local_path = base_staging / Path(item.relative_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if item.download_url:
            resp = self.session.get(item.download_url, stream=True, timeout=120)
        else:
            _, drive_id = self.resolve_site_and_drive()
            content_url = f"{GRAPH_API_BASE_URL}/drives/{drive_id}/items/{item.item_id}/content"
            resp = self.session.get(content_url, headers=self._headers(), stream=True, timeout=120)
        resp.raise_for_status()

        with local_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    fh.write(chunk)
        return local_path

    def seed_target_json(
        self,
        item: RemoteDriveItem,
        local_file_path: str | Path,
        json_out_dir: str | Path | None = None,
    ) -> tuple[Path, str]:
        """Compute SHA-256 and seed `<doc_id>.json` as a canonical `ParsedDoc` with SharePoint provenance."""
        target_dir = Path(json_out_dir) if json_out_dir is not None else self.config.json_out_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        file_hash = compute_file_sha256(local_file_path)
        doc_id = compute_doc_id(item.relative_path)

        seeded_doc = ParsedDoc(
            document_id=doc_id,
            content_hash=file_hash,
            markdown="",
            sections=[],
            metadata=DocumentMetadata(
                source_path=Path(item.relative_path).as_posix(),
                sharepoint_url=item.web_url,
                extra={
                    "file_name": item.name,
                    "file_extension": item.extension,
                    "relative_folder_path": item.relative_folder_path,
                    "sharepoint_item_id": item.item_id,
                    "sharepoint_etag": item.etag or item.ctag,
                    "extraction_method": _default_method_for_ext(item.extension).value,
                    "processed_at": utc_now().isoformat(),
                },
            ),
        )
        json_path = target_dir / f"{doc_id}.json"
        json_path.write_text(seeded_doc.model_dump_json(indent=2), encoding="utf-8")
        return json_path, file_hash

    def download_and_seed(
        self,
        item: RemoteDriveItem,
        staging_dir: str | Path | None = None,
        json_out_dir: str | Path | None = None,
    ) -> tuple[Path, Path, str]:
        local_path = self.download_file(item, staging_dir=staging_dir)
        seeded_json, file_hash = self.seed_target_json(item, local_path, json_out_dir=json_out_dir)
        return local_path, seeded_json, file_hash

    def cleanup_local_file(
        self,
        local_file_path: str | Path,
        staging_dir: str | Path | None = None,
    ) -> None:
        """Delete a downloaded raw binary immediately and prune empty staging directories."""
        path = Path(local_file_path)
        base_staging = (
            Path(staging_dir).resolve()
            if staging_dir is not None
            else self.config.staging_dir.resolve()
        )
        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)

        parent = path.parent.resolve()
        while parent != base_staging and base_staging in parent.parents:
            try:
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
                else:
                    break
            except OSError:
                break
