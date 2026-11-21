"""Ephemeral SharePoint connector via Microsoft Entra ID (`msal`) and Microsoft Graph API."""

import hashlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import msal
import requests
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter

from config import (
    GRAPH_API_BASE_URL,
    GRAPH_DEFAULT_SCOPE,
    SUPPORTED_EXTRACTION_EXTENSIONS,
    SharePointSiteSpec,
    get_settings,
    load_credentials_file,
)

logger = logging.getLogger(__name__)


class SharePointAuthError(RuntimeError):
    """Raised on HTTP 401/403 from SharePoint / Microsoft Graph."""

    def __init__(self, status_code: int, target: str, reason: str = "") -> None:
        self.status_code = status_code
        self.target = target
        self.reason = reason or ("Forbidden" if status_code == 403 else "Unauthorized")
        super().__init__(f"HTTP {status_code} ({self.reason}) accessing '{target}'")


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a local file."""
    hasher = hashlib.sha256()
    with Path(file_path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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

    @property
    def site_slug(self) -> str:
        return self.site_path.strip("/").split("/")[-1] if self.site_path.strip("/") else "default"

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
    last_modified: str | None = None
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
        if session is not None:
            self.session = session
        else:
            self.session = requests.Session()
            adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)
        self._access_token: str | None = None
        self._auth_lock = threading.Lock()
        self._site_id: str | None = None
        self._drive_id: str | None = None

    def for_site(self, spec: SharePointSiteSpec) -> "SharePointConnector":
        """Create a site-scoped connector sharing this connector's pooled session and MSAL token."""
        site_cfg = SharePointConfig(
            tenant_id=self.config.tenant_id,
            client_id=self.config.client_id,
            client_secret=self.config.client_secret,
            hostname=spec.hostname or self.config.hostname,
            site_path=spec.site_path,
            drive_name=self.config.drive_name,
            temp_bearer_token=self.config.temp_bearer_token,
            staging_dir=self.config.staging_dir,
            json_out_dir=self.config.json_out_dir,
        )
        child = SharePointConnector(sp_config=site_cfg, session=self.session)
        child._access_token = self._access_token
        child._auth_lock = self._auth_lock
        return child

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

    def authenticate(self, *, force_refresh: bool = False) -> str:
        self.config.validate_credentials()
        if self.config.temp_bearer_token:
            self._access_token = self.config.temp_bearer_token
            return self._access_token

        if self._access_token and not force_refresh:
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
        with self._auth_lock:
            if not self._access_token:
                self.authenticate()
            token = self._access_token
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    def _request_with_retry(
        self,
        url: str,
        *,
        target_label: str,
        use_auth: bool = True,
        stream: bool = False,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> Any:
        """Execute GET with fail-fast 403, single MSAL refresh on 401, and 429/503 backoff."""
        refreshed_token = False
        for attempt in range(max_retries + 1):
            used_token = self._access_token
            kwargs: dict[str, Any] = {"timeout": timeout}
            if stream:
                kwargs["stream"] = True
            if use_auth:
                kwargs["headers"] = self._headers()
                used_token = self._access_token

            resp = self.session.get(url, **kwargs)
            status = int(getattr(resp, "status_code", 200))

            if status == 403:
                raise SharePointAuthError(403, target_label, "Forbidden")

            if status == 401:
                if use_auth and not refreshed_token and not self.config.temp_bearer_token:
                    with self._auth_lock:
                        if self._access_token == used_token:
                            logger.warning(
                                "HTTP 401 on '%s'; refreshing Microsoft Entra token once.",
                                target_label,
                            )
                            self.authenticate(force_refresh=True)
                    refreshed_token = True
                    continue
                raise SharePointAuthError(401, target_label, "Unauthorized")

            if status in (429, 503) and attempt < max_retries:
                headers = getattr(resp, "headers", None) or {}
                retry_after_raw = headers.get("Retry-After") or headers.get("retry-after")
                try:
                    wait_s = min(float(retry_after_raw), 30.0) if retry_after_raw else 0.0
                except (TypeError, ValueError):
                    wait_s = 0.0
                if wait_s <= 0:
                    wait_s = min(2.0**attempt, 15.0)
                logger.warning(
                    "HTTP %d on '%s' (attempt %d/%d); backing off %.1fs.",
                    status,
                    target_label,
                    attempt + 1,
                    max_retries,
                    wait_s,
                )
                time.sleep(wait_s)
                continue

            resp.raise_for_status()
            return resp

        raise RuntimeError(f"Exhausted retries requesting '{target_label}'")

    def _graph_get(self, url: str, *, target_label: str | None = None) -> dict[str, Any]:
        resp = self._request_with_retry(
            url,
            target_label=target_label or f"{self.config.hostname}{self.config.site_path}",
            use_auth=True,
            stream=False,
            timeout=60,
        )
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
                        logger.debug(
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
            last_modified=raw_item.get("lastModifiedDateTime"),
            size_bytes=int(raw_item.get("size") or 0),
            download_url=raw_item.get("@microsoft.graph.downloadUrl"),
        )

    def download_file(
        self,
        item: RemoteDriveItem,
        staging_dir: str | Path | None = None,
        *,
        target_rel_path: str | Path | None = None,
    ) -> Path:
        base_staging = Path(staging_dir) if staging_dir is not None else self.config.staging_dir
        local_path = base_staging / Path(target_rel_path or item.relative_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if item.download_url:
            resp = self._request_with_retry(
                item.download_url,
                target_label=item.relative_path,
                use_auth=False,
                stream=True,
                timeout=120,
            )
        else:
            _, drive_id = self.resolve_site_and_drive()
            content_url = f"{GRAPH_API_BASE_URL}/drives/{drive_id}/items/{item.item_id}/content"
            resp = self._request_with_retry(
                content_url,
                target_label=item.relative_path,
                use_auth=True,
                stream=True,
                timeout=120,
            )

        with local_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    fh.write(chunk)
        return local_path

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
