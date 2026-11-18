"""Ephemeral SharePoint connector via Microsoft Entra ID (`msal`) and Microsoft Graph API.

Implements a low-disk-footprint streaming workflow:
  1. Remote Tree Discovery (Metadata Only): Lists SharePoint DriveItems without downloading binaries.
  2. Single-File Download & Target JSON Seeding: Downloads one file at a time to a temporary
     staging directory (`data/raw_sharepoint/<relative_path>`), computes its SHA-256 hash, and
     seeds `data/processed/json/<doc_id>.json` with SharePoint location/provenance metadata.
  3. Immediate Local Deletion: Deletes the temporary raw binary immediately after in-place JSON
     enrichment and Markdown generation (inside a `try ... finally` block) and prunes empty folders.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import msal
from pydantic import BaseModel, Field
import requests

import etl.config as config
from etl.config import (
    GRAPH_API_BASE_URL,
    GRAPH_DEFAULT_SCOPE,
    JSON_OUT_DIR,
    RAW_SHAREPOINT_DIR,
    SUPPORTED_EXTRACTION_EXTENSIONS,
)
from etl.metadata_matcher import compute_doc_id
from etl.schemas import DocumentMetadata, ExtractedDocument, ExtractionMethod


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute the SHA-256 hex digest of a local file."""
    hasher = hashlib.sha256()
    path = Path(file_path)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _default_extraction_method_for_extension(ext: str) -> ExtractionMethod:
    """Return initial ExtractionMethod placeholder for a file extension before extraction."""
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
    staging_dir: Path = RAW_SHAREPOINT_DIR
    json_out_dir: Path = JSON_OUT_DIR

    @classmethod
    def from_env(
        cls,
        staging_dir: Optional[Union[str, Path]] = None,
        json_out_dir: Optional[Union[str, Path]] = None,
        credentials_path: Optional[Union[str, Path]] = None,
    ) -> "SharePointConfig":
        """Load Microsoft Entra and SharePoint settings from `credentials.txt` / environment."""
        cred_file = Path(credentials_path) if credentials_path is not None else None
        file_creds = config.load_credentials_file(
            credentials_path=cred_file,
            override=credentials_path is not None,
        )

        def _pick(key: str, default: str = "") -> str:
            if credentials_path is not None and key in file_creds:
                return file_creds[key]
            return os.environ.get(key) or file_creds.get(key) or default

        tenant_id = _pick("AZURE_TENANT_ID", config.AZURE_TENANT_ID)
        client_id = _pick("AZURE_CLIENT_ID", config.AZURE_CLIENT_ID)
        client_secret = _pick("AZURE_CLIENT_SECRET", config.AZURE_CLIENT_SECRET)
        temp_bearer_token = _pick("TEMP_BEARER_TOKEN", config.TEMP_BEARER_TOKEN)
        hostname = _pick("SHAREPOINT_HOSTNAME", config.SHAREPOINT_HOSTNAME)
        site_path = _pick("SHAREPOINT_SITE_PATH", config.SHAREPOINT_SITE_PATH)
        drive_name = _pick("SHAREPOINT_DRIVE_NAME", config.SHAREPOINT_DRIVE_NAME or "Documents")
        return cls(
            tenant_id=tenant_id.strip(),
            client_id=client_id.strip(),
            client_secret=client_secret.strip(),
            hostname=hostname.strip(),
            site_path=site_path.strip(),
            drive_name=drive_name.strip(),
            temp_bearer_token=temp_bearer_token.strip(),
            staging_dir=Path(staging_dir) if staging_dir is not None else RAW_SHAREPOINT_DIR,
            json_out_dir=Path(json_out_dir) if json_out_dir is not None else JSON_OUT_DIR,
        )

    def validate_credentials(self) -> None:
        """Raise ValueError if required Microsoft Entra or SharePoint settings are missing."""
        required_pairs = [
            ("SHAREPOINT_HOSTNAME", self.hostname),
            ("SHAREPOINT_SITE_PATH", self.site_path),
        ]
        if not self.temp_bearer_token:
            required_pairs = [
                ("AZURE_TENANT_ID", self.tenant_id),
                ("AZURE_CLIENT_ID", self.client_id),
                ("AZURE_CLIENT_SECRET", self.client_secret),
            ] + required_pairs
        missing = [name for name, val in required_pairs if not val]
        if missing:
            raise ValueError(
                f"Missing required SharePoint/Entra credentials in credentials.txt or environment: {', '.join(missing)}"
            )


class RemoteDriveItem(BaseModel):
    """Metadata for a single remote SharePoint file discovered via Microsoft Graph API."""

    item_id: str
    name: str
    relative_path: str
    web_url: Optional[str] = None
    parent_path: str = ""
    etag: Optional[str] = None
    ctag: Optional[str] = None
    quick_xor_hash: Optional[str] = None
    sha256_hash: Optional[str] = None
    size_bytes: int = 0
    download_url: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    @property
    def extension(self) -> str:
        """Lowercase file extension including leading dot (e.g., '.pdf')."""
        return Path(self.name).suffix.lower()

    @property
    def relative_folder_path(self) -> str:
        """Relative POSIX folder path of the file inside the SharePoint drive."""
        parent = Path(self.relative_path).parent
        return parent.as_posix() if parent != Path(".") else ""


class SharePointConnector:
    """Streams files from SharePoint via Microsoft Graph API, seeds JSONs, and cleans up raw files."""

    def __init__(
        self,
        sp_config: Optional[SharePointConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = sp_config or SharePointConfig.from_env()
        self.session = session or requests.Session()
        self._access_token: Optional[str] = None
        self._site_id: Optional[str] = None
        self._drive_id: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        staging_dir: Optional[Union[str, Path]] = None,
        json_out_dir: Optional[Union[str, Path]] = None,
        session: Optional[requests.Session] = None,
        credentials_path: Optional[Union[str, Path]] = None,
    ) -> "SharePointConnector":
        """Create a SharePointConnector configured from `credentials.txt` / environment variables."""
        return cls(
            sp_config=SharePointConfig.from_env(
                staging_dir=staging_dir,
                json_out_dir=json_out_dir,
                credentials_path=credentials_path,
            ),
            session=session,
        )

    def authenticate(self) -> str:
        """Acquire a Microsoft Graph bearer token via `credentials.txt` or MSAL ConfidentialClientApplication."""
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
        if not token_result or "access_token" not in token_result:
            error_desc = (
                token_result.get("error_description")
                or token_result.get("error")
                or "Unknown MSAL error"
                if isinstance(token_result, dict)
                else "No token response returned"
            )
            raise RuntimeError(f"Microsoft Entra authentication failed: {error_desc}")

        self._access_token = str(token_result["access_token"])
        return self._access_token

    def _headers(self) -> Dict[str, str]:
        if not self._access_token:
            self.authenticate()
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }

    def _graph_get(self, url: str) -> Dict[str, Any]:
        resp = self.session.get(url, headers=self._headers(), timeout=60)
        resp.raise_for_status()
        return resp.json()

    def resolve_site_and_drive(self) -> Tuple[str, str]:
        """Resolve the SharePoint site_id and document library drive_id via Microsoft Graph."""
        if self._site_id and self._drive_id:
            return self._site_id, self._drive_id

        clean_site_path = "/" + self.config.site_path.strip("/")
        site_url = f"{GRAPH_API_BASE_URL}/sites/{self.config.hostname}:{clean_site_path}"
        site_data = self._graph_get(site_url)
        site_id = site_data.get("id")
        if not site_id:
            raise RuntimeError(f"Could not resolve SharePoint site ID from {site_url}")
        self._site_id = str(site_id)

        drives_url = f"{GRAPH_API_BASE_URL}/sites/{self._site_id}/drives"
        drives_data = self._graph_get(drives_url)
        drives = drives_data.get("value", [])

        target_name = self.config.drive_name.strip().lower()
        selected_drive_id: Optional[str] = None
        for drv in drives:
            drv_name = str(drv.get("name", "")).strip().lower()
            if drv_name == target_name or (
                target_name in {"documents", "shared documents", "gedeelde documenten"}
                and drv_name in {"documents", "shared documents", "gedeelde documenten"}
            ):
                selected_drive_id = str(drv["id"])
                break

        if not selected_drive_id and drives:
            selected_drive_id = str(drives[0]["id"])

        if not selected_drive_id:
            raise RuntimeError(
                f"Could not find SharePoint drive '{self.config.drive_name}' on site {self._site_id}"
            )

        self._drive_id = selected_drive_id
        return self._site_id, self._drive_id

    def list_remote_files(self, supported_only: bool = True) -> List[RemoteDriveItem]:
        """Traverse the SharePoint drive tree and collect file metadata without downloading binaries."""
        print(
            f"[SharePoint] Connecting to {self.config.hostname}{self.config.site_path} ...",
            flush=True,
        )
        _, drive_id = self.resolve_site_and_drive()
        print(
            f"[SharePoint] Connected (Drive: {self.config.drive_name}). Scanning remote folder tree...",
            flush=True,
        )
        root_children_url = f"{GRAPH_API_BASE_URL}/drives/{drive_id}/root/children"
        collected: List[RemoteDriveItem] = []
        self._traverse_folder(
            drive_id=drive_id,
            children_url=root_children_url,
            parent_rel_path="",
            collected=collected,
            supported_only=supported_only,
        )
        collected.sort(key=lambda item: item.relative_path)
        print(
            f"[SharePoint] Discovery complete: {len(collected)} matching files found.",
            flush=True,
        )
        return collected

    def _traverse_folder(
        self,
        drive_id: str,
        children_url: str,
        parent_rel_path: str,
        collected: List[RemoteDriveItem],
        supported_only: bool,
    ) -> None:
        next_url: Optional[str] = children_url
        while next_url:
            page = self._graph_get(next_url)
            items = page.get("value", [])
            for raw_item in items:
                name = str(raw_item.get("name", "")).strip()
                item_id = str(raw_item.get("id", "")).strip()
                if not name or not item_id or name.startswith("."):
                    continue

                if "folder" in raw_item:
                    folder_rel = f"{parent_rel_path}/{name}".strip("/") if parent_rel_path else name
                    if folder_rel.count("/") <= 1:
                        print(
                            f"[SharePoint] Scanning folder: {folder_rel} ({len(collected)} files found so far)...",
                            flush=True,
                        )
                    sub_url = f"{GRAPH_API_BASE_URL}/drives/{drive_id}/items/{item_id}/children"
                    self._traverse_folder(
                        drive_id=drive_id,
                        children_url=sub_url,
                        parent_rel_path=folder_rel,
                        collected=collected,
                        supported_only=supported_only,
                    )
                elif "file" in raw_item:
                    parsed = self._parse_drive_item(raw_item, fallback_parent=parent_rel_path)
                    if supported_only and parsed.extension not in SUPPORTED_EXTRACTION_EXTENSIONS:
                        continue
                    collected.append(parsed)

            next_url = page.get("@odata.nextLink")

    @staticmethod
    def _parse_drive_item(
        raw_item: Dict[str, Any], fallback_parent: str = ""
    ) -> RemoteDriveItem:
        """Convert a Microsoft Graph DriveItem JSON object into a RemoteDriveItem."""
        name = str(raw_item["name"]).strip()
        parent_ref = raw_item.get("parentReference") or {}
        raw_parent_path = str(parent_ref.get("path") or "")

        # Strip Graph prefix such as '/drives/{id}/root:' or '/drive/root:'
        cleaned_parent = re.sub(r"^/drives?/[^/]+/root:?|^/drive/root:?", "", raw_parent_path)
        cleaned_parent = cleaned_parent.strip("/")
        if not cleaned_parent and fallback_parent:
            cleaned_parent = fallback_parent.strip("/")

        rel_path = f"{cleaned_parent}/{name}".strip("/") if cleaned_parent else name

        file_info = raw_item.get("file") or {}
        hashes = file_info.get("hashes") or {}

        return RemoteDriveItem(
            item_id=str(raw_item["id"]),
            name=name,
            relative_path=rel_path,
            web_url=raw_item.get("webUrl"),
            parent_path=raw_parent_path or cleaned_parent,
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
        staging_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Download a single remote SharePoint file to the local temporary staging path."""
        base_staging = Path(staging_dir) if staging_dir is not None else self.config.staging_dir
        local_path = base_staging / Path(item.relative_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if item.download_url:
            resp = self.session.get(item.download_url, stream=True, timeout=120)
        else:
            _, drive_id = self.resolve_site_and_drive()
            content_url = f"{GRAPH_API_BASE_URL}/drives/{drive_id}/items/{item.item_id}/content"
            resp = self.session.get(
                content_url,
                headers=self._headers(),
                stream=True,
                timeout=120,
            )
        resp.raise_for_status()

        with local_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    fh.write(chunk)

        return local_path

    def seed_target_json(
        self,
        item: RemoteDriveItem,
        local_file_path: Union[str, Path],
        json_out_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, str]:
        """Compute SHA-256 and create the target `.json` file pre-populated with SharePoint metadata."""
        local_path = Path(local_file_path)
        target_dir = Path(json_out_dir) if json_out_dir is not None else self.config.json_out_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        file_hash_sha256 = compute_file_sha256(local_path)
        doc_id = compute_doc_id(item.relative_path)

        seeded_metadata = DocumentMetadata(
            doc_id=doc_id,
            file_name=item.name,
            file_extension=item.extension,
            file_hash_sha256=file_hash_sha256,
            sharepoint_url=item.web_url,
            relative_folder_path=item.relative_folder_path,
            sharepoint_item_id=item.item_id,
            sharepoint_etag=item.etag or item.ctag,
            extraction_method=_default_extraction_method_for_extension(item.extension),
        )
        seeded_doc = ExtractedDocument(metadata=seeded_metadata, blocks=[])

        json_path = target_dir / f"{doc_id}.json"
        json_path.write_text(seeded_doc.model_dump_json(indent=2), encoding="utf-8")
        return json_path, file_hash_sha256

    def download_and_seed(
        self,
        item: RemoteDriveItem,
        staging_dir: Optional[Union[str, Path]] = None,
        json_out_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path, str]:
        """Download a single remote file and seed its target `.json` with SharePoint provenance."""
        local_path = self.download_file(item, staging_dir=staging_dir)
        seeded_json_path, file_hash = self.seed_target_json(
            item=item,
            local_file_path=local_path,
            json_out_dir=json_out_dir,
        )
        return local_path, seeded_json_path, file_hash

    def cleanup_local_file(
        self,
        local_file_path: Union[str, Path],
        staging_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Immediately delete a downloaded raw binary and prune any empty parent staging directories."""
        path = Path(local_file_path)
        base_staging = (
            Path(staging_dir).resolve()
            if staging_dir is not None
            else self.config.staging_dir.resolve()
        )

        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)

        # Walk upward and remove empty parent folders inside staging_dir
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
