"""Centralized configuration, paths, environment loading, and compute-device selection.

Single source of truth for all stages (`etl`, `chunking`, `embedding`, `storage`,
`retrieval`, `generation`, `api`). Imports only standard library and `pydantic-settings`.
"""

import logging
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/config.py -> repository root (overrideable via INGENIUM_ROOT in containers)
ROOT: Path = Path(os.environ.get("INGENIUM_ROOT", Path(__file__).resolve().parents[1]))

_CREDENTIAL_KEY_ALIASES: dict[str, str] = {
    "TENANT_ID": "AZURE_TENANT_ID",
    "CLIENT_ID": "AZURE_CLIENT_ID",
    "CLIENT_SECRET": "AZURE_CLIENT_SECRET",
}

SUPPORTED_EXTRACTION_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".xlsx"})
GRAPH_API_BASE_URL: str = "https://graph.microsoft.com/v1.0"
GRAPH_DEFAULT_SCOPE: str = "https://graph.microsoft.com/.default"


def load_credentials_file(
    credentials_path: Path | str | None = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    """Load `KEY=VALUE` pairs from `credentials.txt` and/or `.env` into `os.environ`."""
    candidates = (
        [Path(credentials_path)]
        if credentials_path is not None
        else [ROOT / "credentials.txt", ROOT / ".env"]
    )
    loaded: dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            raw_key, _, raw_val = line.partition("=")
            key = raw_key.strip()
            val = raw_val.strip().strip("'").strip('"')
            if not key or not val or val.upper().startswith("YOUR_"):
                continue
            canonical = _CREDENTIAL_KEY_ALIASES.get(key.upper(), key)
            loaded[canonical] = val
            if override or canonical not in os.environ:
                os.environ[canonical] = val
    return loaded


# Ensure `.env` / `credentials.txt` keys (including TENANT_ID aliases) are available early.
load_credentials_file()


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables and `.env`."""

    # --- Storage & Directory Paths ---
    data_dir: Path = ROOT / "data"
    catalog_path: Path = ROOT / "data" / "catalog" / "projects_catalog.csv"
    raw_sharepoint_dir: Path = ROOT / "data" / "raw_sharepoint"
    processed_dir: Path = ROOT / "data" / "processed"
    logs_dir: Path = ROOT / "data" / "logs"
    qdrant_path: Path = ROOT / "data" / "qdrant_poc"

    # --- Microsoft Entra ID & SharePoint Online ---
    azure_tenant_id: str = ""
    azure_client_id: str = ""
    azure_client_secret: str = ""
    temp_bearer_token: str = ""
    sharepoint_hostname: str = ""
    sharepoint_site_path: str = ""
    sharepoint_site_paths: str = ""
    sharepoint_sites_file: Path | None = None
    sharepoint_drive_name: str = "Documents"

    # --- ETL Extraction & Quality Thresholds ---
    default_project_name: str | None = None
    building_aliases: dict[str, str] = {}
    etl_workers: int = 4
    min_digital_text_chars_per_page: int = 50
    oversized_page_min_dimension_pt: float = 1250.0
    high_vector_drawings_threshold: int = 250
    low_confidence_threshold: float = 0.65
    sparse_diagram_char_threshold: int = 150
    callout_readability_threshold: float = 0.45

    # --- Parent-Child Chunking ---
    parent_size: int = 1000
    parent_overlap: int = 0
    child_size: int = 250
    child_overlap: int = 40

    # --- Embedding & Cross-Encoder Reranking ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 32
    embedding_max_length: int = 512
    embed_batch_buffer_size: int = 256
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = 16
    reranker_max_length: int = 2048
    reranker_min_score: float = 0.0
    device: str | None = None

    # --- Retrieval ---
    retrieve_k: int = 15
    child_candidate_limit: int = 80
    rerank_k: int = 5

    # --- Local LLM (OpenAI-compatible server e.g. Ollama / vLLM). Unset -> ExtractiveStubLLM. ---
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None

    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

    @property
    def markdown_out_dir(self) -> Path:
        return self.processed_dir / "markdown"

    @property
    def json_out_dir(self) -> Path:
        return self.processed_dir / "json"

    @property
    def manifest_path(self) -> Path:
        return self.processed_dir / "manifest.json"

    @property
    def limitations_report_path(self) -> Path:
        return self.logs_dir / "limitations_report.jsonl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton `Settings` instance."""
    return Settings()


@dataclass(frozen=True)
class SharePointSiteSpec:
    """Resolved SharePoint site target (from TSV/CSV sites file, CLI --site, or .env)."""

    hostname: str
    site_path: str
    project_code: str | None = None
    folder_prefix: str | None = None

    @property
    def site_slug(self) -> str:
        return self.site_path.strip("/").split("/")[-1]

    @property
    def key(self) -> str:
        return f"{self.hostname.lower()}/{self.site_path.strip('/').lower()}"


_SP_SITE_URL_RE = re.compile(
    r"^(/sites/[^/]+)(?:/(?:Shared Documents|Documents|Gedeelde documenten)(?:/(.*))?)?$",
    re.IGNORECASE,
)


def parse_sharepoint_site_spec(
    raw: str,
    *,
    default_hostname: str = "",
    project_code: str | None = None,
    folder_prefix_override: str | None = None,
) -> SharePointSiteSpec:
    """Parse a SharePoint site URL or `/sites/<slug>` path into a `SharePointSiteSpec`."""
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("SharePoint site specification cannot be empty.")

    hostname = default_hostname.strip()
    url_prefix: str | None = None

    if "://" in cleaned:
        parsed = urlparse(cleaned)
        hostname = parsed.netloc.strip() or hostname
        path_decoded = unquote(parsed.path).rstrip("/")
        m = _SP_SITE_URL_RE.match(path_decoded)
        if m:
            site_path = m.group(1)
            url_prefix = (m.group(2) or "").strip("/") or None
        else:
            site_path = path_decoded if path_decoded.startswith("/") else f"/{path_decoded}"
    else:
        path_decoded = unquote(cleaned).rstrip("/")
        if not path_decoded.startswith("/"):
            path_decoded = (
                f"/{path_decoded}"
                if path_decoded.lower().startswith("sites/")
                else f"/sites/{path_decoded}"
            )
        m = _SP_SITE_URL_RE.match(path_decoded)
        if m:
            site_path = m.group(1)
            url_prefix = (m.group(2) or "").strip("/") or None
        else:
            site_path = path_decoded

    return SharePointSiteSpec(
        hostname=hostname,
        site_path=site_path,
        project_code=(project_code.strip() or None) if project_code else None,
        folder_prefix=(
            folder_prefix_override.strip("/")
            if folder_prefix_override is not None and folder_prefix_override.strip("/")
            else url_prefix
        ),
    )


def load_sharepoint_sites(
    *,
    sites_file: Path | str | None = None,
    site_override: str | None = None,
    folder_prefix: str | None = None,
    settings: Settings | None = None,
) -> list[SharePointSiteSpec]:
    """Resolve one or more SharePoint sites from CLI `--site`, `--sites-file`, or `.env`."""
    cfg = settings or get_settings()
    default_host = cfg.sharepoint_hostname.strip()

    if site_override and site_override.strip():
        return [
            parse_sharepoint_site_spec(
                site_override,
                default_hostname=default_host,
                folder_prefix_override=folder_prefix,
            )
        ]

    resolved_file = Path(sites_file) if sites_file is not None else cfg.sharepoint_sites_file
    specs: list[SharePointSiteSpec] = []

    if resolved_file is not None:
        if not resolved_file.exists():
            raise FileNotFoundError(f"SharePoint sites file not found: {resolved_file}")
        for raw_line in resolved_file.read_text(encoding="utf-8").splitlines():
            line_no_nl = raw_line.rstrip("\r\n")
            if not line_no_nl.strip() or line_no_nl.lstrip().startswith("#"):
                continue
            parts = (
                line_no_nl.split("\t")
                if "\t" in line_no_nl
                else [p.strip() for p in line_no_nl.split(",", 1)]
            )
            if len(parts) == 1:
                proj_code, raw_url = None, parts[0].strip()
            else:
                proj_code, raw_url = parts[0].strip() or None, parts[1].strip()
            if not raw_url or raw_url.lower() in {"url", "sharepoint_url", "site_url", "site_path"}:
                continue
            specs.append(
                parse_sharepoint_site_spec(
                    raw_url,
                    default_hostname=default_host,
                    project_code=proj_code,
                    folder_prefix_override=folder_prefix,
                )
            )
    elif cfg.sharepoint_site_paths.strip():
        for entry in cfg.sharepoint_site_paths.split(","):
            if entry.strip():
                specs.append(
                    parse_sharepoint_site_spec(
                        entry,
                        default_hostname=default_host,
                        folder_prefix_override=folder_prefix,
                    )
                )
    elif cfg.sharepoint_site_path.strip():
        specs.append(
            parse_sharepoint_site_spec(
                cfg.sharepoint_site_path,
                default_hostname=default_host,
                folder_prefix_override=folder_prefix,
            )
        )

    # Deduplicate identical (hostname, site_path) preserving first non-empty project_code.
    deduped: dict[str, SharePointSiteSpec] = {}
    for spec in specs:
        existing = deduped.get(spec.key)
        if existing is None:
            deduped[spec.key] = spec
        elif existing.project_code is None and spec.project_code is not None:
            deduped[spec.key] = spec
    return list(deduped.values())


def detect_device(preferred: str | None = None) -> str:
    """Select preferred device or auto-detect CUDA > Apple Silicon MPS > CPU."""
    if preferred:
        return preferred
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_NOISY_THIRD_PARTY_LOGGERS: tuple[str, ...] = (
    "transformers",
    "huggingface_hub",
    "FlagEmbedding",
    "torch",
    "httpx",
    "httpcore",
    "urllib3",
    "qdrant_client",
    "sentence_transformers",
    "filelock",
    "msal",
    "datasets",
)


def configure_logging(
    *,
    verbose: bool = False,
    quiet: bool = False,
    json_mode: bool = False,
    default_level: int = logging.WARNING,
    disable_progress: bool | None = None,
    handler: logging.Handler | None = None,
) -> None:
    """Configure consistent stderr-only logging and silence third-party chatter unless verbose."""
    if verbose:
        level = logging.DEBUG
    elif quiet or json_mode:
        level = logging.ERROR
    else:
        level = default_level

    active_handler = handler
    if active_handler is None:
        active_handler = logging.StreamHandler(sys.stderr)
        active_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
            )
        )
    logging.basicConfig(level=level, handlers=[active_handler], force=True)

    third_party_level = logging.INFO if verbose else logging.ERROR
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)

    should_disable_progress = (
        disable_progress
        if disable_progress is not None
        else (not verbose and (quiet or json_mode or default_level >= logging.WARNING))
    )
    if should_disable_progress:
        os.environ.setdefault("TQDM_DISABLE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
