"""Centralized configuration, paths, environment loading, and compute-device selection.

Single source of truth for all stages (`etl`, `chunking`, `embedding`, `storage`,
`retrieval`, `generation`, `api`). Imports only standard library and `pydantic-settings`.
"""

import os
from functools import lru_cache
from pathlib import Path

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
    sharepoint_drive_name: str = "Documents"

    # --- ETL Extraction & Quality Thresholds ---
    default_project_name: str = "AZ Sint-Jan Brugge"
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
    reranker_min_score: float = 0.05
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
