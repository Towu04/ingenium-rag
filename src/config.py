"""Runtime settings for the RAG side of the project (env / `.env` + `configs/*.yaml`)."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/config.py -> repository root (overridable when installed outside the repo, e.g. Docker)
ROOT = Path(os.environ.get("INGENIUM_ROOT", Path(__file__).resolve().parents[1]))


class Settings(BaseSettings):
    # Storage
    qdrant_path: Path = ROOT / "data" / "qdrant_poc"
    processed_dir: Path = ROOT / "data" / "processed"

    # Models
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: str | None = None

    # Retrieval
    retrieve_k: int = 15
    child_candidate_limit: int = 40
    rerank_k: int = 5

    # Local LLM (OpenAI-compatible server such as Ollama / vLLM). Unset -> extractive stub.
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None

    config_dir: Path = ROOT / "configs"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def yaml_config(self, name: str) -> dict[str, Any]:
        path = self.config_dir / name
        with path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration file must contain a mapping: {path}")
        return data


@lru_cache
def get_settings() -> Settings:
    return Settings()
