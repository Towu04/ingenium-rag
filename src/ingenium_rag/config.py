from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    sharepoint_tenant_id: str | None = None
    sharepoint_client_id: str | None = None
    sharepoint_client_secret: str | None = None
    sharepoint_site_id: str | None = None
    database_url: str = "postgresql://postgres:postgres@localhost:5432/ingenium"
    qdrant_url: str = "http://localhost:6333"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
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
