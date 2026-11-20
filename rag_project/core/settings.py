"""Typed settings: YAML defaults, environment overrides, secrets from ``.env``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from core.exceptions import ConfigError
from core.models import SourceType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

Provider = Literal["gemini", "groq", "deepseek"]


class PathsSettings(BaseModel):
    """Project directories (relative to the project root)."""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    logs_dir: Path = Path("logs")
    chroma_dir: Path = Path("chroma_db")


class HttpSettings(BaseModel):
    """Polite-crawling options for the data pipeline."""

    user_agent: str
    min_interval_seconds: float = Field(default=1.0, ge=1.0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=4, ge=0)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    respect_robots: bool = True


class LLMSettings(BaseModel):
    """LLM provider selection and parameters."""

    provider: Provider = "gemini"
    fallback_order: list[Provider] = Field(default_factory=list)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_retries: int = Field(default=2, ge=0)
    min_interval_seconds: float = Field(default=1.0, ge=0.0)  # client-side throttling
    rate_limit_retries: int = Field(default=3, ge=0)
    backoff_seconds: float = Field(default=10.0, ge=0.0)
    models: dict[Provider, str]


class EmbeddingSettings(BaseModel):
    """Embedding model and the e5 prefixes."""

    model: str = "intfloat/multilingual-e5-small"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


def _default_atomic() -> list[SourceType]:
    return ["dataset_stats", "macro_data"]


class ChunkingSettings(BaseModel):
    """Chunking strategy; atomic source types are never split."""

    strategy: Literal["markdown_recursive", "fixed"] = "markdown_recursive"
    chunk_size: int = Field(default=600, ge=50)
    chunk_overlap: int = Field(default=90, ge=0)
    atomic_source_types: list[SourceType] = Field(default_factory=lambda: _default_atomic())

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingSettings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class IndexSettings(BaseModel):
    """Vector index options."""

    collection: str = "rag_churn"
    batch_size: int = Field(default=32, ge=1)


class RetrievalSettings(BaseModel):
    """How many chunks are passed to the LLM."""

    top_k: int = Field(default=4, ge=1)


class Settings(BaseSettings):
    """All project settings.

    Priority (high to low): environment variables, ``.env``, ``config.yaml``.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        yaml_file=DEFAULT_CONFIG,
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    paths: PathsSettings = PathsSettings()
    http: HttpSettings
    llm: LLMSettings
    embeddings: EmbeddingSettings = EmbeddingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    index: IndexSettings = IndexSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    sources: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Shortcut env var (LLM_PROVIDER) that overrides ``llm.provider``.
    llm_provider: Provider | None = None

    # Secrets: read from the environment / .env, never from YAML, never logged.
    google_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    kaggle_username: str | None = None
    kaggle_key: SecretStr | None = None
    hf_token: SecretStr | None = None

    @model_validator(mode="after")
    def _apply_provider_shortcut(self) -> Settings:
        if self.llm_provider is not None:
            self.llm.provider = self.llm_provider
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order sources: init > env > .env > YAML."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    def resolve(self, path: Path) -> Path:
        """Resolve a project-relative path against the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path

    def api_key_for(self, provider: Provider) -> SecretStr | None:
        """Return the API key configured for ``provider`` (or ``None``)."""
        return {
            "gemini": self.google_api_key,
            "groq": self.groq_api_key,
            "deepseek": self.deepseek_api_key,
        }[provider]


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate settings.

    Args:
        config_path: YAML file to use; defaults to ``RAG_CONFIG`` env var or
            ``config.yaml`` at the project root.

    Returns:
        Validated ``Settings``.

    Raises:
        ConfigError: If the file is missing or any value is invalid.
    """
    path = config_path or Path(os.environ.get("RAG_CONFIG", DEFAULT_CONFIG))
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    class _Loaded(Settings):
        model_config = SettingsConfigDict(**{**Settings.model_config, "yaml_file": path})

    try:
        return _Loaded()
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigError(f"Invalid configuration in {path}: {details}") from exc
