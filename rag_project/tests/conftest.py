"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import Settings, load_settings


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove provider/secret env vars so tests don't depend on the developer's shell."""
    for var in (
        "LLM_PROVIDER",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "DEEPSEEK_API_KEY",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    return monkeypatch


@pytest.fixture
def settings(clean_env: pytest.MonkeyPatch) -> Settings:
    """Default settings loaded from the project's config.yaml, without .env."""
    return load_settings()


@pytest.fixture
def fixtures_dir() -> Path:
    """Directory with small saved inputs."""
    return Path(__file__).parent / "fixtures"
