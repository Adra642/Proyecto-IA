from pathlib import Path

import pytest

from core.exceptions import ConfigError
from core.settings import Settings, load_settings


def test_defaults_load(settings: Settings) -> None:
    assert settings.llm.provider == "gemini"
    assert settings.embeddings.query_prefix == "query: "
    assert settings.http.min_interval_seconds >= 1.0


def test_env_shortcut_overrides_provider(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_PROVIDER", "groq")
    assert load_settings().llm.provider == "groq"


def test_secret_not_in_repr(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("GOOGLE_API_KEY", "super-secret")
    assert "super-secret" not in repr(load_settings())


def test_invalid_provider_has_clear_message(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ConfigError, match="llm_provider"):
        load_settings()


def test_min_interval_below_one_second_rejected(
    tmp_path: Path, clean_env: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "http: {user_agent: x, min_interval_seconds: 0.1}\n"
        "llm: {models: {gemini: a, groq: b, deepseek: c}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="min_interval_seconds"):
        load_settings(cfg)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml")
