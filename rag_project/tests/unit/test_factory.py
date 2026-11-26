from core.settings import Settings
from factory import Container


def test_provider_override_selects_that_model_without_fallbacks(settings: Settings) -> None:
    container = Container(settings)
    chosen = container._settings_for("groq")
    assert chosen.llm.provider == "groq" and chosen.llm.fallback_order == []
    assert container.model_id("groq") == f"groq:{settings.llm.models['groq']}"
    assert container._settings_for(None) is settings
    assert (
        container.model_id()
        == f"{settings.llm.provider}:{settings.llm.models[settings.llm.provider]}"
    )
