import pytest

from core.settings import Provider, load_settings
from rag.llm import ADAPTERS

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("provider", ["gemini", "groq", "deepseek"])
def test_provider_answers_one_call(provider: Provider) -> None:
    settings = load_settings()
    key = settings.api_key_for(provider)
    if key is None or not key.get_secret_value().strip():
        pytest.skip(f"No API key for {provider}")
    answer = ADAPTERS[provider](settings).invoke("Reply with the single word: pong")
    assert answer.strip()
