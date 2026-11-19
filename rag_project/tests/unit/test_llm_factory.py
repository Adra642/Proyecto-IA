from typing import Any

import pytest

from core.exceptions import MissingCredentialError, ProviderError, RateLimitError
from rag import llm
from rag.llm import FallbackLLM, build_llm


class FakeLLM:
    def __init__(
        self, name: str, answer: str | None = None, error: Exception | None = None
    ) -> None:
        self.name = name
        self._answer = answer
        self._error = error
        self.calls = 0

    def invoke(self, prompt: str) -> str:
        self.calls += 1
        if self._error:
            raise self._error
        return f"{self._answer}:{prompt}"


class FakeChat:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def invoke(self, prompt: str) -> Any:
        raise RuntimeError("not used")


@pytest.mark.parametrize(
    ("provider", "env_var", "target", "adapter"),
    [
        (
            "gemini",
            "GOOGLE_API_KEY",
            "langchain_google_genai.ChatGoogleGenerativeAI",
            llm.GeminiAdapter,
        ),
        ("groq", "GROQ_API_KEY", "langchain_groq.ChatGroq", llm.GroqAdapter),
        ("deepseek", "DEEPSEEK_API_KEY", "langchain_deepseek.ChatDeepSeek", llm.DeepSeekAdapter),
    ],
)
def test_builds_right_adapter(
    clean_env: pytest.MonkeyPatch, provider: str, env_var: str, target: str, adapter: type
) -> None:
    from core.settings import load_settings

    clean_env.setenv("LLM_PROVIDER", provider)
    clean_env.setenv(env_var, "k")
    clean_env.setattr(target, FakeChat)
    client = build_llm(load_settings())
    assert isinstance(client, adapter)
    assert client.name == provider


@pytest.mark.parametrize(
    ("provider", "env_var"),
    [("gemini", "GOOGLE_API_KEY"), ("groq", "GROQ_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY")],
)
def test_missing_key_names_variable(
    clean_env: pytest.MonkeyPatch, provider: str, env_var: str
) -> None:
    from core.settings import load_settings

    clean_env.setenv("LLM_PROVIDER", provider)
    with pytest.raises(MissingCredentialError, match=env_var):
        build_llm(load_settings())


def test_fallback_follows_order_on_rate_limit() -> None:
    first = FakeLLM("a", error=RateLimitError("429"))
    second = FakeLLM("b", answer="ok")
    third = FakeLLM("c", answer="never")
    assert FallbackLLM([first, second, third]).invoke("q") == "ok:q"
    assert (first.calls, second.calls, third.calls) == (1, 1, 0)


def test_fallback_all_fail() -> None:
    clients = [FakeLLM("a", error=ProviderError("x")), FakeLLM("b", error=RateLimitError("y"))]
    with pytest.raises(ProviderError, match="All LLM providers failed"):
        FallbackLLM(clients).invoke("q")


def test_build_llm_uses_fallback_order_and_skips_missing_keys(
    clean_env: pytest.MonkeyPatch,
) -> None:
    from core.settings import load_settings

    clean_env.setenv("GOOGLE_API_KEY", "k")
    clean_env.setenv("GROQ_API_KEY", "k")  # deepseek key deliberately missing
    clean_env.setenv("LLM__FALLBACK_ORDER", '["groq", "deepseek"]')
    clean_env.setattr("langchain_google_genai.ChatGoogleGenerativeAI", FakeChat)
    clean_env.setattr("langchain_groq.ChatGroq", FakeChat)
    client = build_llm(load_settings())
    assert isinstance(client, FallbackLLM)
    assert client.name == "gemini+groq"
