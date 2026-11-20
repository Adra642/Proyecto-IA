"""LLM provider adapters, factory and fallback decorator."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic import SecretStr

from core.exceptions import MissingCredentialError, ProviderError, RateLimitError
from core.logging_setup import get_logger
from core.settings import Provider, Settings

log = get_logger(__name__)

_ENV_VARS: dict[Provider, str] = {
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


class LLMClient(Protocol):
    """Minimal interface the rest of the code depends on."""

    name: str

    def invoke(self, prompt: str) -> str:
        """Return the model's text answer for ``prompt``."""
        ...


def _is_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    text = f"{type(exc).__name__} {exc}".lower()
    return status == 429 or any(
        marker in text for marker in ("ratelimit", "rate limit", "resource_exhausted", "quota")
    )


class LangChainAdapter:
    """Adapts a LangChain chat model to :class:`LLMClient`."""

    def __init__(self, name: str, chat_model: Any) -> None:
        """Wrap ``chat_model`` (anything with ``invoke(str) -> AIMessage``)."""
        self.name = name
        self._chat_model = chat_model

    def invoke(self, prompt: str) -> str:
        """Call the model and normalise errors.

        Raises:
            RateLimitError: On HTTP 429 / quota errors.
            ProviderError: On any other provider failure.
        """
        try:
            response = self._chat_model.invoke(prompt)
        except Exception as exc:
            if _is_rate_limit(exc):
                raise RateLimitError(f"{self.name} rate-limited: {exc}") from exc
            raise ProviderError(f"{self.name} call failed: {exc}") from exc
        text = getattr(response, "text", None)
        return text if isinstance(text, str) else str(response.content)


def _require_key(settings: Settings, provider: Provider) -> SecretStr:
    key = settings.api_key_for(provider)
    if key is None or not key.get_secret_value():
        raise MissingCredentialError(
            f"Provider '{provider}' is selected but {_ENV_VARS[provider]} is not set. "
            "Add it to .env or the environment."
        )
    return key


class GeminiAdapter(LangChainAdapter):
    """Google Gemini via ``langchain-google-genai``."""

    @classmethod
    def create(cls, settings: Settings) -> GeminiAdapter:
        """Build the adapter from settings."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=settings.llm.models["gemini"],
            google_api_key=_require_key(settings, "gemini"),
            temperature=settings.llm.temperature,
            max_retries=settings.llm.max_retries,
        )
        return cls("gemini", model)


class GroqAdapter(LangChainAdapter):
    """Groq via ``langchain-groq``."""

    @classmethod
    def create(cls, settings: Settings) -> GroqAdapter:
        """Build the adapter from settings."""
        from langchain_groq import ChatGroq

        model = ChatGroq(
            model_name=settings.llm.models["groq"],
            api_key=_require_key(settings, "groq"),
            temperature=settings.llm.temperature,
            max_retries=settings.llm.max_retries,
        )
        return cls("groq", model)


class DeepSeekAdapter(LangChainAdapter):
    """DeepSeek via ``langchain-deepseek`` (pay-as-you-go)."""

    @classmethod
    def create(cls, settings: Settings) -> DeepSeekAdapter:
        """Build the adapter from settings."""
        from langchain_deepseek import ChatDeepSeek

        model = ChatDeepSeek(
            model_name=settings.llm.models["deepseek"],
            api_key=_require_key(settings, "deepseek"),
            temperature=settings.llm.temperature,
            max_retries=settings.llm.max_retries,
        )
        return cls("deepseek", model)


ADAPTERS: dict[Provider, Callable[[Settings], LLMClient]] = {
    "gemini": GeminiAdapter.create,
    "groq": GroqAdapter.create,
    "deepseek": DeepSeekAdapter.create,
}


class FallbackLLM:
    """Decorator: try each client in order when one is rate-limited or fails."""

    def __init__(self, clients: Sequence[LLMClient]) -> None:
        """Args: clients: Ordered clients; the first is the preferred one."""
        if not clients:
            raise ValueError("FallbackLLM needs at least one client")
        self._clients = list(clients)
        self.name = "+".join(c.name for c in clients)

    def invoke(self, prompt: str) -> str:
        """Return the first successful answer.

        Raises:
            ProviderError: If every client fails.
        """
        errors: list[str] = []
        for client in self._clients:
            try:
                return client.invoke(prompt)
            except ProviderError as exc:
                log.warning("LLM %s failed (%s); trying next provider", client.name, exc)
                errors.append(str(exc))
        raise ProviderError("All LLM providers failed: " + " | ".join(errors))


class ThrottledLLM:
    """Decorator: at least ``min_interval`` seconds between two calls."""

    def __init__(
        self,
        inner: LLMClient,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Args: inner: wrapped client; min_interval: seconds; clock/sleep: injectable."""
        self._inner = inner
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self.name = inner.name

    def invoke(self, prompt: str) -> str:
        """Wait if the previous call was too recent, then call the wrapped client."""
        if self._last is not None:
            wait = self._min_interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()
        return self._inner.invoke(prompt)


class RetryingLLM:
    """Decorator: retry on rate limits with exponential backoff (other errors pass through)."""

    def __init__(
        self,
        inner: LLMClient,
        retries: int,
        backoff: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Args: inner: wrapped client; retries: extra attempts; backoff: base seconds."""
        self._inner = inner
        self._retries = retries
        self._backoff = backoff
        self._sleep = sleep
        self.name = inner.name

    def invoke(self, prompt: str) -> str:
        """Call the wrapped client, sleeping ``backoff * 2**attempt`` after each rate limit.

        Raises:
            RateLimitError: If every attempt was rate-limited.
        """
        for attempt in range(self._retries + 1):
            try:
                return self._inner.invoke(prompt)
            except RateLimitError as exc:
                if attempt == self._retries:
                    raise
                delay = self._backoff * 2**attempt
                log.warning("%s rate-limited (%s); retrying in %.0fs", self.name, exc, delay)
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover


def _resilient(client: LLMClient, settings: Settings) -> LLMClient:
    cfg = settings.llm
    throttled = ThrottledLLM(client, cfg.min_interval_seconds)
    return RetryingLLM(throttled, cfg.rate_limit_retries, cfg.backoff_seconds)


def build_llm(settings: Settings) -> LLMClient:
    """Build the configured LLM (with fallbacks when ``fallback_order`` is set).

    The selected provider must have its key set; fallback providers without a key
    are skipped with a warning.

    Raises:
        MissingCredentialError: If the selected provider's key is missing.
    """
    primary = settings.llm.provider
    clients: list[LLMClient] = [_resilient(ADAPTERS[primary](settings), settings)]
    for provider in settings.llm.fallback_order:
        if provider == primary:
            continue
        try:
            clients.append(_resilient(ADAPTERS[provider](settings), settings))
        except MissingCredentialError as exc:
            log.warning("Skipping fallback provider '%s': %s", provider, exc)
    return clients[0] if len(clients) == 1 else FallbackLLM(clients)
