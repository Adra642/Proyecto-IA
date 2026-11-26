"""UI-independent logic of the web app: answering, error messages and source formatting.

Keeping this separate from Gradio makes it testable offline with a fake chain.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from core.exceptions import MissingCredentialError, ProviderError, RateLimitError
from core.logging_setup import get_logger
from core.models import RagAnswer, RetrievedChunk

log = get_logger(__name__)

MODE_NO_RAG = "Sin RAG"
MODE_RAG = "RAG"
MODE_ADVANCED = "RAG avanzado"
MODES: tuple[str, ...] = (MODE_NO_RAG, MODE_RAG, MODE_ADVANCED)

MSG_RATE_LIMIT = (
    "El proveedor del modelo está saturado (límite de peticiones). "
    "Espera unos segundos y vuelve a intentarlo."
)
MSG_NO_KEY = (
    "Falta la clave de API de {provider} en el servidor. "
    "Elige otro modelo o avisa a los responsables."
)
MSG_PROVIDER = "El modelo no respondió correctamente. Inténtalo de nuevo en un momento."
MSG_EMPTY = "Escribe una pregunta."
MAX_QUESTION_CHARS = 1000
MSG_TOO_LONG = f"La pregunta es demasiado larga (máximo {MAX_QUESTION_CHARS} caracteres)."
SNIPPET_CHARS = 700
RETRY_PAUSE_S = 3.0


class Chain(Protocol):
    """What the app needs from the RAG chain."""

    def answer_no_rag(self, question: str) -> RagAnswer:
        """Plain LLM answer."""
        ...

    def answer_rag(self, question: str) -> RagAnswer:
        """Retrieval-grounded answer."""
        ...

    def answer_advanced(
        self, question: str, k: int | None = None, where: dict[str, Any] | None = None
    ) -> RagAnswer:
        """Routed, hybrid-search answer, optionally restricted by a metadata filter."""
        ...


class AppService:
    """Runs questions through the chain and turns failures into friendly Spanish messages."""

    def __init__(
        self,
        chain_for: Callable[[str], Chain],
        default_provider: str,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Set up the service.

        Args:
            chain_for: Builds the chain of an LLM provider (called once per provider).
            default_provider: Used when a question names none.
            sleep: Pause before the single retry (injectable for tests).
        """
        self._chain_for = chain_for
        self._default = default_provider
        self._chains: dict[str, Chain] = {}
        self._sleep = sleep

    def _chain(self, provider: str) -> Chain:
        if provider not in self._chains:
            self._chains[provider] = self._chain_for(provider)
        return self._chains[provider]

    def ask(
        self,
        question: str,
        mode: str,
        where: dict[str, Any] | None = None,
        provider: str | None = None,
    ) -> tuple[str, RagAnswer | None]:
        """Answer ``question`` in ``mode`` with the LLM ``provider`` (default: the configured one).

        ``where`` only applies to the advanced mode.

        Returns:
            ``(text, answer)``. ``answer`` is ``None`` when ``text`` is an error message.
        """
        question = question.strip()
        if not question:
            return MSG_EMPTY, None
        if len(question) > MAX_QUESTION_CHARS:
            return MSG_TOO_LONG, None
        provider = provider or self._default
        try:
            result = self._run(self._chain(provider), question, mode, where)
        except MissingCredentialError:
            log.error("LLM credential missing for %s", provider)
            return MSG_NO_KEY.format(provider=provider), None
        except RateLimitError:
            return MSG_RATE_LIMIT, None
        except ProviderError as exc:
            log.error("LLM provider failed: %s", exc)
            return MSG_PROVIDER, None
        return result.answer, result

    def _run(
        self, chain: Chain, question: str, mode: str, where: dict[str, Any] | None
    ) -> RagAnswer:
        """Call the chain; retry once when the provider is rate-limited."""
        try:
            return self._call(chain, question, mode, where)
        except RateLimitError:
            log.warning("Rate limited; retrying once")
            self._sleep(RETRY_PAUSE_S)
            return self._call(chain, question, mode, where)

    @staticmethod
    def _call(chain: Chain, question: str, mode: str, where: dict[str, Any] | None) -> RagAnswer:
        if mode == MODE_NO_RAG:
            return chain.answer_no_rag(question)
        if mode == MODE_ADVANCED:
            return chain.answer_advanced(question, where=where)
        return chain.answer_rag(question)


def format_sources(chunks: list[RetrievedChunk], cited_ids: list[str]) -> str:
    """Markdown for the "Fuentes" accordion: one block per retrieved chunk."""
    if not chunks:
        return "_Sin fuentes: esta respuesta no usó documentos recuperados._"
    blocks: list[str] = []
    for rank, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        title = meta.get("title") or chunk.doc_id
        url = meta.get("url", "")
        link = f"[{title}]({url})" if url else str(title)
        open_link = f" · [abrir documento ↗]({url})" if url else ""
        cited = " · **citada en la respuesta**" if chunk.doc_id in cited_ids else ""
        text = " ".join(chunk.text.split())
        if len(text) > SNIPPET_CHARS:
            text = text[:SNIPPET_CHARS].rstrip() + "…"
        blocks.append(
            f"**{rank}. {link}**{cited}  \n"
            f"`{chunk.doc_id}` · fuente: {meta.get('source', '?')} · "
            f"tipo: {meta.get('source_type', '?')} · puntuación: {chunk.score:.3f}"
            f"{open_link}\n\n"
            f"> {text}"
        )
    return "\n\n".join(blocks)


def format_status(answer: RagAnswer | None) -> str:
    """One-line status under an answer: latency and model, or nothing for errors."""
    if answer is None:
        return ""
    note = " · rechazada por la guarda de privacidad" if answer.refused else ""
    route = f" · ruta: {answer.route}" if answer.route else ""
    return f"⏱ {answer.latency_s:.1f} s · modelo: `{answer.model}`{note}{route}"
