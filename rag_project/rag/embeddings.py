"""Embedding model adapter (multilingual-e5 needs query/passage prefixes)."""

from __future__ import annotations

from typing import Any, Protocol

from core.settings import Settings


class Embedder(Protocol):
    """Interface used by the indexer and retrievers."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed documents to be stored."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a user query."""
        ...


class E5Embedder:
    """``sentence-transformers`` model that adds the e5 prefixes."""

    def __init__(self, model: Any, query_prefix: str, passage_prefix: str) -> None:
        """Args: model: object with ``encode(list[str], normalize_embeddings=...)``."""
        self._model = model
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed passages with the ``passage: `` prefix."""
        return self._encode([self._passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the ``query: `` prefix."""
        return self._encode([self._query_prefix + text])[0]


def build_embedder(settings: Settings) -> Embedder:
    """Load the configured sentence-transformers model (CPU)."""
    from sentence_transformers import SentenceTransformer

    cfg = settings.embeddings
    return E5Embedder(
        SentenceTransformer(cfg.model, device="cpu"), cfg.query_prefix, cfg.passage_prefix
    )
