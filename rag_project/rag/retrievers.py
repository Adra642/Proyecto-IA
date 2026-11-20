"""Retrievers (Strategy). Stage 3 has plain vector search; Stage 5 adds hybrid and rerank."""

from __future__ import annotations

from typing import Any, Protocol

from core.models import RetrievedChunk
from rag.vectorstore import VectorStoreRepository


class Retriever(Protocol):
    """Finds the chunks most relevant to a query."""

    def retrieve(
        self, query: str, k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks, best first."""
        ...


class VectorRetriever:
    """Dense retrieval: cosine similarity over the Chroma collection."""

    def __init__(self, repo: VectorStoreRepository) -> None:
        """Args: repo: the vector store repository."""
        self._repo = repo

    def retrieve(
        self, query: str, k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return the ``k`` nearest chunks, optionally filtered by metadata."""
        return self._repo.query(query, k, where)
