"""Composition root: the only place that wires components together."""

from __future__ import annotations

from core.settings import Settings
from rag.chunking import Chunker, build_chunker
from rag.embeddings import Embedder, build_embedder
from rag.llm import LLMClient, build_llm
from rag.vectorstore import VectorStoreRepository, open_persistent_client


class Container:
    """Builds components from settings (no module-level singletons)."""

    def __init__(self, settings: Settings) -> None:
        """Args: settings: Validated project settings."""
        self.settings = settings
        self._embedder: Embedder | None = None

    def llm(self) -> LLMClient:
        """Return the configured LLM client."""
        return build_llm(self.settings)

    def embedder(self) -> Embedder:
        """Return the embedder (loaded once per container: the model is slow to load)."""
        if self._embedder is None:
            self._embedder = build_embedder(self.settings)
        return self._embedder

    def chunker(self) -> Chunker:
        """Return the configured chunker."""
        return build_chunker(self.settings.chunking)

    def vector_store(self) -> VectorStoreRepository:
        """Return the repository over the persistent Chroma collection."""
        client = open_persistent_client(self.settings.resolve(self.settings.paths.chroma_dir))
        return VectorStoreRepository(
            client, self.settings.index.collection, self.embedder(), self.settings.index.batch_size
        )
