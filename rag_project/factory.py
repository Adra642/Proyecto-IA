"""Composition root: the only place that wires components together."""

from __future__ import annotations

from core.settings import Settings
from rag.embeddings import Embedder, build_embedder
from rag.llm import LLMClient, build_llm


class Container:
    """Builds components lazily from settings (no module-level singletons)."""

    def __init__(self, settings: Settings) -> None:
        """Args: settings: Validated project settings."""
        self.settings = settings

    def llm(self) -> LLMClient:
        """Return the configured LLM client."""
        return build_llm(self.settings)

    def embedder(self) -> Embedder:
        """Return the configured embedder."""
        return build_embedder(self.settings)
