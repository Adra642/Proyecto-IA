"""Composition root: the only place that wires components together."""

from __future__ import annotations

import hashlib

from core.settings import Settings
from rag.chain import RagChain
from rag.chunking import Chunker, build_chunker
from rag.embeddings import Embedder, build_embedder
from rag.llm import LLMClient, build_llm
from rag.retrievers import Retriever, VectorRetriever
from rag.vectorstore import VectorStoreRepository, open_persistent_client


class Container:
    """Builds components from settings (no module-level singletons)."""

    def __init__(self, settings: Settings) -> None:
        """Args: settings: Validated project settings."""
        self.settings = settings
        self._embedder: Embedder | None = None
        self._store: VectorStoreRepository | None = None

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
        """Return the repository over the persistent Chroma collection (one per container)."""
        if self._store is None:
            client = open_persistent_client(self.settings.resolve(self.settings.paths.chroma_dir))
            self._store = VectorStoreRepository(
                client,
                self.settings.index.collection,
                self.embedder(),
                self.settings.index.batch_size,
            )
        return self._store

    def retriever(self) -> Retriever:
        """Return the retriever used by the RAG chain."""
        return VectorRetriever(self.vector_store())

    def corpus_hash(self) -> str:
        """Short hash of the indexed documents (fingerprints), used in cache keys."""
        hashes = self.vector_store().doc_hashes()
        blob = "".join(f"{k}:{v}" for k, v in sorted(hashes.items()))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def model_id(self) -> str:
        """``provider:model`` of the primary LLM (used in logs and cache keys)."""
        provider = self.settings.llm.provider
        return f"{provider}:{self.settings.llm.models[provider]}"

    def chain(self) -> RagChain:
        """Return the chain (LLM + retriever) used by the evaluation and the app."""
        return RagChain(
            self.llm(), self.retriever(), self.settings.retrieval.top_k, self.model_id()
        )
