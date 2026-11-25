"""Composition root: the only place that wires components together."""

from __future__ import annotations

from core.settings import Settings
from rag.chain import RagChain
from rag.chunking import Chunker, build_chunker
from rag.embeddings import Embedder, build_embedder
from rag.lexical import BM25Retriever
from rag.llm import LLMClient, build_llm
from rag.rerank import FlashRankReranker
from rag.retrievers import (
    HybridRetriever,
    RerankingRetriever,
    Retriever,
    VectorRetriever,
)
from rag.router import RuleRouter
from rag.vectorstore import VectorStoreRepository, open_persistent_client


class Container:
    """Builds components from settings (no module-level singletons)."""

    def __init__(self, settings: Settings) -> None:
        """Args: settings: Validated project settings."""
        self.settings = settings
        self._embedder: Embedder | None = None
        self._store: VectorStoreRepository | None = None
        self._bm25: BM25Retriever | None = None
        self._rerankers: dict[str, FlashRankReranker] = {}

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

    def retriever(self, kind: str = "vector", rerank_model: str | None = None) -> Retriever:
        """Build a retriever.

        Args:
            kind: ``vector``, ``hybrid`` (+BM25), ``hybrid_rerank`` (+FlashRank) or ``advanced``
                (hybrid, plus reranking only when ``retrieval.rerank_enabled`` is true).
            rerank_model: FlashRank model for ``hybrid_rerank`` (default: the configured one).

        The BM25 index is shared between calls on the same container (it is slow to build).
        """
        vector = VectorRetriever(self.vector_store())
        if kind == "vector":
            return vector
        cfg = self.settings.retrieval
        hybrid = HybridRetriever(
            vector,
            self._lexical(),
            cfg.candidates,
            cfg.vector_weight,
            cfg.bm25_weight,
            cfg.rrf_k,
        )
        if kind == "advanced":
            kind = "hybrid_rerank" if cfg.rerank_enabled else "hybrid"
        if kind == "hybrid":
            return hybrid
        if kind == "hybrid_rerank":
            reranker = self._reranker(rerank_model or cfg.rerank_model)
            return RerankingRetriever(hybrid, reranker, cfg.candidates)
        raise ValueError(f"unknown retriever kind: {kind}")

    def _lexical(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = BM25Retriever(self.vector_store().all_chunks)
        return self._bm25

    def _reranker(self, model_name: str) -> FlashRankReranker:
        if model_name not in self._rerankers:
            cache = self.settings.resolve(self.settings.retrieval.rerank_cache_dir)
            self._rerankers[model_name] = FlashRankReranker(model_name, cache)
        return self._rerankers[model_name]

    def model_id(self) -> str:
        """``provider:model`` of the primary LLM (used in logs)."""
        provider = self.settings.llm.provider
        return f"{provider}:{self.settings.llm.models[provider]}"

    def chain(self) -> RagChain:
        """Return the chain (LLM + retriever) used by the evaluation and the app."""
        return RagChain(
            self.llm(),
            self.retriever(),
            self.settings.retrieval.top_k,
            self.model_id(),
            advanced_retriever=self.retriever("advanced"),
            router=RuleRouter(),
        )
