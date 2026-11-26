"""Retrievers (Strategy): vector, hybrid (BM25 + vector fusion) and reranking decorator."""

from __future__ import annotations

from typing import Any, Protocol

from core.models import RetrievedChunk
from rag.rerank import Reranker
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


class HybridRetriever:
    """Weighted reciprocal-rank fusion of a dense and a lexical retriever.

    Ranks, not raw scores, are fused: cosine similarity and BM25 live on different scales.
    A chunk's fused score is ``sum(weight / (rrf_k + rank))`` over the lists that return it.
    """

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        candidates: int,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        """Args: dense/lexical: the two retrievers; candidates: depth fetched from each."""
        self._dense = dense
        self._lexical = lexical
        self._candidates = candidates
        self._weights = (vector_weight, bm25_weight)
        self._rrf_k = rrf_k

    def retrieve(
        self, query: str, k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return the ``k`` chunks with the best fused rank."""
        # Take more candidates than k from each retriever so the fusion has material to work with.
        depth = max(k, self._candidates)
        fused: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}
        for weight, retriever in zip(self._weights, (self._dense, self._lexical), strict=True):
            for rank, chunk in enumerate(retriever.retrieve(query, depth, where), 1):
                # Reciprocal Rank Fusion: weight / (rrf_k + rank). It only uses ranks, so cosine
                # and BM25 scores (different scales) never have to be compared.
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + weight / (
                    self._rrf_k + rank
                )
                chunks.setdefault(chunk.chunk_id, chunk)  # a chunk found by both is kept once
        best = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:k]
        return [chunks[cid].model_copy(update={"score": fused[cid]}) for cid in best]


class RerankingRetriever:
    """Decorator: fetch many candidates from a base retriever, rerank them, keep the top ``k``."""

    def __init__(self, base: Retriever, reranker: Reranker, candidates: int) -> None:
        """Args: base: candidate source; reranker: cross-encoder; candidates: pool size."""
        self._base = base
        self._reranker = reranker
        self._candidates = candidates

    def retrieve(
        self, query: str, k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return the ``k`` best candidates after reranking."""
        pool = self._base.retrieve(query, max(k, self._candidates), where)
        return self._reranker.rerank(query, pool)[:k]
