"""Rerankers (Strategy): reorder retrieved candidates with a cross-encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from core.models import RetrievedChunk


class Reranker(Protocol):
    """Scores candidate chunks against the query and returns them best first."""

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return ``chunks`` reordered, with ``score`` replaced by the reranker's score."""
        ...


def rerank_text(chunk: RetrievedChunk, with_title: bool = True) -> str:
    """Text shown to the cross-encoder: title and header path (optional) plus the chunk."""
    if not with_title:
        return chunk.text
    meta = chunk.metadata
    head = " > ".join(
        p for p in (str(meta.get("title", "")), str(meta.get("header_path", ""))) if p
    )
    return f"{head}\n{chunk.text}" if head else chunk.text


class FlashRankReranker:
    """Local ONNX cross-encoder (FlashRank): no API, no key. The model downloads on first use."""

    def __init__(self, model_name: str, cache_dir: Path, with_title: bool = True) -> None:
        """Args: model_name: FlashRank model id; cache_dir: model file folder.

        with_title: prepend the document title and header path to each passage.
        """
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._with_title = with_title
        self._ranker: Any = None

    def _load(self) -> Any:
        if self._ranker is None:
            from flashrank import Ranker  # imported lazily: it loads onnxruntime

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._ranker = Ranker(model_name=self._model_name, cache_dir=str(self._cache_dir))
        return self._ranker

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Score each chunk with the cross-encoder and sort by score."""
        if not chunks:
            return []
        from flashrank import RerankRequest

        by_id = {c.chunk_id: c for c in chunks}
        passages = [{"id": c.chunk_id, "text": rerank_text(c, self._with_title)} for c in chunks]
        ranked = self._load().rerank(RerankRequest(query=query, passages=passages))
        return [by_id[str(r["id"])].model_copy(update={"score": float(r["score"])}) for r in ranked]


class BlendedReranker:
    """Decorator: mix the cross-encoder's order with the order the candidates arrived in.

    A cross-encoder trained on web passages can push a correct chunk down; blending lets the
    (here very strong) hybrid ranking veto small changes while still allowing big corrections.
    Both orders are combined with reciprocal ranks, so no score scales have to be compared:
    ``weight / (k + rank_cross_encoder) + (1 - weight) / (k + rank_input)``.
    """

    def __init__(self, inner: Reranker, weight: float, rrf_k: int = 60) -> None:
        """Args: inner: the cross-encoder; weight: its share (0..1); rrf_k: rank constant."""
        if not 0.0 <= weight <= 1.0:
            raise ValueError("rerank weight must be between 0 and 1")
        self._inner = inner
        self._weight = weight
        self._k = rrf_k

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return ``chunks`` ordered by the blended score (ties keep the input order)."""
        ranked = self._inner.rerank(query, chunks)
        inner_rank = {c.chunk_id: i for i, c in enumerate(ranked, 1)}
        scored = [
            (
                self._weight / (self._k + inner_rank[c.chunk_id])
                + (1 - self._weight) / (self._k + base),
                base,
                c,
            )
            for base, c in enumerate(chunks, 1)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [c.model_copy(update={"score": s}) for s, _, c in scored]
