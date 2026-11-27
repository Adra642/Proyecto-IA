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


def rerank_text(chunk: RetrievedChunk) -> str:
    """Text shown to the cross-encoder: title and header path plus the chunk."""
    meta = chunk.metadata
    head = " > ".join(
        p for p in (str(meta.get("title", "")), str(meta.get("header_path", ""))) if p
    )
    return f"{head}\n{chunk.text}" if head else chunk.text


class FlashRankReranker:
    """Local ONNX cross-encoder (FlashRank): no API, no key. The model downloads on first use."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        """Args: model_name: FlashRank model id; cache_dir: where the model file is kept."""
        self._model_name = model_name
        self._cache_dir = cache_dir
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
        passages = [{"id": c.chunk_id, "text": rerank_text(c)} for c in chunks]
        ranked = self._load().rerank(RerankRequest(query=query, passages=passages))
        return [by_id[str(r["id"])].model_copy(update={"score": float(r["score"])}) for r in ranked]
