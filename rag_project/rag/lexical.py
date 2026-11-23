"""Lexical (BM25) retrieval and the metadata-filter matcher shared with the router."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from rank_bm25 import BM25Okapi

from core.models import RetrievedChunk

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens (Unicode-aware, so accents in ES/FR/DE are kept)."""
    return _TOKEN.findall(text.lower())


def matches_where(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Evaluate a Chroma-style filter against one chunk's metadata.

    Supports equality, ``$in``, ``$and`` and ``$or``: the subset the app uses, so the lexical
    retriever filters exactly like the vector store does.
    """
    if not where:
        return True
    for key, cond in where.items():
        if key == "$and":
            if not all(matches_where(meta, c) for c in cond):
                return False
        elif key == "$or":
            if not any(matches_where(meta, c) for c in cond):
                return False
        elif isinstance(cond, dict):
            if "$in" in cond and meta.get(key) not in cond["$in"]:
                return False
            if "$eq" in cond and meta.get(key) != cond["$eq"]:
                return False
        elif meta.get(key) != cond:
            return False
    return True


def lexical_text(chunk: RetrievedChunk) -> str:
    """Text indexed by BM25: title and header path give exact-term matches their context."""
    meta = chunk.metadata
    return f"{meta.get('title', '')} {meta.get('header_path', '')} {chunk.text}"


class BM25Retriever:
    """Okapi BM25 over every chunk; the index is built on the first query."""

    def __init__(self, load_chunks: Callable[[], list[RetrievedChunk]]) -> None:
        """Args: load_chunks: returns all indexed chunks (called once, lazily)."""
        self._load = load_chunks
        self._chunks: list[RetrievedChunk] = []
        self._bm25: BM25Okapi | None = None

    def _ensure(self) -> BM25Okapi | None:
        if self._bm25 is None:
            self._chunks = self._load()
            if self._chunks:
                self._bm25 = BM25Okapi([tokenize(lexical_text(c)) for c in self._chunks])
        return self._bm25

    def retrieve(
        self, query: str, k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return the ``k`` best BM25 matches that pass ``where`` (score = BM25, > 0 only)."""
        bm25 = self._ensure()
        tokens = tokenize(query)
        if bm25 is None or not tokens:
            return []
        scores = bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[RetrievedChunk] = []
        for i in order:
            if scores[i] <= 0 or len(out) >= k:
                break
            chunk = self._chunks[i]
            if matches_where(chunk.metadata, where):
                out.append(chunk.model_copy(update={"score": float(scores[i])}))
        return out
