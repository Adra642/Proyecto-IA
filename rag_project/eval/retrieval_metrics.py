"""Retrieval metrics: hit@k and reciprocal rank against gold documents and gold facts."""

from __future__ import annotations

from collections.abc import Sequence

from eval.numbers import contains_number
from eval.questions import GoldNumber


def first_gold_rank(ranked_doc_ids: Sequence[str], gold: set[str]) -> int | None:
    """1-based rank of the first retrieved chunk whose document is gold (``None`` if absent)."""
    for rank, doc_id in enumerate(ranked_doc_ids, 1):
        if doc_id in gold:
            return rank
    return None


def hit_at_k(ranked_doc_ids: Sequence[str], gold: set[str], k: int) -> bool:
    """True if any of the top-``k`` chunks belongs to a gold document."""
    rank = first_gold_rank(ranked_doc_ids, gold)
    return rank is not None and rank <= k


def reciprocal_rank(ranked_doc_ids: Sequence[str], gold: set[str]) -> float:
    """``1 / rank`` of the first gold chunk, 0 if none was retrieved."""
    rank = first_gold_rank(ranked_doc_ids, gold)
    return 0.0 if rank is None else 1.0 / rank


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean (0 for an empty sequence)."""
    return sum(values) / len(values) if values else 0.0


def first_true_rank(flags: Sequence[bool]) -> int | None:
    """1-based position of the first ``True`` flag (``None`` if there is none)."""
    for rank, flag in enumerate(flags, 1):
        if flag:
            return rank
    return None


def chunk_has_fact(
    chunk_doc_id: str, chunk_text: str, gold_docs: set[str], numbers: Sequence[GoldNumber]
) -> bool:
    """True if the chunk belongs to a gold document **and** contains a gold number.

    Stricter than a document-level hit: the right document can be retrieved while the chunk
    holding the figure is not, which is exactly what a reranker should fix.
    """
    return chunk_doc_id in gold_docs and any(
        contains_number(chunk_text, n.value, n.tolerance, n.percent) for n in numbers
    )
