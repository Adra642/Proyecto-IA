"""Retrieval ablation: baseline vector -> +hybrid -> +rerank -> +filters (router), plus rerankers.

Every configuration is run retrieval-only (no LLM, deterministic, free) on the evaluation
questions that have gold sources, scored with hit@1, hit@4 and MRR against gold documents, and
with fact@1, fact@4 and fact MRR (a chunk of a gold document that contains a gold number).
Writes ``eval/ablation.csv`` (one row per configuration) and ``eval/ablation_detail.csv``
(first gold rank per question and configuration, to find questions a technique fixed or broke).

Usage::

    uv run python -m eval.ablation
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from core.logging_setup import get_logger, setup_logging
from core.models import RetrievedChunk
from core.settings import load_settings
from eval.questions import Question, load_questions, retrieval_questions
from eval.retrieval_metrics import (
    chunk_has_fact,
    first_gold_rank,
    first_true_rank,
    hit_at_k,
    mean,
    reciprocal_rank,
)
from factory import Container
from rag.retrievers import Retriever
from rag.router import RuleRouter

log = get_logger(__name__)
OUT_DIR = Path(__file__).parent
BLEND_WEIGHT = 0.7  # cross-encoder share in the blended rows (best of eval/rerank_sweep.csv)
DEPTH = 5  # chunks retrieved per question; hit@4 is what the LLM sees (top_k = 4)


@dataclass(frozen=True)
class Config:
    """One ablation row: how a question is turned into ranked chunks."""

    name: str
    retrieve: Callable[[str], list[RetrievedChunk]]


def filtered(retriever: Retriever, router: RuleRouter) -> Callable[[str], list[RetrievedChunk]]:
    """Retrieve with the router's filter, repeating without it when nothing is found."""

    def run(question: str) -> list[RetrievedChunk]:
        where = router.route(question).where
        chunks = retriever.retrieve(question, DEPTH, where)
        return chunks or retriever.retrieve(question, DEPTH, None)

    return run


def build_configs(container: Container) -> list[Config]:
    """Cumulative configurations; the last rows compare rerankers (see PROGRESS)."""
    tiny = "ms-marco-TinyBERT-L-2-v2"
    vector = container.retriever("vector")
    hybrid = container.retriever("hybrid")
    # The plain rows use the cross-encoder alone (weight 1.0), whatever config.yaml says.
    pure_cfg = container.settings.retrieval.model_copy(
        update={"rerank_weight": 1.0, "rerank_with_title": True}
    )
    rerank = container.retriever("hybrid_rerank", tiny, cfg=pure_cfg)
    # Same cross-encoder, but its order is blended with the hybrid order (see rag.rerank).
    blend_cfg = container.settings.retrieval.model_copy(
        update={"rerank_weight": BLEND_WEIGHT, "rerank_with_title": True}
    )
    blend = container.retriever("hybrid_rerank", tiny, cfg=blend_cfg)
    router = RuleRouter()
    return [
        Config("vector", lambda q: vector.retrieve(q, DEPTH)),
        Config("+hybrid", lambda q: hybrid.retrieve(q, DEPTH)),
        Config("+rerank (TinyBERT)", lambda q: rerank.retrieve(q, DEPTH)),
        Config("+filters (hybrid)", filtered(hybrid, router)),
        Config("+filters+rerank (TinyBERT)", filtered(rerank, router)),
        Config("+rerank blend (TinyBERT)", lambda q: blend.retrieve(q, DEPTH)),
        Config("+filters+rerank blend (TinyBERT)", filtered(blend, router)),
        Config(
            "rerank MiniLM-L-12",
            filtered(
                container.retriever("hybrid_rerank", "ms-marco-MiniLM-L-12-v2", cfg=pure_cfg),
                router,
            ),
        ),
        Config(
            "rerank MultiBERT-L-12",
            filtered(
                container.retriever("hybrid_rerank", "ms-marco-MultiBERT-L-12", cfg=pure_cfg),
                router,
            ),
        ),
    ]


def score_retrieval(
    name: str, retrieve: Callable[[str], list[RetrievedChunk]], questions: Sequence[Question]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Score one retrieval function: document-level hit@1/hit@4/MRR and chunk-level fact metrics.

    The *fact* metrics count a hit only when a retrieved chunk of a gold document contains a gold
    number; they are computed over the questions that have one.
    """
    hit1: list[float] = []
    hit4: list[float] = []
    rr: list[float] = []
    fact1: list[float] = []
    fact4: list[float] = []
    fact_rr: list[float] = []
    detail: list[dict[str, object]] = []
    for q in questions:
        chunks = retrieve(q.question)
        ids = [c.doc_id for c in chunks]
        gold = set(q.gold_source_ids)
        hit1.append(float(hit_at_k(ids, gold, 1)))
        hit4.append(float(hit_at_k(ids, gold, 4)))
        rr.append(reciprocal_rank(ids, gold))
        fact_rank: int | None = None
        if q.gold_numbers:
            numbers = list(q.gold_numbers.values())
            fact_rank = first_true_rank(
                [chunk_has_fact(c.doc_id, c.text, gold, numbers) for c in chunks]
            )
            fact1.append(float(fact_rank is not None and fact_rank <= 1))
            fact4.append(float(fact_rank is not None and fact_rank <= 4))
            fact_rr.append(0.0 if fact_rank is None else 1.0 / fact_rank)
        detail.append(
            {
                "config": name,
                "id": q.id,
                "gold_rank": first_gold_rank(ids, gold),
                "fact_rank": fact_rank,
            }
        )
    row: dict[str, object] = {
        "config": name,
        "questions": len(questions),
        "hit@1": round(mean(hit1), 3),
        "hit@4": round(mean(hit4), 3),
        "mrr": round(mean(rr), 3),
        "fact_questions": len(fact1),
        "fact@1": round(mean(fact1), 3),
        "fact@4": round(mean(fact4), 3),
        "fact_mrr": round(mean(fact_rr), 3),
    }
    log.info(
        "%-32s hit@1 %.2f MRR %.3f | fact@1 %.2f fact@4 %.2f",
        name,
        mean(hit1),
        mean(rr),
        mean(fact1),
        mean(fact4),
    )
    return row, detail


def evaluate(
    configs: Sequence[Config], questions: Sequence[Question]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Score every configuration; returns (summary rows, per-question detail rows)."""
    summary: list[dict[str, object]] = []
    detail: list[dict[str, object]] = []
    for cfg in configs:
        row, rows = score_retrieval(cfg.name, cfg.retrieve, questions)
        summary.append(row)
        detail.extend(rows)
    return summary, detail


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write dict rows as UTF-8 CSV with LF line endings."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """CLI entry point."""
    settings = load_settings()
    setup_logging(settings.resolve(settings.paths.logs_dir))
    questions = retrieval_questions(load_questions())
    summary, detail = evaluate(build_configs(Container(settings)), questions)
    write_csv(OUT_DIR / "ablation.csv", summary)
    write_csv(OUT_DIR / "ablation_detail.csv", detail)
    log.info("Wrote eval/ablation.csv and eval/ablation_detail.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
