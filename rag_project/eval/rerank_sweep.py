"""Sweep of reranking options (retrieval only, no LLM, deterministic).

For each combination of pool size (``candidates``), cross-encoder weight in the blend and
whether the cross-encoder sees the title, retrieval is scored with the document-level and the
chunk-level (fact) metrics of ``eval.ablation``. Writes ``eval/rerank_sweep.csv``.

Usage::

    uv run python -m eval.rerank_sweep
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

from core.logging_setup import get_logger, setup_logging
from core.settings import load_settings
from eval.ablation import DEPTH, filtered, score_retrieval, write_csv
from eval.questions import load_questions, retrieval_questions
from factory import Container
from rag.router import RuleRouter

log = get_logger(__name__)
OUT = Path(__file__).parent / "rerank_sweep.csv"
MODEL = "ms-marco-TinyBERT-L-2-v2"
POOLS = (10, 20, 30)
WEIGHTS = (1.0, 0.7, 0.5, 0.3)
WITH_TITLE = (True, False)


def main() -> int:
    """CLI entry point."""
    settings = load_settings()
    setup_logging(settings.resolve(settings.paths.logs_dir))
    container = Container(settings)
    questions = retrieval_questions(load_questions())
    router = RuleRouter()
    base = settings.retrieval
    rows: list[dict[str, object]] = []
    hybrids = {
        n: container.retriever("hybrid", cfg=base.model_copy(update={"candidates": n}))
        for n in POOLS
    }
    for n in POOLS:  # reference rows: hybrid alone at each pool size
        row, _ = score_retrieval(f"hybrid pool={n}", filtered(hybrids[n], router), questions)
        rows.append({"pool": n, "weight": "", "title": "", **row})
    for pool, weight, title in itertools.product(POOLS, WEIGHTS, WITH_TITLE):
        cfg = base.model_copy(
            update={"candidates": pool, "rerank_weight": weight, "rerank_with_title": title}
        )
        retriever = container.retriever("hybrid_rerank", MODEL, cfg=cfg)
        name = f"rerank pool={pool} w={weight} title={'si' if title else 'no'}"

        row, _ = score_retrieval(name, filtered(retriever, router), questions)
        rows.append({"pool": pool, "weight": weight, "title": "si" if title else "no", **row})
    write_csv(OUT, rows)
    log.info("Wrote %s (%d rows; DEPTH=%d)", OUT.name, len(rows), DEPTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
