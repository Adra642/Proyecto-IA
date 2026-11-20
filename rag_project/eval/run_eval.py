"""Run every evaluation question in No-RAG and RAG mode and save the answers.

Writes ``eval/results.json`` (answers, retrieved chunks, latency) and ``eval/results.csv``.
Answers are cached on disk (key: provider and model, prompts, index, retrieval settings).

Usage::

    uv run python -m eval.run_eval [--modes no_rag,rag] [--ids D1,R3] [--refresh]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from core.logging_setup import get_logger, setup_logging
from core.models import RagAnswer
from core.settings import Settings, load_settings
from eval.answer_cache import CACHE_DIR, AnswerCache
from eval.questions import Question, load_questions
from factory import Container
from rag.prompts import PROMPT_VERSION

log = get_logger(__name__)
OUT_DIR = Path(__file__).parent
MODES = ("no_rag", "rag")
CSV_FIELDS = [
    "id", "category", "mode", "model", "question", "answer", "retrieved_doc_ids",
    "cited_ids", "refused", "latency_s", "from_cache", "error",
]  # fmt: skip


class Answerer(Protocol):
    """What the runner needs from the chain (lets tests use a fake)."""

    def answer_no_rag(self, question: str) -> RagAnswer:
        """Answer with the plain LLM."""
        ...

    def answer_rag(self, question: str) -> RagAnswer:
        """Answer with retrieval."""
        ...


@dataclass
class ResultRow:
    """One answer to one question in one mode."""

    question: Question
    mode: str
    answer: RagAnswer | None
    from_cache: bool
    error: str = ""


def key_parts(
    settings: Settings, model_id: str, corpus_hash: str, q: Question, mode: str
) -> dict[str, Any]:
    """Everything that determines an answer (the cache key)."""
    return {
        "mode": mode,
        "question": q.question,
        "model": model_id,
        "prompts": PROMPT_VERSION,
        "top_k": settings.retrieval.top_k if mode == "rag" else None,
        "chunking": settings.chunking.model_dump() if mode == "rag" else None,
        "embedding": settings.embeddings.model if mode == "rag" else None,
        "corpus": corpus_hash if mode == "rag" else None,
    }


def run_questions(
    chain: Answerer,
    questions: Sequence[Question],
    modes: Sequence[str],
    cache: AnswerCache,
    parts_for: Any,
) -> list[ResultRow]:
    """Answer each question in each mode; failures are recorded, not raised."""
    rows: list[ResultRow] = []
    for q in questions:
        for mode in modes:
            parts = parts_for(q, mode)
            cached = cache.get(parts)
            if cached is not None:
                rows.append(ResultRow(q, mode, cached, True))
                log.info("%s %-6s cached", q.id, mode)
                continue
            try:
                answer = (
                    chain.answer_no_rag(q.question)
                    if mode == "no_rag"
                    else chain.answer_rag(q.question)
                )
            except Exception as exc:
                log.error("%s %s failed: %s", q.id, mode, exc)
                rows.append(ResultRow(q, mode, None, False, error=str(exc)))
                continue
            cache.put(parts, answer)
            rows.append(ResultRow(q, mode, answer, False))
            log.info("%s %-6s %.1fs", q.id, mode, answer.latency_s)
    return rows


def to_record(row: ResultRow) -> dict[str, Any]:
    """Flat JSON-friendly record of a result row (keeps the retrieved chunk texts)."""
    a = row.answer
    return {
        "id": row.question.id,
        "category": row.question.category,
        "mode": row.mode,
        "model": a.model if a else "",
        "question": row.question.question,
        "answer": a.answer if a else "",
        "retrieved_doc_ids": [c.doc_id for c in a.chunks] if a else [],
        "cited_ids": a.cited_ids if a else [],
        "refused": a.refused if a else False,
        "latency_s": round(a.latency_s, 2) if a else None,
        "from_cache": row.from_cache,
        "error": row.error,
        "retrieved_chunks": [
            {
                "chunk_id": c.chunk_id,
                "score": round(c.score, 4),
                "header_path": c.metadata.get("header_path", ""),
                "text": c.text,
            }
            for c in (a.chunks if a else [])
        ],
    }


def write_results(rows: Sequence[ResultRow], out_dir: Path = OUT_DIR) -> list[dict[str, Any]]:
    """Write ``results.json`` and ``results.csv``."""
    records = [to_record(r) for r in rows]
    (out_dir / "results.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for rec in records:
            flat = {
                **rec,
                "retrieved_doc_ids": "|".join(rec["retrieved_doc_ids"]),
                "cited_ids": "|".join(rec["cited_ids"]),
            }
            writer.writerow(flat)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns 1 if any answer failed."""
    parser = argparse.ArgumentParser(description="Run the No-RAG vs RAG evaluation.")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--ids", help="comma-separated question ids (default: all)")
    parser.add_argument("--refresh", action="store_true", help="ignore the answer cache")
    args = parser.parse_args(argv)
    settings = load_settings()
    setup_logging(settings.resolve(settings.paths.logs_dir))
    container = Container(settings)
    questions = load_questions()
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        questions = [q for q in questions if q.id in wanted]
    corpus_hash = container.corpus_hash()
    model_id = container.model_id()
    log.info("Evaluating %d questions with %s", len(questions), model_id)
    rows = run_questions(
        container.chain(),
        questions,
        args.modes.split(","),
        AnswerCache(CACHE_DIR, refresh=args.refresh),
        lambda q, mode: key_parts(settings, model_id, corpus_hash, q, mode),
    )
    write_results(rows)
    failed = [r for r in rows if r.error]
    log.info("Done: %d answers, %d failed. See eval/results.csv", len(rows), len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
