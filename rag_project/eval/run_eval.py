"""Run every evaluation question in No-RAG and RAG mode and save the answers.

Writes ``eval/results.json`` (answers, retrieved chunks, latency) and ``eval/results.csv``.
Every run calls the LLM again (no answer cache).

Usage::

    uv run python -m eval.run_eval [--modes no_rag,rag,rag_advanced] [--ids D1,R3]
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
from core.settings import load_settings
from eval.questions import Question, load_questions
from factory import Container

log = get_logger(__name__)
OUT_DIR = Path(__file__).parent
MODES = ("no_rag", "rag", "rag_advanced")
CSV_FIELDS = [
    "id", "category", "mode", "model", "question", "answer", "retrieved_doc_ids",
    "cited_ids", "refused", "route", "latency_s", "error",
]  # fmt: skip


class Answerer(Protocol):
    """What the runner needs from the chain (lets tests use a fake)."""

    def answer_no_rag(self, question: str) -> RagAnswer:
        """Answer with the plain LLM."""
        ...

    def answer_rag(self, question: str) -> RagAnswer:
        """Answer with retrieval."""
        ...

    def answer_advanced(self, question: str) -> RagAnswer:
        """Answer with routing, hybrid search and optional reranking."""
        ...


@dataclass
class ResultRow:
    """One answer to one question in one mode."""

    question: Question
    mode: str
    answer: RagAnswer | None
    error: str = ""


def run_questions(
    chain: Answerer,
    questions: Sequence[Question],
    modes: Sequence[str],
) -> list[ResultRow]:
    """Answer each question in each mode; failures are recorded, not raised."""
    rows: list[ResultRow] = []
    for q in questions:
        for mode in modes:
            try:
                answer = {
                    "no_rag": chain.answer_no_rag,
                    "rag": chain.answer_rag,
                    "rag_advanced": chain.answer_advanced,
                }[mode](q.question)
            except Exception as exc:
                log.error("%s %s failed: %s", q.id, mode, exc)
                rows.append(ResultRow(q, mode, None, error=str(exc)))
                continue
            rows.append(ResultRow(q, mode, answer))
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
        "route": a.route if a else "",
        "latency_s": round(a.latency_s, 2) if a else None,
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
    args = parser.parse_args(argv)
    settings = load_settings()
    setup_logging(settings.resolve(settings.paths.logs_dir))
    container = Container(settings)
    questions = load_questions()
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        questions = [q for q in questions if q.id in wanted]
    model_id = container.model_id()
    log.info("Evaluating %d questions with %s", len(questions), model_id)
    rows = run_questions(
        container.chain(),
        questions,
        args.modes.split(","),
    )
    write_results(rows)
    failed = [r for r in rows if r.error]
    log.info("Done: %d answers, %d failed. See eval/results.csv", len(rows), len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
