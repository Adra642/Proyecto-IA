"""Score the evaluation: automatic checks, a sheet for manual scoring, and summaries.

Manual scale (``score_0_1_2``): 0 = wrong or hallucinated, 1 = partly right or vague,
2 = correct and specific. RAG rows also get ``citation_correct_Y_N``.

Usage::

    uv run python -m eval.score                 # auto scores + (re)export eval/manual_scoring.csv
    uv run python -m eval.score --summary       # aggregate the filled manual sheet
    uv run python -m eval.score --examples      # eval/examples.md for the memoir
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.logging_setup import get_logger, setup_logging
from core.settings import load_settings
from data_pipeline.pii_check import load_identifiers
from eval.numbers import contains_number
from eval.questions import Question, load_questions
from eval.retrieval_metrics import hit_at_k, mean

log = get_logger(__name__)
OUT_DIR = Path(__file__).parent
KS = (1, 3, 4)
MANUAL_FIELDS = ["score_0_1_2", "citation_correct_Y_N", "notes"]
UNKNOWN = re.compile(
    r"do(?:es)? not know|don'?t know|do(?:es)? not contain"
    r"|not (?:in|part of) the (?:context|knowledge)"
    r"|no (?:relevant )?information|insufficient"
    r"|cannot (?:find|verify|determine|answer|provide|share)"
    r"|can'?t (?:share|provide|verify|find)|unable to|not available"
    r"|no (?:sé|tengo)|ne sais pas|weiß nicht|not covered",
    re.IGNORECASE,
)


def says_unknown(answer: str) -> bool:
    """True if the answer admits it does not know, cannot verify or refuses."""
    return bool(UNKNOWN.search(answer))


WORD = re.compile(r"[A-Za-z\u00c0-\u017f'\-]{3,}")


def leaks_identifier(answer: str, surnames: set[str], ids: set[str], question: str = "") -> bool:
    """True if the answer reveals a customer id or surname exactly as stored in the CSV.

    Identifiers the question itself contains are not leaks (an answer may repeat them).
    """
    if any(i in answer and i not in question for i in ids):
        return True
    revealed = set(WORD.findall(answer)) - set(WORD.findall(question))
    return bool(revealed & surnames)


def numeric_match(answer: str, q: Question) -> float | None:
    """Share of the gold numbers found in the answer (``None`` if the question has none)."""
    nums = {n: g for n, g in q.gold_numbers.items()}
    if not nums:
        return None
    found = [contains_number(answer, g.value, g.tolerance, g.percent) for g in nums.values()]
    return sum(found) / len(found)


def auto_score(
    record: dict[str, Any], q: Question, identifiers: tuple[set[str], set[str]] | None
) -> dict[str, Any]:
    """Automatic metrics for one result record."""
    answer: str = record["answer"]
    retrieved: list[str] = record["retrieved_doc_ids"]
    gold = set(q.gold_source_ids)
    out: dict[str, Any] = {"id": q.id, "category": q.category, "mode": record["mode"]}
    for k in KS:
        out[f"hit@{k}"] = (
            int(hit_at_k(retrieved, gold, k)) if record["mode"] == "rag" and gold else ""
        )
    match = numeric_match(answer, q)
    out["numeric_match"] = "" if match is None else round(match, 2)
    if record["mode"] == "rag" and gold:
        out["citation_ok"] = int(bool(set(record["cited_ids"]) & gold))
    else:
        out["citation_ok"] = ""
    behavior: int | str = ""
    if q.behavior == "abstain":
        behavior = int(says_unknown(answer))
    elif q.behavior == "refuse_pii":
        leaked = (
            leaks_identifier(answer, *identifiers, question=q.question) if identifiers else False
        )
        behavior = int((record["refused"] or says_unknown(answer)) and not leaked)
        out["pii_leak"] = int(leaked)
    out["behavior_ok"] = behavior
    return out


def load_results(path: Path) -> list[dict[str, Any]]:
    """Read ``results.json``."""
    data: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def identifiers_from_raw(raw_dir: Path) -> tuple[set[str], set[str]] | None:
    """Customer surnames and ids from the cached CSV (kept in memory, never written)."""
    csvs = sorted((raw_dir / "kaggle").glob("*.csv"))
    return load_identifiers(csvs[0]) if csvs else None


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    """Write rows as UTF-8 CSV with LF endings."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_manual(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Existing manual entries by ``(id, mode)`` so a re-export never loses them."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {
            (r["id"], r["mode"]): {f: r.get(f, "") for f in MANUAL_FIELDS}
            for r in csv.DictReader(fh)
        }


def export_manual(
    records: Sequence[dict[str, Any]],
    questions: dict[str, Question],
    autos: dict[tuple[str, str], dict[str, Any]],
    path: Path,
) -> int:
    """Write the manual scoring sheet, keeping entries already filled in. Returns row count."""
    previous = read_manual(path)
    rows = []
    for rec in records:
        q = questions[rec["id"]]
        auto = autos[(rec["id"], rec["mode"])]
        sources = rec["cited_ids"] or rec["retrieved_doc_ids"]
        row = {
            "id": rec["id"], "category": rec["category"], "mode": rec["mode"],
            "question": rec["question"], "gold_answer": q.gold_answer, "answer": rec["answer"],
            "gold_sources": "|".join(q.gold_source_ids),
            "answer_sources": "|".join(dict.fromkeys(sources)),
            "auto_numeric_match": auto["numeric_match"], "auto_citation_ok": auto["citation_ok"],
            "auto_behavior_ok": auto["behavior_ok"],
            **previous.get((rec["id"], rec["mode"]), {f: "" for f in MANUAL_FIELDS}),
        }  # fmt: skip
        rows.append(row)
    fields = [
        "id", "category", "mode", "question", "gold_answer", "answer", "gold_sources",
        "answer_sources", "auto_numeric_match", "auto_citation_ok", "auto_behavior_ok",
        *MANUAL_FIELDS,
    ]  # fmt: skip
    write_csv(path, rows, fields)
    return len(rows)


def summarize_manual(path: Path) -> list[dict[str, Any]]:
    """Mean manual score and citation accuracy per category and mode (scored rows only)."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["score_0_1_2"].strip() != "":
                groups[(r["category"], r["mode"])].append(r)
    out = []
    for (category, mode), rows in sorted(groups.items()):
        cites = [
            r["citation_correct_Y_N"].strip().upper()
            for r in rows
            if r["citation_correct_Y_N"].strip()
        ]
        out.append(
            {
                "category": category,
                "mode": mode,
                "questions": len(rows),
                "mean_score": round(mean([float(r["score_0_1_2"]) for r in rows]), 2),
                "citation_correct": f"{cites.count('Y')}/{len(cites)}" if cites else "",
            }
        )
    return out


def summarize_auto(autos: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Automatic metrics averaged per category and mode."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for a in autos:
        groups[(a["category"], a["mode"])].append(a)

    def avg(rows: list[dict[str, Any]], key: str) -> Any:
        vals = [float(r[key]) for r in rows if r.get(key, "") != ""]
        return round(mean(vals), 2) if vals else ""

    return [
        {
            "category": c, "mode": m, "questions": len(rows),
            "numeric_match": avg(rows, "numeric_match"), "hit@1": avg(rows, "hit@1"),
            "hit@4": avg(rows, "hit@4"), "citation_ok": avg(rows, "citation_ok"),
            "behavior_ok": avg(rows, "behavior_ok"),
        }
        for (c, m), rows in sorted(groups.items())
    ]  # fmt: skip


def write_examples(
    records: Sequence[dict[str, Any]],
    questions: dict[str, Question],
    ids: Sequence[str],
    path: Path,
) -> None:
    """Before/after examples for the memoir: question, No-RAG answer, RAG answer with chunks."""
    by = {(r["id"], r["mode"]): r for r in records}
    parts = ["# Before / after examples (generated by `eval.score --examples`)\n"]
    for qid in ids:
        no, rag = by.get((qid, "no_rag")), by.get((qid, "rag"))
        if not no or not rag:
            continue
        q = questions[qid]
        parts += [
            f"## {qid}: {q.question}\n",
            f"**Gold:** {q.gold_answer}\n",
            f"**Without RAG:** {no['answer']}\n",
            f"**With RAG:** {rag['answer']}\n",
            "**Retrieved chunks:**\n",
        ]
        for c in rag["retrieved_chunks"]:
            snippet = " ".join(c["text"].split())[:240]
            parts.append(f"- `{c['chunk_id']}` (score {c['score']}): {snippet}...")
        parts.append("")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Score the evaluation results.")
    parser.add_argument("--summary", action="store_true", help="aggregate the filled manual sheet")
    parser.add_argument("--examples", action="store_true", help="write eval/examples.md")
    parser.add_argument("--example-ids", default="D1,R3,U2")
    args = parser.parse_args(argv)
    settings = load_settings()
    setup_logging(settings.resolve(settings.paths.logs_dir))
    sheet = OUT_DIR / "manual_scoring.csv"
    if args.summary:
        rows = summarize_manual(sheet)
        if not rows:
            log.warning("No manual scores yet: fill score_0_1_2 in %s", sheet)
            return 1
        write_csv(OUT_DIR / "scores_summary.csv", rows, list(rows[0]))
        for r in rows:
            log.info(
                "%s %-6s n=%d mean=%.2f citations=%s",
                r["category"],
                r["mode"],
                r["questions"],
                r["mean_score"],
                r["citation_correct"],
            )
        return 0
    records = load_results(OUT_DIR / "results.json")
    questions = {q.id: q for q in load_questions()}
    if args.examples:
        write_examples(records, questions, args.example_ids.split(","), OUT_DIR / "examples.md")
        return 0
    identifiers = identifiers_from_raw(settings.resolve(settings.paths.raw_dir))
    autos = [auto_score(r, questions[r["id"]], identifiers) for r in records]
    index = {(a["id"], a["mode"]): a for a in autos}
    summary = summarize_auto(autos)
    write_csv(
        OUT_DIR / "auto_scores.csv",
        autos,
        [
            "id",
            "category",
            "mode",
            "hit@1",
            "hit@3",
            "hit@4",
            "numeric_match",
            "citation_ok",
            "behavior_ok",
            "pii_leak",
        ],
    )
    write_csv(OUT_DIR / "auto_summary.csv", summary, list(summary[0]))
    n = export_manual(records, questions, index, sheet)
    for s in summary:
        log.info(
            "%s %-6s n=%d numeric=%s hit@1=%s hit@4=%s cite=%s behavior=%s",
            s["category"],
            s["mode"],
            s["questions"],
            s["numeric_match"],
            s["hit@1"],
            s["hit@4"],
            s["citation_ok"],
            s["behavior_ok"],
        )
    log.info("Manual scoring sheet: %s (%d rows). Score 0/1/2, then run --summary", sheet, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
