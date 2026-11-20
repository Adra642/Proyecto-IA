import csv
import json
from pathlib import Path
from typing import Any

import pytest

from core.models import RagAnswer, RetrievedChunk
from core.settings import Settings
from eval.answer_cache import AnswerCache, cache_key
from eval.questions import GoldNumber, Question
from eval.run_eval import key_parts, run_questions, to_record, write_results
from eval.score import (
    auto_score,
    export_manual,
    leaks_identifier,
    numeric_match,
    read_manual,
    says_unknown,
    summarize_auto,
    summarize_manual,
    write_examples,
)


def question(qid: str = "D1", category: str = "D", behavior: str = "answer", **kw: Any) -> Question:
    germany = GoldNumber(value=32.4, tolerance=0.05, display="32.4", percent=True)
    base: dict[str, Any] = {
        "id": qid, "category": category, "behavior": behavior, "question": f"Question {qid}?",
        "gold_answer": "gold", "gold_source_ids": ["card_de"], "gold_numbers": {"germany": germany},
    }  # fmt: skip
    return Question(**{**base, **kw})


def chunk(doc_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}#0000", doc_id=doc_id, text="Churn 32.4%", score=0.9,
        metadata={"header_path": "H", "title": "T", "source": "s", "geography": "DE"},
    )  # fmt: skip


class FakeChain:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def answer_no_rag(self, q: str) -> RagAnswer:
        self.calls.append(("no_rag", q))
        return RagAnswer(
            question=q, mode="no_rag", answer="I do not know.", latency_s=1.0, model="m"
        )

    def answer_rag(self, q: str) -> RagAnswer:
        self.calls.append(("rag", q))
        if "boom" in q:
            raise RuntimeError("provider down")
        return RagAnswer(
            question=q, mode="rag", answer="32.4% [card_de]", chunks=[chunk("card_de")],
            cited_ids=["card_de"], latency_s=2.0, model="m",
        )  # fmt: skip


def parts_for(q: Question, mode: str) -> dict[str, Any]:
    return {"q": q.question, "mode": mode}


def test_cache_key_is_stable_and_sensitive(settings: Settings) -> None:
    a = key_parts(settings, "gemini:x", "hash", question(), "rag")
    assert cache_key(a) == cache_key(dict(a))
    assert cache_key(a) != cache_key(key_parts(settings, "groq:y", "hash", question(), "rag"))
    assert cache_key(a) != cache_key(key_parts(settings, "gemini:x", "other", question(), "rag"))
    assert cache_key(a) != cache_key(key_parts(settings, "gemini:x", "hash", question(), "no_rag"))
    no_rag = key_parts(settings, "gemini:x", "hash", question(), "no_rag")
    assert no_rag["corpus"] is None and no_rag["top_k"] is None


def test_answers_are_cached_and_refresh_bypasses(tmp_path: Path) -> None:
    chain = FakeChain()
    qs = [question("D1"), question("D2")]
    first = run_questions(chain, qs, ["no_rag", "rag"], AnswerCache(tmp_path), parts_for)
    assert len(chain.calls) == 4 and not any(r.from_cache for r in first)
    second = run_questions(chain, qs, ["no_rag", "rag"], AnswerCache(tmp_path), parts_for)
    assert len(chain.calls) == 4 and all(r.from_cache for r in second)
    cached = second[1].answer
    assert cached is not None and cached == first[1].answer and cached.latency_s == 2.0
    run_questions(chain, qs, ["rag"], AnswerCache(tmp_path, refresh=True), parts_for)
    assert len(chain.calls) == 6


def test_a_failing_question_is_recorded_not_raised(tmp_path: Path) -> None:
    qs = [question("D1"), question("D2", question="boom?")]
    rows = run_questions(FakeChain(), qs, ["rag"], AnswerCache(tmp_path), parts_for)
    assert [r.error for r in rows] == ["", "provider down"] and rows[1].answer is None
    records = write_results(rows, tmp_path)
    assert records[1]["error"] == "provider down" and records[1]["answer"] == ""
    assert not (tmp_path / "x").exists()


def test_results_files(tmp_path: Path) -> None:
    rows = run_questions(
        FakeChain(), [question()], ["no_rag", "rag"], AnswerCache(tmp_path / "c"), parts_for
    )
    write_results(rows, tmp_path)
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert data[1]["retrieved_chunks"][0]["chunk_id"] == "card_de#0000" and data[1]["mode"] == "rag"
    with (tmp_path / "results.csv").open(encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert csv_rows[1]["retrieved_doc_ids"] == "card_de" and csv_rows[1]["cited_ids"] == "card_de"
    assert to_record(rows[0])["retrieved_doc_ids"] == []


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("I do not know: the context is insufficient.", True),
        ("I don't know.", True),
        ("I can't share personal identifiers.", True),
        ("No tengo información; no sé.", True),
        ("The churn rate in Portugal is 12.0%.", False),
    ],
)
def test_says_unknown(answer: str, expected: bool) -> None:
    assert says_unknown(answer) is expected


def test_leaks_identifier() -> None:
    surnames, ids = {"Hargrave", "Hill"}, {"15634602"}
    assert leaks_identifier("The surname is Hargrave.", surnames, ids)
    assert leaks_identifier("CustomerId 15634602 is", surnames, ids)
    assert not leaks_identifier("I can't share that; the hill is steep.", surnames, ids)
    # repeating what the question already contains is not a leak
    asked = "Surname of CustomerId 15634602?"
    assert not leaks_identifier("I do not know the surname of 15634602.", surnames, ids, asked)
    assert leaks_identifier("It is Hargrave.", surnames, ids, asked)


def test_numeric_match_fraction() -> None:
    q = question(
        gold_numbers={
            "a": GoldNumber(value=32.4, tolerance=0.05, display="32.4", percent=True),
            "b": GoldNumber(value=16.2, tolerance=0.05, display="16.2", percent=True),
        }
    )
    assert numeric_match("Germany 32,4 % and France 16.2%", q) == 1.0
    assert numeric_match("Germany 32.4%", q) == 0.5
    assert numeric_match("no idea", q) == 0.0
    assert numeric_match("x", question(gold_numbers={})) is None


def record(
    mode: str, answer: str, retrieved: list[str], cited: list[str], refused: bool = False
) -> dict[str, Any]:
    return {"id": "D1", "category": "D", "mode": mode, "question": "q", "answer": answer,
            "retrieved_doc_ids": retrieved, "cited_ids": cited, "refused": refused}  # fmt: skip


def test_auto_score_rag_and_no_rag() -> None:
    q = question()
    rag = auto_score(record("rag", "32.4% [card_de]", ["x", "card_de"], ["card_de"]), q, None)
    assert (rag["hit@1"], rag["hit@3"], rag["hit@4"], rag["citation_ok"], rag["numeric_match"]) == (
        0,
        1,
        1,
        1,
        1.0,
    )
    base = auto_score(record("no_rag", "About 30%", [], []), q, None)
    assert base["hit@1"] == "" and base["citation_ok"] == "" and base["numeric_match"] == 0.0


def test_auto_score_behaviours() -> None:
    abstain = question("U1", "U", "abstain", gold_source_ids=[], gold_numbers={})
    assert auto_score(record("rag", "I do not know.", [], []), abstain, None)["behavior_ok"] == 1
    assert auto_score(record("no_rag", "It is 12.3%.", [], []), abstain, None)["behavior_ok"] == 0
    pii = question("U2", "U", "refuse_pii", gold_source_ids=[], gold_numbers={})
    ids = ({"Hargrave"}, {"15634602"})
    ok = auto_score(
        record("rag", "I can't share personal identifiers.", [], [], refused=True), pii, ids
    )
    assert ok["behavior_ok"] == 1 and ok["pii_leak"] == 0
    leak = auto_score(record("no_rag", "The surname is Hargrave.", [], []), pii, ids)
    assert leak["behavior_ok"] == 0 and leak["pii_leak"] == 1


def test_manual_sheet_export_preserves_scores_and_summarizes(tmp_path: Path) -> None:
    q = question()
    recs = [
        record("no_rag", "About 30%", [], []),
        record("rag", "32.4% [card_de]", ["card_de"], ["card_de"]),
    ]
    autos = {(r["id"], r["mode"]): auto_score(r, q, None) for r in recs}
    sheet = tmp_path / "manual.csv"
    assert export_manual(recs, {"D1": q}, autos, sheet) == 2
    with sheet.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["score_0_1_2"], rows[1]["score_0_1_2"], rows[1]["citation_correct_Y_N"] = "0", "2", "Y"
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    export_manual(recs, {"D1": q}, autos, sheet)  # re-export must keep the manual entries
    assert read_manual(sheet)[("D1", "rag")]["score_0_1_2"] == "2"
    summary = {(r["category"], r["mode"]): r for r in summarize_manual(sheet)}
    assert (
        summary[("D", "rag")]["mean_score"] == 2.0
        and summary[("D", "rag")]["citation_correct"] == "1/1"
    )
    assert summary[("D", "no_rag")]["mean_score"] == 0.0


def test_auto_summary_and_examples(tmp_path: Path) -> None:
    q = question()
    recs = [
        {**record("no_rag", "About 30%", [], []), "retrieved_chunks": []},
        {**record("rag", "32.4% [card_de]", ["card_de"], ["card_de"]),
         "retrieved_chunks": [{"chunk_id": "card_de#0000", "score": 0.9, "text": "Churn 32.4%"}]},
    ]  # fmt: skip
    autos = [auto_score(r, q, None) for r in recs]
    by_mode = {(s["category"], s["mode"]): s for s in summarize_auto(autos)}
    assert (
        by_mode[("D", "rag")]["numeric_match"] == 1.0
        and by_mode[("D", "no_rag")]["numeric_match"] == 0.0
    )
    out = tmp_path / "examples.md"
    write_examples(recs, {"D1": q}, ["D1", "MISSING"], out)
    text = out.read_text(encoding="utf-8")
    assert "Without RAG" in text and "card_de#0000" in text and "MISSING" not in text
