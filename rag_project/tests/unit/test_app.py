"""App tests: handler logic, source formatting and a smoke test that builds the UI (offline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.handlers import (
    MAX_QUESTION_CHARS,
    MODE_ADVANCED,
    MODE_NO_RAG,
    MODE_RAG,
    MODES,
    MSG_EMPTY,
    MSG_NO_KEY,
    MSG_PROVIDER,
    MSG_RATE_LIMIT,
    MSG_TOO_LONG,
    AppService,
    format_sources,
    format_status,
)
from app.ui import build_demo, load_example_questions
from core.exceptions import MissingCredentialError, ProviderError, RateLimitError
from core.models import RagAnswer, RetrievedChunk


def _chunk(doc_id: str = "doc_a", text: str = "Germany churn is 32.4%") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}#0",
        doc_id=doc_id,
        text=text,
        score=0.812,
        metadata={
            "title": "Card A",
            "url": "https://x.org/a",
            "source": "kaggle_derived",
            "source_type": "dataset_stats",
        },
    )


class FakeChain:
    """Chain double: returns canned answers or raises the queued exceptions."""

    def __init__(self, errors: list[Exception] | None = None) -> None:
        self.errors = list(errors or [])
        self.calls: list[tuple[str, str]] = []
        self.wheres: list[dict[str, Any] | None] = []

    def _next(self, mode: str, question: str) -> RagAnswer:
        self.calls.append((mode, question))
        if self.errors:
            raise self.errors.pop(0)
        chunks = [_chunk()] if mode != "no_rag" else []
        return RagAnswer(
            question=question,
            mode="no_rag" if mode == "no_rag" else ("rag" if mode == "rag" else "rag_advanced"),
            answer=f"{mode} answer",
            chunks=chunks,
            cited_ids=["doc_a"] if chunks else [],
            latency_s=1.25,
            model="fake:model",
        )

    def answer_no_rag(self, question: str) -> RagAnswer:
        return self._next("no_rag", question)

    def answer_rag(self, question: str) -> RagAnswer:
        return self._next("rag", question)

    def answer_advanced(
        self, question: str, k: int | None = None, where: dict[str, Any] | None = None
    ) -> RagAnswer:
        self.wheres.append(where)
        return self._next("advanced", question)


def _service(chain: FakeChain) -> AppService:
    return AppService(chain, sleep=lambda _s: None)


def test_ask_routes_by_mode() -> None:
    chain = FakeChain()
    service = _service(chain)
    assert service.ask("q1", MODE_NO_RAG)[0] == "no_rag answer"
    assert service.ask("q2", MODE_RAG)[0] == "rag answer"
    assert chain.calls == [("no_rag", "q1"), ("rag", "q2")]


def test_advanced_mode_passes_the_filter_to_the_chain() -> None:
    chain = FakeChain()
    where = {"source_type": {"$in": ["regulation"]}}
    text, answer = _service(chain).ask("q", MODE_ADVANCED, where)
    assert text == "advanced answer" and answer is not None and answer.mode == "rag_advanced"
    assert chain.wheres == [where]


def test_empty_and_long_questions_do_not_call_the_chain() -> None:
    chain = FakeChain()
    service = _service(chain)
    assert service.ask("   ", MODE_RAG) == (MSG_EMPTY, None)
    assert service.ask("x" * (MAX_QUESTION_CHARS + 1), MODE_RAG) == (MSG_TOO_LONG, None)
    assert chain.calls == []


def test_rate_limit_is_retried_once_then_recovers() -> None:
    chain = FakeChain([RateLimitError("429")])
    text, answer = _service(chain).ask("q", MODE_RAG)
    assert text == "rag answer" and answer is not None
    assert len(chain.calls) == 2


def test_rate_limit_twice_shows_friendly_message() -> None:
    chain = FakeChain([RateLimitError("429"), RateLimitError("429")])
    assert _service(chain).ask("q", MODE_RAG) == (MSG_RATE_LIMIT, None)


@pytest.mark.parametrize(
    ("error", "message"),
    [(MissingCredentialError("no key"), MSG_NO_KEY), (ProviderError("boom"), MSG_PROVIDER)],
)
def test_provider_errors_become_friendly_messages(error: Exception, message: str) -> None:
    assert _service(FakeChain([error])).ask("q", MODE_RAG) == (message, None)


def test_format_sources_shows_title_link_citation_and_score() -> None:
    md = format_sources([_chunk()], ["doc_a"])
    assert "[Card A](https://x.org/a)" in md
    assert "citada en la respuesta" in md and "0.812" in md and "`doc_a`" in md


def test_format_sources_truncates_and_handles_empty() -> None:
    assert "Sin fuentes" in format_sources([], [])
    assert format_sources([_chunk(text="w " * 2000)], []).count("…") == 1


def test_format_status() -> None:
    assert format_status(None) == ""
    answer = FakeChain().answer_rag("q")
    assert "1.2 s" in format_status(answer) and "fake:model" in format_status(answer)


def test_ui_builds_and_exposes_both_modes() -> None:
    demo = build_demo(_service(FakeChain()), ["example question"])
    assert demo.title == "RAG: churn en banca europea"
    assert MODES == (MODE_NO_RAG, MODE_RAG, MODE_ADVANCED)


def test_example_questions_prefer_the_spanish_wording(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    items = [
        {"id": "D1", "question": "Churn?", "question_es": "¿Churn?"},
        {"id": "D2", "question": "Only English?", "question_es": ""},
    ]
    path.write_text(json.dumps(items), encoding="utf-8")
    assert load_example_questions(path) == ["¿Churn?", "Only English?"]
