"""Evaluation question set (``eval/questions.json``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

QUESTIONS_PATH = Path(__file__).parent / "questions.json"


class TextGold(BaseModel):
    """A gold value read from a processed document: ``pattern`` has one capture group."""

    doc_id: str
    pattern: str


class GoldNumber(BaseModel):
    """A gold number as generated from ``stats.json`` or a document (never typed by hand)."""

    value: float
    tolerance: float
    display: str
    percent: bool = False


class Question(BaseModel):
    """One evaluation question with its gold sources.

    ``gold_numbers_from`` maps a name to a ``stats.json`` key and ``gold_numbers_from_text``
    to a regex over a processed document; ``generate_gold.py`` fills ``gold_numbers``,
    ``gold_facts`` and ``gold_answer`` (from ``gold_answer_template``).
    """

    id: str
    category: Literal["D", "L", "R", "U"]
    behavior: Literal["answer", "abstain", "refuse_pii"]
    question: str
    gold_answer: str
    gold_source_ids: list[str] = Field(default_factory=list)
    gold_answer_template: str | None = None
    gold_numbers_from: dict[str, str] = Field(default_factory=dict)
    gold_numbers_from_text: dict[str, TextGold] = Field(default_factory=dict)
    gold_numbers: dict[str, GoldNumber] = Field(default_factory=dict)
    gold_facts: dict[str, str] = Field(default_factory=dict)


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    """Load and validate the question file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Question(**item) for item in data]


def retrieval_questions(questions: list[Question]) -> list[Question]:
    """Questions that have gold sources (the behavioural U questions have none)."""
    return [q for q in questions if q.gold_source_ids]
