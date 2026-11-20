"""Evaluation question set (``eval/questions.json``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

QUESTIONS_PATH = Path(__file__).parent / "questions.json"


class Question(BaseModel):
    """One evaluation question with its gold sources.

    ``gold_numbers_from`` maps a name to a ``stats.json`` key; ``generate_gold.py``
    (Stage 3) fills the numbers so none is typed by hand.
    """

    id: str
    category: Literal["D", "L", "R", "U"]
    behavior: Literal["answer", "abstain", "refuse_pii"]
    question: str
    gold_answer: str
    gold_source_ids: list[str] = Field(default_factory=list)
    gold_numbers_from: dict[str, str] = Field(default_factory=dict)


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    """Load and validate the question file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Question(**item) for item in data]


def retrieval_questions(questions: list[Question]) -> list[Question]:
    """Questions that have gold sources (the behavioural U questions have none)."""
    return [q for q in questions if q.gold_source_ids]
