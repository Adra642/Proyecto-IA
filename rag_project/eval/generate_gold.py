"""Fill gold numbers and gold answers from ``stats.json`` and the processed documents.

Nothing is typed by hand: dataset numbers come from ``stats.json`` and regulatory / ECB
values are read from the ingested texts with the regexes in ``questions.json`` (a human
can check each regex against the source). Idempotent.

Usage::

    uv run python -m eval.generate_gold
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from core.corpus import read_document
from core.exceptions import ConfigError
from core.logging_setup import get_logger, setup_logging
from core.settings import Settings, load_settings
from eval.numbers import extract_numbers
from eval.questions import QUESTIONS_PATH, GoldNumber, Question, load_questions

log = get_logger(__name__)


def is_rate(key: str) -> bool:
    """True for ``stats.json`` entries that are fractions (churn rates and shares)."""
    name = key.rsplit(".", 1)[-1]
    return "churn_rate" in name or name.startswith("share")


def number_from_stat(key: str, value: float) -> GoldNumber:
    """Gold number from a ``stats.json`` entry (rates are shown as percentages)."""
    if is_rate(key):
        pct = round(value * 100, 1)
        return GoldNumber(value=pct, tolerance=0.05, display=f"{pct:.1f}", percent=True)
    if float(value).is_integer():
        return GoldNumber(value=value, tolerance=0.5, display=f"{int(value):,}")
    return GoldNumber(value=round(value, 1), tolerance=0.05, display=f"{value:.1f}")


def read_text_gold(processed_dir: Path, doc_id: str, pattern: str) -> str:
    """Capture group of ``pattern`` in the body of a processed document.

    Raises:
        ConfigError: If the document or the pattern is not found (the text changed).
    """
    paths = list(processed_dir.glob(f"*/{doc_id}.md"))
    if not paths:
        raise ConfigError(f"gold source document not found: {doc_id}")
    body = read_document(paths[0]).text
    match = re.search(pattern, body)
    if match is None:
        raise ConfigError(f"pattern {pattern!r} not found in {doc_id}")
    return match.group(1)


def fill_question(q: Question, stats: dict[str, float], processed_dir: Path) -> Question:
    """Return ``q`` with ``gold_numbers``, ``gold_facts`` and ``gold_answer`` generated."""
    numbers: dict[str, GoldNumber] = {}
    facts: dict[str, str] = {}
    for name, key in q.gold_numbers_from.items():
        if key not in stats or stats[key] is None:
            raise ConfigError(f"{q.id}: stats key not found: {key}")
        numbers[name] = number_from_stat(key, stats[key])
    for name, spec in q.gold_numbers_from_text.items():
        raw = read_text_gold(processed_dir, spec.doc_id, spec.pattern)
        parsed = extract_numbers(raw)
        if parsed and re.fullmatch(r"[\d., ]+", raw):
            percent_style = "%" in spec.pattern
            numbers[name] = GoldNumber(
                value=parsed[0],
                tolerance=0.05 if parsed[0] < 100 else 0.5,
                display=raw.replace(",", ".") if percent_style else raw,
                percent=percent_style,
            )
        else:
            facts[name] = raw
    answer = q.gold_answer
    if q.gold_answer_template:
        values = {n: g.display for n, g in numbers.items()} | facts
        answer = q.gold_answer_template.format(**values)
    return q.model_copy(
        update={"gold_numbers": numbers, "gold_facts": facts, "gold_answer": answer}
    )


def generate(settings: Settings, path: Path = QUESTIONS_PATH) -> list[Question]:
    """Fill every question and rewrite the question file."""
    processed = settings.resolve(settings.paths.processed_dir)
    stats = json.loads((processed / "stats.json").read_text(encoding="utf-8"))
    questions = [fill_question(q, stats, processed) for q in load_questions(path)]
    payload = [q.model_dump(exclude_defaults=False) for q in questions]
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return questions


def main() -> int:
    """CLI entry point."""
    settings = load_settings()
    setup_logging(settings.resolve(settings.paths.logs_dir))
    for q in generate(settings):
        log.info("%s: %s", q.id, q.gold_answer[:110])
    return 0


if __name__ == "__main__":
    sys.exit(main())
