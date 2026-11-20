import json
from pathlib import Path

import pytest

from core.exceptions import ConfigError
from core.settings import Settings
from data_pipeline.writer import write_documents
from eval.generate_gold import fill_question, generate, is_rate, number_from_stat, read_text_gold
from eval.numbers import contains_number, extract_numbers
from eval.questions import Question, TextGold, load_questions
from tests.unit.test_writer_pii_runall import mk


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The rate is 32.4% in Germany", [32.4]),
        ("32,4 %", [32.4]),
        ("EUR 100 000 per depositor", [100000.0]),
        ("EUR 100,000 and €100.000", [100000.0, 100000.0]),
        ("0,2 % and 0,3 %", [0.2, 0.3]),
        ("1,234.56 or 1.234,56", [1234.56, 1234.56]),
        ("in 2026-09-16 the rate is 2.5%", [2026.0, 9.0, 16.0, 2.5]),
        ("no numbers here", []),
        ("ids like abc12 or x3y are ignored", []),
    ],
)
def test_extract_numbers(text: str, expected: list[float]) -> None:
    assert extract_numbers(text) == pytest.approx(expected)


def test_contains_number_with_tolerance_and_percent_form() -> None:
    assert contains_number("churn is 32.4%", 32.4, 0.05, percent=True)
    assert contains_number("churn is 32,4 %", 32.4, 0.05, percent=True)
    assert contains_number("churn is 0.324", 32.4, 0.05, percent=True)
    assert not contains_number("churn is 32%", 32.4, 0.05, percent=True)
    assert contains_number("EUR 100,000", 100000.0, 0.5)
    assert not contains_number("EUR 50 000", 100000.0, 0.5)


def test_is_rate() -> None:
    assert is_rate("geography.Germany.churn_rate") and is_rate("leakage.share_churners_complained")
    assert is_rate("leakage.churn_rate_not_complained")
    assert not is_rate("card_type.GOLD.mean_points") and not is_rate("geography.Germany.customers")


def test_number_from_stat() -> None:
    rate = number_from_stat("geography.Germany.churn_rate", 0.32443)
    assert (rate.value, rate.display, rate.percent) == (32.4, "32.4", True)
    assert number_from_stat("complain.1.customers", 2044.0).display == "2,044"
    assert number_from_stat("card_type.GOLD.mean_points", 610.249).display == "610.2"


def corpus(tmp_path: Path) -> Path:
    docs = [
        mk("law", text="# Law\n\nA fixed coverage level of EUR 100 000 applies. Fee: 0,2 %."),
        mk(
            "ecb",
            text="# Rate\n\nThe ECB deposit facility rate is 2.50% (in effect since 2026-09-16).",
        ),
    ]
    write_documents(docs, tmp_path)
    return tmp_path


def test_text_gold_reads_the_document(tmp_path: Path) -> None:
    root = corpus(tmp_path)
    assert read_text_gold(root, "law", r"coverage level of EUR (\d{1,3}(?: \d{3})*)") == "100 000"
    with pytest.raises(ConfigError, match="not found in law"):
        read_text_gold(root, "law", r"nothing like this (\d+)")
    with pytest.raises(ConfigError, match="document not found"):
        read_text_gold(root, "missing", r"(\d+)")


def test_fill_question_from_stats_and_text(tmp_path: Path) -> None:
    root = corpus(tmp_path)
    q = Question(
        id="X", category="R", behavior="answer", question="?", gold_answer="draft",
        gold_answer_template="Churn {germany}%, EUR {cover}, rate {rate}% since {since}.",
        gold_numbers_from={"germany": "geography.Germany.churn_rate"},
        gold_numbers_from_text={
            "cover": TextGold(doc_id="law", pattern=r"coverage level of EUR (\d{1,3}(?: \d{3})*)"),
            "rate": TextGold(doc_id="ecb", pattern=r"rate is (\d+(?:\.\d+)?)%"),
            "since": TextGold(doc_id="ecb", pattern=r"since (\d{4}-\d{2}-\d{2})"),
        },
    )  # fmt: skip
    out = fill_question(q, {"geography.Germany.churn_rate": 0.3244}, root)
    assert out.gold_answer == "Churn 32.4%, EUR 100 000, rate 2.50% since 2026-09-16."
    assert out.gold_numbers["cover"].value == 100000.0 and out.gold_facts == {"since": "2026-09-16"}
    assert out.gold_numbers["rate"].value == 2.5


def test_missing_stats_key_fails_clearly(tmp_path: Path) -> None:
    q = Question(id="X", category="D", behavior="answer", question="?", gold_answer="a",
                 gold_numbers_from={"a": "no.such.key"})  # fmt: skip
    with pytest.raises(ConfigError, match="stats key not found"):
        fill_question(q, {}, tmp_path)


def test_generated_questions_in_the_repo_are_consistent(settings: Settings, tmp_path: Path) -> None:
    qs = {q.id: q for q in load_questions()}
    assert qs["D1"].gold_numbers["germany"].value == pytest.approx(32.4)
    assert qs["R1"].gold_numbers["coverage_eur"].value == 100000.0
    assert qs["R2"].gold_numbers["debit_cap_pct"].value == 0.2
    assert "2.50" in qs["R3"].gold_answer or "2.5" in qs["R3"].gold_answer
    copy = tmp_path / "q.json"
    copy.write_text(json.dumps([q.model_dump() for q in qs.values()]), encoding="utf-8")
    again = generate(settings, copy)
    assert [q.gold_answer for q in again] == [q.gold_answer for q in qs.values()]
