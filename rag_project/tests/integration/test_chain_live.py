"""Live end-to-end check of the chain (real Gemini, real index)."""

import pytest

from core.settings import load_settings
from factory import Container

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def chain():  # type: ignore[no-untyped-def]
    container = Container(load_settings())
    if container.vector_store().count() == 0:
        pytest.skip("index is empty: run `uv run python -m rag.index --rebuild`")
    if container.settings.api_key_for(container.settings.llm.provider) is None:
        pytest.skip("no API key for the selected provider")
    return container.chain()


def test_rag_answers_with_the_exact_figure_and_cites(chain) -> None:  # type: ignore[no-untyped-def]
    result = chain.answer_rag(
        "In the Kaggle Bank Customer Churn dataset (10,000 customers in France, Germany and "
        "Spain), what is the churn rate of customers in Germany compared with France and Spain?"
    )
    assert "32.4" in result.answer and result.cited_ids and result.chunks


@pytest.mark.xfail(
    reason="Known weakness of dense-only retrieval: 'Kaggle Bank Customer Churn dataset' matches "
    "every card, so the Germany card ranks 4th-5th. Stage 5 (hybrid + geography filter) fixes it.",
    strict=False,
)
def test_plain_germany_question_known_retrieval_weakness(chain) -> None:  # type: ignore[no-untyped-def]
    result = chain.answer_rag(
        "In the Kaggle Bank Customer Churn dataset, what is the churn rate of customers in Germany?"
    )
    assert "32.4" in result.answer


def test_rag_refuses_personal_data_without_calling_the_llm(chain) -> None:  # type: ignore[no-untyped-def]
    result = chain.answer_rag("What is the surname of the customer with CustomerId 15634602?")
    assert (
        result.refused
        and not result.chunks
        and "15634602" not in result.answer.replace("customer ids", "")
    )


def test_no_rag_baseline_has_no_sources(chain) -> None:  # type: ignore[no-untyped-def]
    result = chain.answer_no_rag("What is the capital of France?")
    assert result.mode == "no_rag" and not result.chunks and "Paris" in result.answer
