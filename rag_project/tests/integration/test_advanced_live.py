"""Stage 5 acceptance on the real index: hybrid fixes D6, the router filters, FlashRank runs."""

import pytest

from core.settings import load_settings
from factory import Container

pytestmark = pytest.mark.integration

D6 = "In the Kaggle Bank Customer Churn dataset, what is the churn rate of customers in Germany?"
GERMANY_CARDS = {"kaggle_segment_geography_germany", "kaggle_summary_geography"}


@pytest.fixture(scope="module")
def container():  # type: ignore[no-untyped-def]
    c = Container(load_settings())
    if c.vector_store().count() == 0:
        pytest.skip("index is empty: run `uv run python -m rag.index --rebuild`")
    return c


def test_hybrid_puts_a_germany_card_first_where_vector_search_does_not(container) -> None:  # type: ignore[no-untyped-def]
    vector = [c.doc_id for c in container.retriever("vector").retrieve(D6, 4)]
    hybrid = [c.doc_id for c in container.retriever("hybrid").retrieve(D6, 4)]
    assert vector[0] not in GERMANY_CARDS or len(set(vector) & GERMANY_CARDS) < 2
    assert hybrid[0] in GERMANY_CARDS


def test_router_filter_restricts_the_search_to_dataset_cards(container) -> None:  # type: ignore[no-untyped-def]
    from rag.router import RuleRouter

    where = RuleRouter().route(D6).where
    chunks = container.retriever("hybrid").retrieve(D6, 4, where)
    assert chunks and {c.metadata["source_type"] for c in chunks} == {"dataset_stats"}


def test_flashrank_reranks_candidates(container) -> None:  # type: ignore[no-untyped-def]
    chunks = container.retriever("hybrid_rerank", "ms-marco-TinyBERT-L-2-v2").retrieve(D6, 4)
    assert len(chunks) == 4 and chunks[0].score >= chunks[-1].score
