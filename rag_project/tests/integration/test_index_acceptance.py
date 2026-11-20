"""Stage 2 acceptance: the real index answers retrieval queries correctly."""

import pytest

from core.settings import load_settings
from factory import Container

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def repo():  # type: ignore[no-untyped-def]
    store = Container(load_settings()).vector_store()
    if store.count() == 0:
        pytest.skip("index is empty: run `uv run python -m rag.index --rebuild`")
    return store


def test_germany_card_in_top_3(repo) -> None:  # type: ignore[no-untyped-def]
    ids = [h.doc_id for h in repo.query("churn rate in Germany", k=3)]
    assert "kaggle_segment_geography_germany" in ids


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is the deposit facility rate of the ECB?", "ecb_ecb_deposit_facility_rate"),
        ("¿Cómo presento una reclamación al Banco de España?", "bde_como_realizar_una_reclamacion"),
        ("Wie wird der SCHUFA-Score berechnet?", "credit_registers_schufa_scoring_und_daten"),
    ],
)
def test_multilingual_retrieval(repo, query: str, expected: str) -> None:  # type: ignore[no-untyped-def]
    assert expected in [h.doc_id for h in repo.query(query, k=3)]


def test_metadata_filter_by_geography(repo) -> None:  # type: ignore[no-untyped-def]
    hits = repo.query("churn rate", k=5, where={"geography": "DE"})
    assert hits and all(h.metadata["geography"] == "DE" for h in hits)
