import pytest

from core.settings import load_settings
from rag.embeddings import build_embedder

pytestmark = pytest.mark.integration


def test_embedding_returns_one_vector() -> None:
    vec = build_embedder(load_settings()).embed_query("churn rate in Germany")
    assert len(vec) == 384
