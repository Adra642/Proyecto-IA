from datetime import date
from pathlib import Path

import pytest

from core.models import Document, DocumentMetadata
from core.settings import ChunkingSettings
from rag.chunking import build_chunker
from rag.index import Indexer, fingerprint
from rag.vectorstore import VectorStoreRepository, open_persistent_client
from tests.fakes import FakeEmbedder

CFG = ChunkingSettings(chunk_size=200, chunk_overlap=30)


def mk(
    doc_id: str, text: str, source_type: str = "regulation", geography: str = "EU", title: str = ""
) -> Document:
    meta = DocumentMetadata(
        doc_id=doc_id, title=title or doc_id, source="s", source_type=source_type, url="https://x",
        language="en", geography=geography, topic="t", columns=["Balance", "Exited"],
        retrieved_at=date(2026, 11, 19), license="l",
    )  # fmt: skip
    return Document(text=text, metadata=meta)


GERMANY = mk(
    "card_de", "# Germany\n\nThe churn rate in Germany is 32.4 percent.", "dataset_stats", "DE"
)
FRANCE = mk(
    "card_fr", "# France\n\nThe churn rate in France is 16.2 percent.", "dataset_stats", "FR"
)
CLAUSES = " ".join(f"Coverage level clause {i} of EUR 100 000." for i in range(25))
LAW = mk("dgs", "# Deposit guarantee\n\n### Article 6\n\n" + CLAUSES)


@pytest.fixture
def repo(tmp_path: Path) -> VectorStoreRepository:
    return VectorStoreRepository(
        open_persistent_client(tmp_path / "chroma"), "test_col", FakeEmbedder()
    )


def indexer(repo: VectorStoreRepository) -> Indexer:
    return Indexer(repo, build_chunker(CFG), CFG, "fake-model")


def test_query_finds_the_right_document(repo: VectorStoreRepository) -> None:
    indexer(repo).sync([GERMANY, FRANCE, LAW])
    hits = repo.query("churn rate in Germany", k=3)
    assert hits[0].doc_id == "card_de" and hits[0].score > hits[1].score
    assert hits[0].metadata["geography"] == "DE" and hits[0].metadata["columns"] == "Balance,Exited"


def test_metadata_filter(repo: VectorStoreRepository) -> None:
    indexer(repo).sync([GERMANY, FRANCE, LAW])
    only_fr = repo.query("churn rate", k=3, where={"geography": "FR"})
    assert [h.doc_id for h in only_fr] == ["card_fr"]
    either = repo.query("churn rate", k=5, where={"geography": {"$in": ["DE", "FR"]}})
    assert {h.doc_id for h in either} == {"card_de", "card_fr"}


def test_atomic_docs_are_one_chunk_and_laws_are_split(repo: VectorStoreRepository) -> None:
    report = indexer(repo).sync([GERMANY, LAW])
    assert repo.count() == report.chunks_total
    de = repo.query("churn Germany", k=10, where={"doc_id": "card_de"})
    law = repo.query("coverage level", k=50, where={"doc_id": "dgs"})
    assert len(de) == 1 and len(law) > 3
    assert all(
        h.metadata["header_path"] == "Deposit guarantee > Article 6"
        for h in law
        if h.metadata["chunk_index"] > 0
    )


def test_sync_is_incremental_and_idempotent(repo: VectorStoreRepository) -> None:
    ix = indexer(repo)
    first = ix.sync([GERMANY, FRANCE, LAW])
    assert (first.added, first.unchanged) == (3, 0)
    count = repo.count()
    again = ix.sync([GERMANY, FRANCE, LAW])
    assert (again.added, again.updated, again.unchanged, again.removed) == (0, 0, 3, 0)
    assert repo.count() == count


def test_changed_document_is_replaced_and_removed_one_deleted(repo: VectorStoreRepository) -> None:
    ix = indexer(repo)
    ix.sync([GERMANY, FRANCE, LAW])
    changed = mk(
        "card_de",
        "# Germany\n\nThe churn rate in Germany is 33.0 percent now.",
        "dataset_stats",
        "DE",
    )
    report = ix.sync([changed, LAW])
    assert (report.updated, report.removed, report.unchanged) == (1, 1, 1)
    assert "33.0" in repo.query("churn Germany", k=1)[0].text
    assert repo.query("France", k=5, where={"doc_id": "card_fr"}) == []


def test_rebuild_starts_from_scratch(repo: VectorStoreRepository) -> None:
    ix = indexer(repo)
    ix.sync([GERMANY, FRANCE])
    report = ix.rebuild([GERMANY])
    assert (report.added, report.chunks_total) == (1, 1)


def test_fingerprint_depends_on_text_metadata_config_and_model() -> None:
    base = fingerprint(GERMANY, CFG, "m")
    assert fingerprint(GERMANY, CFG, "m") == base
    assert fingerprint(GERMANY, CFG, "other") != base
    assert fingerprint(GERMANY, ChunkingSettings(chunk_size=300, chunk_overlap=30), "m") != base
    assert fingerprint(mk("card_de", GERMANY.text + "x", "dataset_stats", "DE"), CFG, "m") != base
    assert fingerprint(mk("card_de", GERMANY.text, "dataset_stats", "FR"), CFG, "m") != base


def test_query_on_empty_store_returns_nothing(repo: VectorStoreRepository) -> None:
    assert repo.query("anything") == [] and repo.doc_hashes() == {}


def test_invalid_collection_name_has_clear_error(tmp_path: Path) -> None:
    from core.exceptions import IndexingError

    with pytest.raises(IndexingError, match="cannot open collection"):
        VectorStoreRepository(open_persistent_client(tmp_path / "c"), "x", FakeEmbedder())
