from pathlib import Path

import pytest

from core.corpus import load_documents, read_document
from core.exceptions import ConfigError
from data_pipeline.writer import write_documents
from tests.unit.test_index import GERMANY, LAW


def test_roundtrip_through_the_writer(tmp_path: Path) -> None:
    write_documents([GERMANY, LAW], tmp_path)
    docs = load_documents(tmp_path)
    assert [d.doc_id for d in docs] == ["card_de", "dgs"]
    assert docs[0].metadata == GERMANY.metadata
    assert docs[0].text.strip() == GERMANY.text.strip()
    assert read_document(next(tmp_path.glob("*/card_de.md"))).metadata.geography == "DE"


def test_empty_corpus_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="run_all"):
        load_documents(tmp_path)
