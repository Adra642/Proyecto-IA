from datetime import date

import pytest
from pydantic import ValidationError

from core.models import Document, DocumentMetadata
from core.settings import ChunkingSettings
from rag.chunking import (
    FixedSizeChunker,
    MarkdownRecursiveChunker,
    PassthroughChunker,
    build_chunker,
)


def doc(
    text: str, source_type: str = "regulation", doc_id: str = "law_1", title: str = "Law"
) -> Document:
    meta = DocumentMetadata(
        doc_id=doc_id, title=title, source="eurlex", source_type=source_type, url="u",
        language="en", geography="EU", topic="t", retrieved_at=date(2026, 1, 1), license="l",
    )  # fmt: skip
    return Document(text=text, metadata=meta)


def sentences(prefix: str, n: int) -> str:
    return " ".join(f"{prefix} sentence number {i} talks about deposits." for i in range(n))


LAW = (
    "# Directive\n\n## CHAPTER I\n\n### Article 1 - Scope\n\n" + sentences("Scope", 30)
    + "\n\n### Article 2 - Definitions\n\n" + sentences("Definition", 30)
)  # fmt: skip


def test_passthrough_keeps_document_whole() -> None:
    d = doc("# Card\n\nChurn rate 32.4%.", "dataset_stats")
    (chunk,) = PassthroughChunker().chunk(d)
    assert chunk.text == "# Card\n\nChurn rate 32.4%." and chunk.chunk_id == "law_1#0000"


def test_chunks_respect_size_and_overlap() -> None:
    chunks = MarkdownRecursiveChunker(200, 40).chunk(doc(LAW))
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 for c in chunks)
    same_section = [c for c in chunks if c.header_path.endswith("Article 1 - Scope")]
    assert len(same_section) > 2
    for a, b in zip(same_section, same_section[1:], strict=False):
        assert set(a.text.split()[-3:]) & set(b.text.split()[:12])  # overlapping words


def test_header_path_is_kept_per_chunk() -> None:
    chunks = MarkdownRecursiveChunker(200, 40).chunk(doc(LAW))
    paths = {c.header_path for c in chunks}
    assert "CHAPTER I > Article 1 - Scope" in paths  # the h1 (document title) is not repeated
    assert "CHAPTER I > Article 2 - Definitions" in paths
    assert all("Definition sentence" not in c.text for c in chunks if "Article 1" in c.header_path)


def test_chunk_ids_are_stable_and_ordered() -> None:
    a = MarkdownRecursiveChunker(200, 40).chunk(doc(LAW))
    b = MarkdownRecursiveChunker(200, 40).chunk(doc(LAW))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.index for c in a] == list(range(len(a)))


def test_embedding_text_adds_title_and_header() -> None:
    chunk = MarkdownRecursiveChunker(200, 40).chunk(doc(LAW, title="DGS Directive"))[0]
    # title, then the header path without the h1 (the document title is not repeated)
    assert chunk.embedding_text.startswith("DGS Directive > CHAPTER I > Article 1 - Scope\n")
    assert chunk.embedding_text.endswith(chunk.text)


def test_fixed_chunker_ignores_headers() -> None:
    chunks = FixedSizeChunker(200, 40).chunk(doc(LAW))
    assert len(chunks) > 5 and all(c.header_path == "" for c in chunks)


def test_short_document_gives_one_chunk() -> None:
    assert len(MarkdownRecursiveChunker(600, 90).chunk(doc("# T\n\nShort text."))) == 1


def test_router_never_splits_atomic_types() -> None:
    chunker = build_chunker(ChunkingSettings(chunk_size=100, chunk_overlap=10))
    long_note = "# Note\n\n" + sentences("Rate", 40)
    assert len(chunker.chunk(doc(long_note, "dataset_stats"))) == 1
    assert len(chunker.chunk(doc(long_note, "macro_data"))) == 1
    assert len(chunker.chunk(doc(long_note, "regulation"))) > 3


def test_factory_builds_fixed_strategy() -> None:
    chunker = build_chunker(ChunkingSettings(strategy="fixed", chunk_size=100, chunk_overlap=10))
    assert all(c.header_path == "" for c in chunker.chunk(doc(LAW)))


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        ChunkingSettings(chunk_size=100, chunk_overlap=100)
