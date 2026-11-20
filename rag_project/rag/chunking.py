"""Chunking strategies (Strategy pattern) and the factory that routes documents to them."""

from __future__ import annotations

from typing import Protocol

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from core.models import Chunk, Document
from core.settings import ChunkingSettings

# Bump when chunk boundaries or header paths change, so the index is rebuilt (see rag.index).
CHUNKER_VERSION = 2
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
# h1 is the document title, which the metadata already carries; repeating it (EU laws have a
# 150-character official title) in every chunk's context would drown the distinguishing part.
PATH_LEVELS = ["h2", "h3", "h4"]
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # paragraph, line, sentence, word, character


class Chunker(Protocol):
    """Splits one document into chunks."""

    def chunk(self, doc: Document) -> list[Chunk]:
        """Return the chunks of ``doc`` (never empty for a non-empty document)."""
        ...


def _make(doc: Document, index: int, text: str, header_path: str = "") -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}#{index:04d}",
        index=index,
        text=text.strip(),
        header_path=header_path,
        metadata=doc.metadata,
    )


class PassthroughChunker:
    """One document = one chunk.

    For short, atomic notes where splitting would separate a number from its label
    (segment cards, data dictionary, ECB and macro notes).
    """

    def chunk(self, doc: Document) -> list[Chunk]:
        """Return the whole document as a single chunk."""
        return [_make(doc, 0, doc.text)]


class MarkdownRecursiveChunker:
    """Split on Markdown headers first (keeping the header path), then by size.

    The header split respects the document structure (articles, sections); the recursive
    split then cuts long sections at paragraph, line, sentence and word boundaries, with an
    overlap so sentences that cross a boundary appear in both chunks.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Args: chunk_size: maximum characters per chunk; chunk_overlap: shared characters."""
        self._headers = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)
        self._sizer = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=SEPARATORS
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` by headers, then by size."""
        chunks: list[Chunk] = []
        for section in self._headers.split_text(doc.text):
            path = " > ".join(
                str(section.metadata[h]) for h in PATH_LEVELS if h in section.metadata
            )
            for piece in self._sizer.split_text(section.page_content):
                if piece.strip():
                    chunks.append(_make(doc, len(chunks), piece, path))
        return chunks or [_make(doc, 0, doc.text)]


class FixedSizeChunker:
    """Size-only splitting that ignores document structure (baseline for comparisons)."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Args: chunk_size: maximum characters per chunk; chunk_overlap: shared characters."""
        self._sizer = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=SEPARATORS
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` by size only."""
        pieces = [p for p in self._sizer.split_text(doc.text) if p.strip()]
        return [_make(doc, i, p) for i, p in enumerate(pieces)] or [_make(doc, 0, doc.text)]


class RoutingChunker:
    """Sends atomic source types to one chunker and everything else to another."""

    def __init__(self, atomic: Chunker, splitter: Chunker, atomic_source_types: set[str]) -> None:
        """Args: atomic/splitter: the two strategies; atomic_source_types: routing rule."""
        self._atomic = atomic
        self._splitter = splitter
        self._atomic_types = atomic_source_types

    def chunk(self, doc: Document) -> list[Chunk]:
        """Chunk ``doc`` with the strategy chosen by its ``source_type``."""
        target = self._atomic if doc.metadata.source_type in self._atomic_types else self._splitter
        return target.chunk(doc)


def build_chunker(cfg: ChunkingSettings) -> Chunker:
    """Build the configured chunker (the composition root calls this)."""
    splitter: Chunker
    if cfg.strategy == "fixed":
        splitter = FixedSizeChunker(cfg.chunk_size, cfg.chunk_overlap)
    else:
        splitter = MarkdownRecursiveChunker(cfg.chunk_size, cfg.chunk_overlap)
    return RoutingChunker(PassthroughChunker(), splitter, set(cfg.atomic_source_types))
