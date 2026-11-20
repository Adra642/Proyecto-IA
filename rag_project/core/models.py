"""Domain models shared by the pipeline and the RAG code (Section 5.3 metadata)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field

Language = Literal["en", "es", "fr", "de"]
Geography = Literal["FR", "DE", "ES", "EU", "global", "US"]
SourceType = Literal[
    "dataset_stats",
    "paper",
    "encyclopedia",
    "regulation",
    "macro_data",
    "regulator_guide",
    "press",
]


class DocumentMetadata(BaseModel):
    """Metadata carried by every ingested document and, later, every chunk."""

    doc_id: str = Field(min_length=1)
    title: str
    source: str
    source_type: SourceType
    url: str
    language: Language
    geography: Geography
    topic: str
    columns: list[str] = Field(default_factory=list)
    date: dt.date | None = None
    retrieved_at: dt.date
    license: str
    format: Literal["csv", "json", "html", "pdf", "generated"] = "generated"


class Document(BaseModel):
    """A cleaned document: Markdown text plus its metadata."""

    text: str
    metadata: DocumentMetadata

    @property
    def doc_id(self) -> str:
        """Shortcut for ``metadata.doc_id``."""
        return self.metadata.doc_id


class Chunk(BaseModel):
    """A retrievable piece of a document, with the document's metadata."""

    chunk_id: str
    index: int
    text: str
    header_path: str = ""  # e.g. "CHAPTER I > Article 1"
    metadata: DocumentMetadata

    @property
    def doc_id(self) -> str:
        """Id of the document this chunk comes from."""
        return self.metadata.doc_id

    @property
    def embedding_text(self) -> str:
        """Text that is embedded: title and header path give each chunk its context."""
        context = " > ".join(p for p in (self.metadata.title, self.header_path) if p)
        return f"{context}\n{self.text}"


class RetrievedChunk(BaseModel):
    """A chunk returned by a retriever, with its score."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class RagAnswer(BaseModel):
    """An answer with the evidence behind it (JSON-serialisable, so it can be cached)."""

    question: str
    mode: Literal["no_rag", "rag"]
    answer: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    cited_ids: list[str] = Field(default_factory=list)
    latency_s: float
    model: str
    refused: bool = False  # answered by a deterministic guard, without calling the LLM
