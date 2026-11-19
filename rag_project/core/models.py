"""Domain models shared by the pipeline and the RAG code (Section 5.3 metadata)."""

from __future__ import annotations

import datetime as dt
from typing import Literal

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
