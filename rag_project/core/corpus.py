"""Read the processed corpus (Markdown files with YAML front matter) back into documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from core.exceptions import ConfigError
from core.models import Document, DocumentMetadata


def read_front_matter(path: Path) -> dict[str, Any]:
    """Parse the YAML front matter of a processed document."""
    _, front, _ = path.read_text(encoding="utf-8").split("---\n", 2)
    data: dict[str, Any] = yaml.safe_load(front)
    return data


def read_document(path: Path) -> Document:
    """Load one processed Markdown file as a :class:`Document`."""
    _, front, body = path.read_text(encoding="utf-8").split("---\n", 2)
    meta = DocumentMetadata(**yaml.safe_load(front))
    return Document(text=body.strip() + "\n", metadata=meta)


def load_documents(processed_dir: Path) -> list[Document]:
    """Load every ``<source>/<doc_id>.md`` under ``processed_dir`` (sorted by doc_id).

    Raises:
        ConfigError: If the directory has no documents (the pipeline was not run).
    """
    docs = [read_document(p) for p in sorted(processed_dir.glob("*/*.md"))]
    if not docs:
        raise ConfigError(
            f"No documents in {processed_dir}. Run: uv run python -m data_pipeline.run_all"
        )
    return sorted(docs, key=lambda d: d.doc_id)
