"""Build and update the Chroma index from ``data/processed`` (idempotent and incremental).

Usage::

    uv run python -m rag.index [--rebuild] [--query "churn rate in Germany"]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

from core.corpus import load_documents
from core.logging_setup import get_logger, setup_logging
from core.models import Document
from core.settings import ChunkingSettings, load_settings
from factory import Container
from rag.chunking import Chunker
from rag.vectorstore import VectorStoreRepository

log = get_logger(__name__)


@dataclass
class IndexReport:
    """What a sync did."""

    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks_total: int = 0


def fingerprint(doc: Document, cfg: ChunkingSettings, embedding_model: str) -> str:
    """Hash of everything that changes a document's chunks or embeddings."""
    payload = "\x1f".join(
        [
            doc.text,
            doc.metadata.model_dump_json(),
            cfg.model_dump_json(),
            embedding_model,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Indexer:
    """Chunks documents and keeps the repository in sync with them."""

    def __init__(
        self,
        repo: VectorStoreRepository,
        chunker: Chunker,
        chunking: ChunkingSettings,
        embedding_model: str,
    ) -> None:
        """Args: repo: store; chunker: strategy; chunking, embedding_model: fingerprint inputs."""
        self._repo = repo
        self._chunker = chunker
        self._chunking = chunking
        self._model = embedding_model

    def sync(self, docs: Sequence[Document]) -> IndexReport:
        """Add new or changed documents, skip unchanged ones, delete removed ones."""
        report = IndexReport()
        existing = self._repo.doc_hashes()
        current = {d.doc_id for d in docs}
        for gone in sorted(set(existing) - current):
            self._repo.delete_document(gone)
            report.removed += 1
            log.info("Removed %s from the index", gone)
        for doc in docs:
            digest = fingerprint(doc, self._chunking, self._model)
            if existing.get(doc.doc_id) == digest:
                report.unchanged += 1
                continue
            if doc.doc_id in existing:
                self._repo.delete_document(doc.doc_id)
                report.updated += 1
            else:
                report.added += 1
            self._repo.add(self._chunker.chunk(doc), digest)
        report.chunks_total = self._repo.count()
        return report

    def rebuild(self, docs: Sequence[Document]) -> IndexReport:
        """Drop everything and index all documents again."""
        self._repo.rebuild()
        return self.sync(docs)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build or update the vector index.")
    parser.add_argument("--rebuild", action="store_true", help="drop the collection first")
    parser.add_argument("--query", help="after indexing, show the top results for this query")
    parser.add_argument("-k", type=int, default=5, help="results to show with --query")
    args = parser.parse_args(argv)
    settings = load_settings()
    setup_logging(settings.resolve(settings.paths.logs_dir))
    container = Container(settings)
    docs = load_documents(settings.resolve(settings.paths.processed_dir))
    repo = container.vector_store()
    indexer = Indexer(repo, container.chunker(), settings.chunking, settings.embeddings.model)
    start = time.monotonic()
    report = indexer.rebuild(docs) if args.rebuild else indexer.sync(docs)
    log.info(
        "Index ready in %.0fs: %d chunks (added %d docs, updated %d, unchanged %d, removed %d)",
        time.monotonic() - start,
        report.chunks_total,
        report.added,
        report.updated,
        report.unchanged,
        report.removed,
    )
    if args.query:
        for rank, hit in enumerate(repo.query(args.query, args.k), 1):
            log.info("%d. %.3f %s | %s", rank, hit.score, hit.chunk_id, hit.metadata["header_path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
