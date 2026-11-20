"""Repository over ChromaDB: the rest of the code never imports chromadb directly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings

from core.exceptions import IndexingError
from core.models import Chunk, RetrievedChunk
from rag.embeddings import Embedder

DOC_HASH_KEY = "doc_hash"


def open_persistent_client(path: Path) -> ClientAPI:
    """Open (or create) an on-disk Chroma database, with telemetry disabled."""
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
    )


def chunk_metadata(chunk: Chunk, doc_hash: str) -> dict[str, str | int | float | bool]:
    """Flatten a chunk's metadata to the scalar types Chroma accepts."""
    m = chunk.metadata
    return {
        "doc_id": m.doc_id,
        "title": m.title,
        "source": m.source,
        "source_type": m.source_type,
        "url": m.url,
        "language": m.language,
        "geography": m.geography,
        "topic": m.topic,
        "columns": ",".join(m.columns),
        "date": m.date.isoformat() if m.date else "",
        "retrieved_at": m.retrieved_at.isoformat(),
        "license": m.license,
        "format": m.format,
        "chunk_index": chunk.index,
        "header_path": chunk.header_path,
        DOC_HASH_KEY: doc_hash,
    }


class VectorStoreRepository:
    """``add / query / delete / rebuild`` over one Chroma collection (cosine distance)."""

    def __init__(
        self,
        client: ClientAPI,
        collection: str,
        embedder: Embedder,
        batch_size: int = 32,
    ) -> None:
        """Args: client: Chroma client; collection: name; embedder: passage/query encoder."""
        self._client = client
        self._name = collection
        self._embedder = embedder
        self._batch = batch_size
        self._col = self._open()

    def _open(self) -> Any:
        try:
            return self._client.get_or_create_collection(
                self._name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:  # e.g. an invalid collection name in config.yaml
            raise IndexingError(f"cannot open collection '{self._name}': {exc}") from exc

    def count(self) -> int:
        """Number of chunks stored."""
        return int(self._col.count())

    def add(self, chunks: list[Chunk], doc_hash: str = "") -> None:
        """Embed and store ``chunks`` (upsert by ``chunk_id``).

        Raises:
            IndexingError: If the embedding or the store fails.
        """
        try:
            for start in range(0, len(chunks), self._batch):
                part = chunks[start : start + self._batch]
                self._col.upsert(
                    ids=[c.chunk_id for c in part],
                    documents=[c.text for c in part],
                    embeddings=self._embedder.embed_passages([c.embedding_text for c in part]),
                    metadatas=[chunk_metadata(c, doc_hash) for c in part],
                )
        except Exception as exc:
            raise IndexingError(f"could not add chunks to '{self._name}': {exc}") from exc

    def query(
        self, text: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return the ``k`` most similar chunks (score = cosine similarity), best first."""
        if self.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=[self._embedder.embed_query(text)],
            n_results=min(k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for cid, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            strict=True,
        ):
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=str(meta["doc_id"]),
                    text=doc,
                    score=1.0 - float(dist),
                    metadata=dict(meta),
                )
            )
        return out

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk of a document."""
        self._col.delete(where={"doc_id": doc_id})

    def doc_hashes(self) -> dict[str, str]:
        """Map ``doc_id`` to the fingerprint stored with its chunks."""
        if self.count() == 0:
            return {}
        rows = self._col.get(include=["metadatas"])["metadatas"]
        return {str(m["doc_id"]): str(m.get(DOC_HASH_KEY, "")) for m in rows}

    def rebuild(self) -> None:
        """Drop the collection and start empty."""
        self._client.delete_collection(self._name)
        self._col = self._open()
