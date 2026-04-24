"""Vector store abstraction — decouple from any specific backend.

ChromaDB is the default implementation; swap by implementing BaseVectorStore.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from phil_mind_rag.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single result from a similarity search."""

    text: str
    score: float
    metadata: dict[str, str]


class BaseVectorStore(ABC):
    """Interface every vector store must implement."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Persist chunks with their embeddings."""

    @abstractmethod
    def query(self, embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        """Return the top-k most similar chunks."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored chunks."""


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-backed vector store."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        import chromadb

        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' (%d documents)",
            collection_name,
            self._collection.count(),
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._ensure_embedding_dimension(len(embeddings[0]) if embeddings else None)
        ids = [_stable_vector_id(chunk, index) for index, chunk in enumerate(chunks)]
        self._collection.upsert(
            ids=ids,
            documents=[c.text for c in chunks],
            embeddings=embeddings,  # type: ignore
            metadatas=[c.metadata for c in chunks],
        )
        logger.info("Upserted %d chunks to ChromaDB", len(chunks))

    def query(self, embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        self._ensure_embedding_dimension(len(embedding))
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

        retrieval_results: list[RetrievalResult] = []
        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        for text, dist, meta in zip(documents, distances, metadatas, strict=True):
            retrieval_results.append(
                RetrievalResult(
                    text=text,
                    score=1.0 - dist,  # cosine distance → similarity
                    metadata=meta,
                )
            )

        return retrieval_results

    def count(self) -> int:
        return self._collection.count()

    def _ensure_embedding_dimension(self, actual_dimension: int | None) -> None:
        """Fail fast when the active embedding model changes vector dimension."""
        if actual_dimension is None:
            return

        metadata = self._collection.metadata or {}
        stored_dimension = metadata.get("embedding_dimension")
        if isinstance(stored_dimension, (int, float)):
            expected_dimension = int(stored_dimension)
        else:
            expected_dimension = self._existing_embedding_dimension()

        if expected_dimension is None:
            updated_metadata = {
                key: value
                for key, value in metadata.items()
                if not str(key).startswith("hnsw:")
            }
            updated_metadata["embedding_dimension"] = actual_dimension
            self._collection.modify(metadata=updated_metadata)
            return

        if expected_dimension != actual_dimension:
            raise ValueError(
                "Embedding dimension mismatch for Chroma collection. "
                f"Collection expects {expected_dimension}, but the active "
                f"embedding backend produced {actual_dimension}. "
                "Delete the existing Chroma data directory or use a different "
                "collection name before re-ingesting documents."
            )

    def _existing_embedding_dimension(self) -> int | None:
        if self._collection.count() == 0:
            return None
        sample = self._collection.get(limit=1, include=["embeddings"])
        embeddings = sample.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return len(embeddings[0])


def _stable_vector_id(chunk: Chunk, index: int) -> str:
    """Return a deterministic vector-store ID for a chunk."""
    if source_chunk_id := chunk.metadata.get("source_chunk_id"):
        return source_chunk_id
    digest = sha256(f"{index}\0{chunk.text}".encode()).hexdigest()[:16]
    return f"chunk_{index}_{digest}"
