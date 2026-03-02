"""Vector store abstraction — decouple from any specific backend.

ChromaDB is the default implementation; swap by implementing BaseVectorStore.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    def query(
        self, embedding: list[float], top_k: int = 5
    ) -> list[RetrievalResult]:
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
        ids = [f"chunk_{i}_{hash(c.text) % 10**8}" for i, c in enumerate(chunks)]
        self._collection.add(
            ids=ids,
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks],  # type: ignore[arg-type]
        )
        logger.info("Added %d chunks to ChromaDB", len(chunks))

    def query(
        self, embedding: list[float], top_k: int = 5
    ) -> list[RetrievalResult]:
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

        retrieval_results: list[RetrievalResult] = []
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for text, dist, meta in zip(documents, distances, metadatas, strict=True):
            retrieval_results.append(
                RetrievalResult(
                    text=text,
                    score=1.0 - dist,  # cosine distance → similarity
                    metadata=meta,  # type: ignore[arg-type]
                )
            )

        return retrieval_results

    def count(self) -> int:
        return self._collection.count()
