"""Retriever — orchestrates embedding + vector search.

Decoupled from both the embedding provider and the store backend.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from phil_mind_rag.retrieval.store import BaseVectorStore, RetrievalResult

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Interface every retriever must implement."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return the most relevant chunks for a query."""


class VectorRetriever(BaseRetriever):
    """Retriever that embeds the query and searches a vector store."""

    def __init__(
        self,
        store: BaseVectorStore,
        embed_fn: callable,  # type: ignore[type-arg]
    ) -> None:
        self._store = store
        self._embed_fn = embed_fn

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        logger.info("Retrieving top-%d for: %s", top_k, query[:80])
        embedding = self._embed_fn(query)
        return self._store.query(embedding, top_k=top_k)
