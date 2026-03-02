"""Pipeline orchestration — the ONLY place that wires components together.

To swap frameworks, only this file needs to change.
"""

from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI

from phil_mind_rag.config import Settings
from phil_mind_rag.generation.llm import OpenAILLM
from phil_mind_rag.generation.prompts import RAGPrompt
from phil_mind_rag.ingestion.chunker import SectionAwareChunker
from phil_mind_rag.ingestion.parser import UnstructuredPDFParser
from phil_mind_rag.retrieval.retriever import VectorRetriever
from phil_mind_rag.retrieval.store import ChromaVectorStore, RetrievalResult
from phil_mind_rag.security import sanitise_query, validate_document

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for Philosophy of Mind papers."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        api_key = settings.openai_api_key.get_secret_value()

        # --- Components (all behind interfaces) -------------------------
        self._parser = UnstructuredPDFParser()
        self._chunker = SectionAwareChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._store = ChromaVectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )

        # OpenAI client for embeddings
        self._openai = OpenAI(api_key=api_key)
        self._embedding_model = settings.openai_embedding_model

        self._retriever = VectorRetriever(
            store=self._store,
            embed_fn=self._embed_text,
        )
        self._llm = OpenAILLM(api_key=api_key, model=settings.openai_chat_model)
        self._prompt = RAGPrompt()

    # --- Embedding helper -----------------------------------------------

    def _embed_text(self, text: str) -> list[float]:
        response = self._openai.embeddings.create(
            input=text,
            model=self._embedding_model,
        )
        return response.data[0].embedding

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._openai.embeddings.create(
            input=texts,
            model=self._embedding_model,
        )
        return [item.embedding for item in response.data]

    # --- Public API -----------------------------------------------------

    def ingest(self, pdf_path: Path) -> int:
        """Parse, chunk, embed, and store a PDF. Returns chunk count."""
        validate_document(
            pdf_path,
            allowed_exts=self._settings.allowed_extensions,
            max_mb=self._settings.max_document_size_mb,
        )

        document = self._parser.parse(pdf_path)
        chunks = self._chunker.chunk(document)

        if not chunks:
            logger.warning("No chunks produced for %s", pdf_path.name)
            return 0

        embeddings = self._embed_batch([c.text for c in chunks])
        self._store.add_chunks(chunks, embeddings)

        return len(chunks)

    def query(self, question: str, top_k: int = 5) -> str:
        """Answer a question using retrieved context."""
        clean_query = sanitise_query(question)
        contexts: list[RetrievalResult] = self._retriever.retrieve(
            clean_query, top_k=top_k
        )

        if not contexts:
            return "No relevant context found for your question."

        prompt = self._prompt.build(clean_query, contexts)
        return self._llm.generate(prompt)

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve contexts without generating — useful for eval & debug."""
        clean_query = sanitise_query(question)
        return self._retriever.retrieve(clean_query, top_k=top_k)

    @property
    def document_count(self) -> int:
        return self._store.count()
