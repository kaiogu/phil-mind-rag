"""Pipeline orchestration — the ONLY place that wires components together.

To swap frameworks, only this file needs to change.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

from phil_mind_rag.config import Settings
from phil_mind_rag.generation.llm import OpenAILLM
from phil_mind_rag.generation.prompts import RAGPrompt
from phil_mind_rag.ingestion.chunker import Chunk, SectionAwareChunker
from phil_mind_rag.ingestion.metadata_extractor import (
    ExtractedMetadata,
    MetadataExtractor,
)
from phil_mind_rag.ingestion.parser import ParsedDocument, UnstructuredPDFParser
from phil_mind_rag.ingestion.registry import DocumentRecord, DocumentRegistry
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
        self._registry = DocumentRegistry(settings.registry_path)

        # OpenAI client for embeddings
        self._openai = OpenAI(api_key=api_key)
        self._embedding_model = settings.openai_embedding_model

        self._retriever = VectorRetriever(
            store=self._store,
            embed_fn=self._embed_text,
        )
        self._llm = OpenAILLM(api_key=api_key, model=settings.openai_chat_model)
        self._prompt = RAGPrompt()
        self._metadata_extractor = MetadataExtractor(llm=self._llm)

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

    # --- Public API (individual steps) ------------------------------------

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Validate and parse a PDF into structured sections."""
        validate_document(
            pdf_path,
            allowed_exts=self._settings.allowed_extensions,
            max_mb=self._settings.max_document_size_mb,
        )
        return self._parser.parse(pdf_path)

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Split a parsed document into retrieval-sized chunks."""
        return self._chunker.chunk(document)

    def embed(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed a list of chunks via OpenAI."""
        return self._embed_batch([c.text for c in chunks])

    def store(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Persist chunks and their embeddings in the vector store."""
        self._store.add_chunks(chunks, embeddings)

    def register(
        self,
        pdf_path: Path,
        chunk_count: int,
        title: str | None = None,
        author: str | None = None,
    ) -> DocumentRecord:
        record = DocumentRecord(
            source=pdf_path.name,
            title=title.strip() if title and title.strip() else pdf_path.stem,
            author=author.strip() if author and author.strip() else "",
            chunk_count=chunk_count,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            chunker=f"{type(self._chunker).__module__}.{type(self._chunker).__qualname__}",
            embedding_model=self._embedding_model,
            ingested_at=datetime.now(UTC).isoformat(),
        )
        self._registry.add(record)
        return record

    def extract_metadata(self, pdf_path: Path) -> ExtractedMetadata:
        """Extract title/author from a PDF's embedded metadata or via LLM."""
        return self._metadata_extractor.extract(pdf_path)

    # --- Convenience (all-in-one) ----------------------------------------

    def ingest(
        self,
        pdf_path: Path,
        title: str | None = None,
        author: str | None = None,
    ) -> int:
        """Parse, chunk, embed, and store a PDF. Returns chunk count."""
        document = self.parse(pdf_path)
        chunks = self.chunk(document)

        if not chunks:
            logger.warning("No chunks produced for %s", pdf_path.name)
            return 0

        embeddings = self.embed(chunks)
        self.store(chunks, embeddings)
        self.register(pdf_path, len(chunks), title, author)
        return len(chunks)

    def query(self, question: str, top_k: int = 5) -> str:
        """Answer a question using retrieved context."""
        answer, _ = self.query_with_sources(question, top_k=top_k)
        return answer

    def query_with_sources(
        self, question: str, top_k: int = 5
    ) -> tuple[str, list[RetrievalResult]]:
        """Answer a question and return the retrieved source chunks."""
        clean_query = sanitise_query(question)
        contexts: list[RetrievalResult] = self._retriever.retrieve(
            clean_query, top_k=top_k
        )

        if not contexts:
            return "No relevant context found for your question.", []

        prompt = self._prompt.build(clean_query, contexts)
        return self._llm.generate(prompt), contexts

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve contexts without generating — useful for eval & debug."""
        clean_query = sanitise_query(question)
        return self._retriever.retrieve(clean_query, top_k=top_k)

    def list_documents(self) -> list[DocumentRecord]:
        return self._registry.list_all()

    @property
    def document_count(self) -> int:
        return self._store.count()
