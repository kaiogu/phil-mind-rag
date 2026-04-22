"""Tests for vector-store persistence helpers."""

from unittest.mock import MagicMock

from phil_mind_rag.ingestion.chunker import Chunk
from phil_mind_rag.retrieval.store import ChromaVectorStore


def _store_with_fake_collection() -> tuple[ChromaVectorStore, MagicMock]:
    store = object.__new__(ChromaVectorStore)
    collection = MagicMock()
    store._collection = collection
    return store, collection


def test_add_chunks_uses_stable_source_chunk_ids_when_available() -> None:
    store, collection = _store_with_fake_collection()
    chunks = [
        Chunk(
            text="Consciousness has subjective character.",
            metadata={"source_chunk_id": "nagel_bat:intro:chunk_0"},
        )
    ]

    store.add_chunks(chunks, embeddings=[[0.1, 0.2]])

    assert collection.upsert.call_args.kwargs["ids"] == ["nagel_bat:intro:chunk_0"]


def test_add_chunks_falls_back_to_deterministic_content_ids() -> None:
    first_store, first_collection = _store_with_fake_collection()
    second_store, second_collection = _store_with_fake_collection()
    chunks = [Chunk(text="Same chunk text.", metadata={})]

    first_store.add_chunks(chunks, embeddings=[[0.1]])
    second_store.add_chunks(chunks, embeddings=[[0.1]])

    first_ids = first_collection.upsert.call_args.kwargs["ids"]
    second_ids = second_collection.upsert.call_args.kwargs["ids"]
    assert first_ids == second_ids
    assert first_ids[0].startswith("chunk_0_")


def test_add_chunks_upserts_to_allow_reingesting_stable_ids() -> None:
    store, collection = _store_with_fake_collection()

    store.add_chunks([Chunk(text="Text.", metadata={})], embeddings=[[0.1]])

    collection.upsert.assert_called_once()
    collection.add.assert_not_called()
