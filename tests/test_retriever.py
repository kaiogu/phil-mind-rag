"""Tests for VectorRetriever."""

from unittest.mock import MagicMock

from phil_mind_rag.retrieval.retriever import VectorRetriever
from phil_mind_rag.retrieval.store import RetrievalResult


def _make_retriever(
    results: list[RetrievalResult] | None = None,
    embedding: list[float] | None = None,
) -> tuple[VectorRetriever, MagicMock, MagicMock]:
    store = MagicMock()
    store.query.return_value = results or []
    embed_fn = MagicMock(return_value=embedding or [0.1, 0.2, 0.3])
    return VectorRetriever(store=store, embed_fn=embed_fn), store, embed_fn


class TestVectorRetriever:
    def test_embed_fn_called_with_query(self) -> None:
        retriever, _, embed_fn = _make_retriever()
        retriever.retrieve("what is consciousness?", top_k=3)
        embed_fn.assert_called_once_with("what is consciousness?")

    def test_store_queried_with_embedding_and_top_k(self) -> None:
        vec = [0.5, 0.6, 0.7]
        retriever, store, _ = _make_retriever(embedding=vec)
        retriever.retrieve("query", top_k=4)
        call = store.query.call_args
        assert call.args[0] == vec
        assert call.kwargs.get("top_k") == 4

    def test_returns_results_from_store(self) -> None:
        expected = [RetrievalResult(text="passage", score=0.88, metadata={})]
        retriever, _, _ = _make_retriever(results=expected)
        assert retriever.retrieve("q") == expected

    def test_returns_empty_list_when_store_empty(self) -> None:
        retriever, _, _ = _make_retriever(results=[])
        assert retriever.retrieve("q") == []

    def test_default_top_k_is_five(self) -> None:
        retriever, store, _ = _make_retriever()
        retriever.retrieve("q")
        call = store.query.call_args
        top_k = call.args[1] if len(call.args) > 1 else call.kwargs.get("top_k")
        assert top_k == 5
