"""LangGraph state contract for the multi-agent philosophy pipeline."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from phil_mind_rag.agents.schema import StanceMemo, SynthesisReport
from phil_mind_rag.retrieval.store import RetrievalResult


class AgentState(TypedDict):
    question: str
    chunks: list[RetrievalResult]
    materialist_chunks: NotRequired[list[RetrievalResult]]
    idealist_chunks: NotRequired[list[RetrievalResult]]
    dualist_chunks: NotRequired[list[RetrievalResult]]
    materialist_memo: StanceMemo | None
    idealist_memo: StanceMemo | None
    dualist_memo: StanceMemo | None
    report: SynthesisReport | None
