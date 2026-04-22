"""Stance agent node functions — materialist, idealist, dualist."""

from __future__ import annotations

from typing import TYPE_CHECKING

from phil_mind_rag.agents._llm import generate_structured
from phil_mind_rag.agents.evidence import StanceName, build_evidence_pack
from phil_mind_rag.agents.prompts import stance_prompt
from phil_mind_rag.agents.schema import StanceMemo

if TYPE_CHECKING:
    from openai import OpenAI

    from phil_mind_rag.agents.state import AgentState
    from phil_mind_rag.retrieval.store import RetrievalResult

_MATERIALIST = (
    "Build the strongest charitable case for physicalism. "
    "Foreground physicalist, functionalist, and neuroscientific arguments. "
    "Engage seriously with anti-physicalist objections before dismissing them."
)

_IDEALIST = (
    "Build the strongest charitable case for idealism. "
    "Foreground consciousness-first, anti-physicalist, and phenomenological arguments. "
    "Engage seriously with physicalist objections before dismissing them."
)

_DUALIST = (
    "Build the strongest charitable case for property dualism and substance dualism. "
    "Engage both the materialist and the idealist critiques charitably. "
    "Acknowledge where each rival position has genuine force."
)


def run_materialist(
    state: AgentState, client: OpenAI, model: str
) -> dict[str, StanceMemo]:
    return _run_stance(
        state=state,
        client=client,
        model=model,
        stance="materialist",
        instruction=_MATERIALIST,
        result_key="materialist_memo",
    )


def run_idealist(
    state: AgentState, client: OpenAI, model: str
) -> dict[str, StanceMemo]:
    return _run_stance(
        state=state,
        client=client,
        model=model,
        stance="idealist",
        instruction=_IDEALIST,
        result_key="idealist_memo",
    )


def run_dualist(state: AgentState, client: OpenAI, model: str) -> dict[str, StanceMemo]:
    return _run_stance(
        state=state,
        client=client,
        model=model,
        stance="dualist",
        instruction=_DUALIST,
        result_key="dualist_memo",
    )


def _run_stance(
    *,
    state: AgentState,
    client: OpenAI,
    model: str,
    stance: StanceName,
    instruction: str,
    result_key: str,
) -> dict[str, StanceMemo]:
    evidence_pack = build_evidence_pack(stance, _chunks_for_stance(state, stance))
    system, user = stance_prompt(
        stance, instruction, state["question"], evidence_pack.chunks
    )
    memo = generate_structured(client, model, system, user, StanceMemo)
    memo = memo.model_copy(update={"evidence_chunk_ids": evidence_pack.chunk_ids})
    return {result_key: memo}


def _chunks_for_stance(
    state: AgentState,
    stance: StanceName,
) -> list[RetrievalResult]:
    if stance == "materialist":
        return state.get("materialist_chunks", state["chunks"])
    if stance == "idealist":
        return state.get("idealist_chunks", state["chunks"])
    return state.get("dualist_chunks", state["chunks"])
