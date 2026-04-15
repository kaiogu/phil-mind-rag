"""Stance agent node functions — materialist, idealist, dualist."""

from __future__ import annotations

from typing import TYPE_CHECKING

from phil_mind_rag.agents._llm import generate_structured
from phil_mind_rag.agents.prompts import stance_prompt
from phil_mind_rag.agents.schema import StanceMemo

if TYPE_CHECKING:
    from openai import OpenAI

    from phil_mind_rag.agents.state import AgentState

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
    system, user = stance_prompt(
        "materialist", _MATERIALIST, state["question"], state["chunks"]
    )
    memo = generate_structured(client, model, system, user, StanceMemo)
    return {"materialist_memo": memo}


def run_idealist(
    state: AgentState, client: OpenAI, model: str
) -> dict[str, StanceMemo]:
    system, user = stance_prompt(
        "idealist", _IDEALIST, state["question"], state["chunks"]
    )
    memo = generate_structured(client, model, system, user, StanceMemo)
    return {"idealist_memo": memo}


def run_dualist(state: AgentState, client: OpenAI, model: str) -> dict[str, StanceMemo]:
    system, user = stance_prompt(
        "dualist", _DUALIST, state["question"], state["chunks"]
    )
    memo = generate_structured(client, model, system, user, StanceMemo)
    return {"dualist_memo": memo}
