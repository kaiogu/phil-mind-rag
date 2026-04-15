"""Grounding / adjudication agent node function."""

from __future__ import annotations

from openai import OpenAI

from phil_mind_rag.agents._llm import generate_structured
from phil_mind_rag.agents.prompts import grounding_prompt
from phil_mind_rag.agents.schema import StanceMemo, SynthesisReport
from phil_mind_rag.agents.state import AgentState


def run_grounding(
    state: AgentState, client: OpenAI, model: str
) -> dict[str, SynthesisReport]:
    memos: list[StanceMemo] = [
        m
        for m in (
            state["materialist_memo"],
            state["idealist_memo"],
            state["dualist_memo"],
        )
        if m is not None
    ]
    system, user = grounding_prompt(state["question"], state["chunks"], memos)
    report = generate_structured(client, model, system, user, SynthesisReport)
    return {"report": report}
