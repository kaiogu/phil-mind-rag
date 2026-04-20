"""Plain Python orchestration baseline for framework comparison.

This module intentionally mirrors the LangGraph flow without depending on a
graph framework. It gives KGU-93 a concrete comparison target for evaluating
inspectability, boilerplate, and control-flow explicitness.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import OpenAI

from phil_mind_rag.agents.graph import AnalysisResult
from phil_mind_rag.agents.grounding import run_grounding
from phil_mind_rag.agents.stance import run_dualist, run_idealist, run_materialist

if TYPE_CHECKING:
    from collections.abc import Callable

    from phil_mind_rag.agents.schema import StanceMemo
    from phil_mind_rag.agents.state import AgentState
    from phil_mind_rag.config import Settings
    from phil_mind_rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestrationProfile:
    """Static comparison metadata for an orchestration implementation."""

    name: str
    framework: str
    control_flow: str
    parallelism: str
    state_model: str
    inspectability_notes: str
    tradeoffs: list[str]


LANGGRAPH_PROFILE = OrchestrationProfile(
    name="langgraph-v1",
    framework="LangGraph",
    control_flow="Compiled state graph: retrieve -> run_stances -> grounding.",
    parallelism="ThreadPool fan-out inside the stance node.",
    state_model="Typed AgentState passed through graph nodes.",
    inspectability_notes=(
        "Best when explicit graph topology, future conditional routing, retries, "
        "or observability hooks matter."
    ),
    tradeoffs=[
        "Adds framework concepts and compile-time state validation.",
        "Parallel stance execution is still implemented manually inside one node.",
    ],
)

PLAIN_PROFILE = OrchestrationProfile(
    name="plain-python-v1",
    framework="Plain Python",
    control_flow="Direct function calls: retrieve, parallel stances, grounding.",
    parallelism="ThreadPool fan-out over stance node functions.",
    state_model="Typed AgentState dict assembled and updated explicitly.",
    inspectability_notes=(
        "Lowest abstraction overhead and easiest to step through in a debugger; "
        "less useful once routing or durable execution gets more complex."
    ),
    tradeoffs=[
        "No graph visualization or built-in orchestration lifecycle.",
        "Retries, tracing, and conditional transitions must be built by hand.",
    ],
)


def orchestration_profiles() -> list[OrchestrationProfile]:
    """Return the currently implemented orchestration comparison targets."""
    return [LANGGRAPH_PROFILE, PLAIN_PROFILE]


def run_plain_analysis(
    question: str,
    pipeline: RAGPipeline,
    settings: Settings,
) -> AnalysisResult:
    """Run the multi-agent analysis using plain Python orchestration."""
    api_key = settings.openai_api_key.get_secret_value()
    client = OpenAI(api_key=api_key)
    model = settings.openai_chat_model

    logger.info("Retrieving context for: %s", question[:80])
    chunks = pipeline.retrieve(question)
    state: AgentState = {
        "question": question,
        "chunks": chunks,
        "materialist_memo": None,
        "idealist_memo": None,
        "dualist_memo": None,
        "report": None,
    }

    logger.info("Running stance agents in parallel with plain orchestration")
    stance_fns: list[Callable[[AgentState, OpenAI, str], dict[str, StanceMemo]]] = [
        run_materialist,
        run_idealist,
        run_dualist,
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fn, state, client, model) for fn in stance_fns]
        for future in as_completed(futures):
            result = future.result()
            if "materialist_memo" in result:
                state["materialist_memo"] = result["materialist_memo"]
            if "idealist_memo" in result:
                state["idealist_memo"] = result["idealist_memo"]
            if "dualist_memo" in result:
                state["dualist_memo"] = result["dualist_memo"]

    logger.info("Running grounding agent with plain orchestration")
    state["report"] = run_grounding(state, client, model)["report"]

    report = state["report"]
    mat = state["materialist_memo"]
    ide = state["idealist_memo"]
    dua = state["dualist_memo"]
    if report is None or mat is None or ide is None or dua is None:
        raise RuntimeError("One or more agents returned no output")

    baseline_answer = pipeline.answer_from_contexts(question, chunks)
    return AnalysisResult(
        baseline_answer=baseline_answer,
        report=report,
        chunks=chunks,
        materialist_memo=mat,
        idealist_memo=ide,
        dualist_memo=dua,
    )
