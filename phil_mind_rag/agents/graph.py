"""LangGraph orchestration — parallel fan-out to stance agents, then grounding."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from phil_mind_rag.agents.grounding import run_grounding
from phil_mind_rag.agents.schema import StanceMemo, SynthesisReport
from phil_mind_rag.agents.stance import run_dualist, run_idealist, run_materialist
from phil_mind_rag.agents.state import AgentState
from phil_mind_rag.config import Settings
from phil_mind_rag.pipeline import RAGPipeline
from phil_mind_rag.retrieval.store import RetrievalResult


@dataclass
class AnalysisResult:
    """Full output of a multi-agent analysis run."""

    report: SynthesisReport
    chunks: list[RetrievalResult]
    materialist_memo: StanceMemo
    idealist_memo: StanceMemo
    dualist_memo: StanceMemo

logger = logging.getLogger(__name__)


def build_graph(pipeline: RAGPipeline, settings: Settings):  # type: ignore[return]
    """Compile and return the LangGraph StateGraph."""
    api_key = settings.openai_api_key.get_secret_value()
    client = OpenAI(api_key=api_key)
    model = settings.openai_chat_model

    def retrieve(state: AgentState) -> dict[str, list[RetrievalResult]]:
        logger.info("Retrieving context for: %s", state["question"][:80])
        chunks = pipeline.retrieve(state["question"])
        return {"chunks": chunks}

    def run_stances(state: AgentState) -> dict[str, object]:
        logger.info("Running stance agents in parallel")
        fns = [run_materialist, run_idealist, run_dualist]
        result: dict[str, object] = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(fn, state, client, model): fn for fn in fns}
            for future in as_completed(futures):
                result.update(future.result())
        return result

    def grounding(state: AgentState) -> dict[str, SynthesisReport]:
        logger.info("Running grounding agent")
        return run_grounding(state, client, model)

    graph = StateGraph(AgentState)  # ty: ignore[invalid-argument-type]
    graph.add_node("retrieve", retrieve)
    graph.add_node("run_stances", run_stances)
    graph.add_node("grounding", grounding)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "run_stances")
    graph.add_edge("run_stances", "grounding")
    graph.add_edge("grounding", END)
    return graph.compile()


def run_analysis(
    question: str,
    pipeline: RAGPipeline,
    settings: Settings,
) -> AnalysisResult:
    """Run the full multi-agent analysis and return an AnalysisResult."""
    compiled = build_graph(pipeline, settings)
    initial: AgentState = {
        "question": question,
        "chunks": [],
        "materialist_memo": None,
        "idealist_memo": None,
        "dualist_memo": None,
        "report": None,
    }
    final: AgentState = compiled.invoke(initial)

    report = final["report"]
    mat = final["materialist_memo"]
    ide = final["idealist_memo"]
    dua = final["dualist_memo"]

    if report is None or mat is None or ide is None or dua is None:
        raise RuntimeError("One or more agents returned no output")

    return AnalysisResult(
        report=report,
        chunks=final["chunks"],
        materialist_memo=mat,
        idealist_memo=ide,
        dualist_memo=dua,
    )
