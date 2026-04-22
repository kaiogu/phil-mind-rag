"""LangGraph orchestration — parallel fan-out to stance agents, then grounding."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from phil_mind_rag.agents.argument_map import build_argument_map
from phil_mind_rag.agents.claim_extraction import extract_analysis_claims
from phil_mind_rag.agents.claim_verification import verify_claims
from phil_mind_rag.agents.evidence import (
    StanceEvidenceSets,
    StanceName,
    build_stance_evidence_sets,
    build_stance_query,
)
from phil_mind_rag.agents.grounding import run_grounding
from phil_mind_rag.agents.stance import run_dualist, run_idealist, run_materialist
from phil_mind_rag.agents.state import AgentState

if TYPE_CHECKING:
    from phil_mind_rag.agents.schema import (
        ArgumentMap,
        StanceMemo,
        SynthesisReport,
        VerifiedClaim,
    )
    from phil_mind_rag.config import Settings
    from phil_mind_rag.pipeline import RAGPipeline
    from phil_mind_rag.retrieval.store import RetrievalResult


@dataclass
class AnalysisResult:
    """Full output of a multi-agent analysis run."""

    baseline_answer: str
    report: SynthesisReport
    chunks: list[RetrievalResult]
    materialist_memo: StanceMemo
    idealist_memo: StanceMemo
    dualist_memo: StanceMemo
    verified_claims: list[VerifiedClaim] = field(default_factory=list)
    argument_map: ArgumentMap | None = None


logger = logging.getLogger(__name__)


def build_graph(pipeline: RAGPipeline, settings: Settings):  # type: ignore[return]
    """Compile and return the LangGraph StateGraph."""
    api_key = settings.openai_api_key.get_secret_value()
    client = OpenAI(api_key=api_key)
    model = settings.openai_chat_model

    def retrieve(state: AgentState) -> dict[str, list[RetrievalResult]]:
        logger.info("Retrieving context for: %s", state["question"][:80])
        evidence_sets = retrieve_stance_evidence(state["question"], pipeline)
        return {
            "chunks": evidence_sets.chunks,
            "materialist_chunks": evidence_sets.by_stance["materialist"],
            "idealist_chunks": evidence_sets.by_stance["idealist"],
            "dualist_chunks": evidence_sets.by_stance["dualist"],
        }

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

    baseline_answer = pipeline.answer_from_contexts(question, final["chunks"])
    verified_claims = verify_analysis_claims(
        baseline_answer=baseline_answer,
        report=report,
        chunks=final["chunks"],
        materialist_memo=mat,
        idealist_memo=ide,
        dualist_memo=dua,
    )
    argument_map = build_argument_map(report=report, memos=[mat, ide, dua])

    return AnalysisResult(
        baseline_answer=baseline_answer,
        report=report,
        chunks=final["chunks"],
        materialist_memo=mat,
        idealist_memo=ide,
        dualist_memo=dua,
        verified_claims=verified_claims,
        argument_map=argument_map,
    )


def retrieve_stance_evidence(
    question: str,
    pipeline: RAGPipeline,
) -> StanceEvidenceSets:
    """Retrieve base and stance-specific chunks with shared prompt IDs."""
    stances: tuple[StanceName, ...] = ("materialist", "idealist", "dualist")
    base_chunks = pipeline.retrieve(question)
    stance_chunks = {
        stance: pipeline.retrieve(build_stance_query(question, stance))
        for stance in stances
    }
    return build_stance_evidence_sets(
        base_chunks=base_chunks,
        stance_chunks=stance_chunks,
    )


def verify_analysis_claims(
    *,
    baseline_answer: str,
    report: SynthesisReport,
    chunks: list[RetrievalResult],
    materialist_memo: StanceMemo,
    idealist_memo: StanceMemo,
    dualist_memo: StanceMemo,
) -> list[VerifiedClaim]:
    """Extract and structurally verify claims from a complete analysis result."""
    claims = extract_analysis_claims(
        memos=[materialist_memo, idealist_memo, dualist_memo],
        report=report,
        baseline_answer=baseline_answer,
    )
    return verify_claims(claims, chunks)
