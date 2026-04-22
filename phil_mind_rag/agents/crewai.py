"""Optional CrewAI orchestration path for framework comparison.

CrewAI is intentionally lazy-imported so LangGraph/plain orchestration remain
the default paths and the project does not require CrewAI unless this comparison
path is explicitly exercised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

from phil_mind_rag.agents.argument_map import build_argument_map
from phil_mind_rag.agents.graph import AnalysisResult, verify_analysis_claims
from phil_mind_rag.agents.schema import StanceMemo, SynthesisReport

if TYPE_CHECKING:
    from phil_mind_rag.config import Settings
    from phil_mind_rag.pipeline import RAGPipeline
    from phil_mind_rag.retrieval.store import RetrievalResult


class CrewAIUnavailableError(RuntimeError):
    """Raised when the optional CrewAI dependency is not installed."""


@dataclass(frozen=True)
class CrewAIArtifacts:
    """CrewAI objects created for inspection and debugging."""

    crew: Any
    agents: dict[str, Any]
    tasks: dict[str, Any]


def run_crewai_analysis(
    question: str,
    pipeline: RAGPipeline,
    settings: Settings,
) -> AnalysisResult:
    """Run the multi-agent analysis through CrewAI-compatible tasks.

    The expected CrewAI kickoff output is a JSON object with keys:
    `materialist_memo`, `idealist_memo`, `dualist_memo`, and `report`.
    Tests fake this output so the path is covered without paid model calls.
    """
    chunks = pipeline.retrieve(question)
    artifacts = build_crewai_artifacts(
        question=question,
        chunks=chunks,
        model=settings.openai_chat_model,
    )
    raw_output = artifacts.crew.kickoff(
        inputs={
            "question": question,
            "chunks": _format_chunks(chunks),
        }
    )
    mat, ide, dua, report = _parse_crewai_output(raw_output)
    baseline_answer = pipeline.answer_from_contexts(question, chunks)
    verified_claims = verify_analysis_claims(
        baseline_answer=baseline_answer,
        report=report,
        chunks=chunks,
        materialist_memo=mat,
        idealist_memo=ide,
        dualist_memo=dua,
    )
    argument_map = build_argument_map(report=report, memos=[mat, ide, dua])

    return AnalysisResult(
        baseline_answer=baseline_answer,
        report=report,
        chunks=chunks,
        materialist_memo=mat,
        idealist_memo=ide,
        dualist_memo=dua,
        verified_claims=verified_claims,
        argument_map=argument_map,
    )


def build_crewai_artifacts(
    *,
    question: str,
    chunks: list[RetrievalResult],
    model: str,
) -> CrewAIArtifacts:
    """Build CrewAI agents/tasks for the philosophy-of-mind graph."""
    crewai = _load_crewai()
    llm = _build_crewai_llm(crewai, model)

    agents = {
        "materialist": crewai.Agent(
            role="Materialist philosophy-of-mind stance agent",
            goal="Write the strongest source-grounded physicalist memo.",
            backstory=(
                "You foreground physicalist, functionalist, and neuroscientific "
                "arguments while citing retrieved chunk IDs."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        ),
        "idealist": crewai.Agent(
            role="Idealist philosophy-of-mind stance agent",
            goal="Write the strongest source-grounded idealist memo.",
            backstory=(
                "You foreground consciousness-first and anti-physicalist arguments "
                "while citing retrieved chunk IDs."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        ),
        "dualist": crewai.Agent(
            role="Dualist philosophy-of-mind stance agent",
            goal="Write the strongest source-grounded dualist memo.",
            backstory=(
                "You foreground property/substance dualism and engage rival "
                "positions charitably while citing retrieved chunk IDs."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        ),
        "grounding": crewai.Agent(
            role="Grounding and adjudication agent",
            goal="Audit stance memos against retrieved evidence and synthesize.",
            backstory=(
                "You make no metaphysical commitment; you flag unsupported claims, "
                "equivocations, and source gaps."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        ),
    }

    tasks = {
        "materialist": _stance_task(crewai, "materialist", agents["materialist"]),
        "idealist": _stance_task(crewai, "idealist", agents["idealist"]),
        "dualist": _stance_task(crewai, "dualist", agents["dualist"]),
        "grounding": crewai.Task(
            description=(
                "Using the retrieved chunks and all stance memos, produce a "
                "SynthesisReport JSON object. Include supported_claims, "
                "unsupported_claims, areas_of_disagreement, strongest_arguments, "
                "synthesis, decisive_chunks, and source_chunks_used. "
                f"Question: {question}\n\nChunks:\n{_format_chunks(chunks)}"
            ),
            expected_output=(
                "JSON object with top-level key `report` matching SynthesisReport."
            ),
            agent=agents["grounding"],
        ),
    }

    crew = crewai.Crew(
        agents=list(agents.values()),
        tasks=list(tasks.values()),
        process=getattr(crewai.Process, "sequential", None),
        verbose=False,
    )
    return CrewAIArtifacts(crew=crew, agents=agents, tasks=tasks)


def _stance_task(crewai: Any, stance: str, agent: Any) -> Any:
    return crewai.Task(
        description=(
            f"Write a {stance} StanceMemo JSON object from the retrieved chunks. "
            "Include stance, thesis, supporting_claims, rival_critiques, "
            "confidence, and uncertainty_notes. Every claim must cite chunk IDs."
        ),
        expected_output=f"JSON object with key `{stance}_memo` matching StanceMemo.",
        agent=agent,
    )


def _parse_crewai_output(
    raw_output: Any,
) -> tuple[StanceMemo, StanceMemo, StanceMemo, SynthesisReport]:
    payload = _coerce_output_payload(raw_output)
    return (
        StanceMemo.model_validate(payload["materialist_memo"]),
        StanceMemo.model_validate(payload["idealist_memo"]),
        StanceMemo.model_validate(payload["dualist_memo"]),
        SynthesisReport.model_validate(payload["report"]),
    )


def _coerce_output_payload(raw_output: Any) -> dict[str, Any]:
    if isinstance(raw_output, dict):
        return raw_output
    if hasattr(raw_output, "json_dict") and isinstance(raw_output.json_dict, dict):
        return raw_output.json_dict
    if hasattr(raw_output, "raw"):
        raw_output = raw_output.raw
    if isinstance(raw_output, str):
        return json.loads(raw_output)
    raise TypeError(
        "CrewAI output must be a dict, JSON string, or object with raw/json_dict"
    )


def _load_crewai() -> Any:
    try:
        return import_module("crewai")
    except ImportError as exc:
        raise CrewAIUnavailableError(
            "CrewAI is not installed. Install the optional `crewai` package to "
            "run this framework-comparison path."
        ) from exc


def _build_crewai_llm(crewai: Any, model: str) -> Any:
    llm_cls = getattr(crewai, "LLM", None)
    if llm_cls is None:
        return None
    return llm_cls(model=model)


def _format_chunks(chunks: list[RetrievalResult]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "?")
        section = chunk.metadata.get("section", "?")
        parts.append(f"[chunk_{i}] ({source} - {section})\n{chunk.text}")
    return "\n\n---\n\n".join(parts)
