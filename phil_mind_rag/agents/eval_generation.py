"""Eval-generation agent for corpus-pinned question sets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from phil_mind_rag.agents._llm import generate_structured

if TYPE_CHECKING:
    from openai import OpenAI

    from phil_mind_rag.retrieval.store import RetrievalResult

EvalQuestionType = Literal[
    "factual_retrieval",
    "stance_divergence",
    "grounding_fidelity",
]

_EVAL_GENERATION_SYSTEM = (
    "You are an eval designer for a philosophy-of-mind RAG system. "
    "Generate evaluation questions only from the supplied source chunks. "
    "Every question must be pinned to exactly one provided chunk ID. "
    "Do not use outside knowledge or free-floating philosophical priors. "
    "Prefer questions that test retrieval, traceability, and argument structure. "
    "Return strict JSON matching the provided schema."
)


class GeneratedEvalQuestion(BaseModel):
    """One corpus-pinned generated eval question."""

    question_type: EvalQuestionType
    question: str
    reference_answer: str
    source_chunk_id: str
    source_excerpt: str
    expected_stances: list[str] = []
    grading_note: str


class GeneratedEvalSet(BaseModel):
    """Structured output from the eval-generation agent."""

    field: str
    source: str
    questions: list[GeneratedEvalQuestion]
    human_review_notes: list[str]


def generate_eval_set(
    *,
    field: str,
    source: str,
    chunks: list[RetrievalResult],
    client: OpenAI,
    model: str,
    questions_per_chunk: int = 3,
) -> GeneratedEvalSet:
    """Generate a source-grounded eval question set from retrieved chunks."""
    if not field.strip():
        raise ValueError("field must not be blank")
    if not source.strip():
        raise ValueError("source must not be blank")
    if not chunks:
        raise ValueError("chunks must not be empty")
    if questions_per_chunk < 1:
        raise ValueError("questions_per_chunk must be at least 1")

    user = (
        f"Field: {field.strip()}\n"
        f"Source: {source.strip()}\n"
        f"Questions per chunk: {questions_per_chunk}\n\n"
        "Required question types:\n"
        "- factual_retrieval: asks for information directly recoverable from "
        "the source chunk.\n"
        "- stance_divergence: asks a question where materialist, idealist, "
        "and dualist interpretations should diverge, based on the chunk's "
        "actual argument structure.\n"
        "- grounding_fidelity: asks a probe with a known source passage so "
        "claim traceability can be checked mechanically.\n\n"
        "Source chunks:\n\n"
        f"{_format_eval_chunks(chunks)}"
    )
    eval_set = generate_structured(
        client=client,
        model=model,
        system=_EVAL_GENERATION_SYSTEM,
        user=user,
        schema_cls=GeneratedEvalSet,
    )
    _validate_generated_eval_set(eval_set, chunk_count=len(chunks))
    return eval_set


def to_ragas_eval_rows(eval_set: GeneratedEvalSet) -> list[dict[str, str]]:
    """Convert generated evals to the existing RAGAS eval_set.json shape."""
    return [
        {
            "question": question.question,
            "ground_truth": question.reference_answer,
        }
        for question in eval_set.questions
    ]


def _validate_generated_eval_set(
    eval_set: GeneratedEvalSet,
    *,
    chunk_count: int,
) -> None:
    valid_chunk_ids = {f"chunk_{i}" for i in range(chunk_count)}
    invalid = [
        question.source_chunk_id
        for question in eval_set.questions
        if question.source_chunk_id not in valid_chunk_ids
    ]
    if invalid:
        raise ValueError(
            "generated eval questions cite unknown chunk IDs: "
            + ", ".join(sorted(set(invalid)))
        )

    missing_grounding = [
        question.question
        for question in eval_set.questions
        if not question.source_excerpt.strip() or not question.reference_answer.strip()
    ]
    if missing_grounding:
        raise ValueError("generated eval questions must include grounded answers")


def _format_eval_chunks(chunks: list[RetrievalResult]) -> str:
    blocks: list[str] = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "?")
        section = chunk.metadata.get("section", "?")
        blocks.append(
            "\n".join(
                [
                    f"[chunk_{i}]",
                    f"Source: {source}",
                    f"Section: {section}",
                    f"Text: {chunk.text}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)
