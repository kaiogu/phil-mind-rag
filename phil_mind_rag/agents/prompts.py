"""Prompt builders for stance agents and the grounding agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phil_mind_rag.agents.schema import StanceMemo
    from phil_mind_rag.retrieval.store import RetrievalResult

_STANCE_SYSTEM = (
    "You are a philosopher of mind representing the {stance} position. "
    "{instruction} "
    "Ground every claim in the retrieved evidence below. "
    "Use the chunk IDs (chunk_0, chunk_1, …) to cite supporting passages. "
    "Respond strictly in the JSON schema provided — no prose outside it."
)

_GROUNDING_SYSTEM = (
    "You are an impartial philosophical adjudicator. "
    "You make no metaphysical commitments of your own. "
    "Compare each stance memo against the retrieved evidence. "
    "Flag claims that are not traceable to a cited chunk ID, "
    "equivocations, and source gaps. "
    "Identify genuine points of disagreement — not generic summaries. "
    "Respond strictly in the JSON schema provided — no prose outside it."
)


def _format_chunks(chunks: list[RetrievalResult]) -> str:
    parts = []
    for i, c in enumerate(chunks):
        src = c.metadata.get("source", "?")
        sec = c.metadata.get("section", "?")
        parts.append(f"[chunk_{i}] ({src} — {sec})\n{c.text}")
    return "\n\n---\n\n".join(parts)


def stance_prompt(
    stance: str,
    instruction: str,
    question: str,
    chunks: list[RetrievalResult],
) -> tuple[str, str]:
    """Return (system, user) strings for a stance agent."""
    system = _STANCE_SYSTEM.format(stance=stance, instruction=instruction)
    user = f"Question: {question}\n\nRetrieved evidence:\n\n{_format_chunks(chunks)}"
    return system, user


def grounding_prompt(
    question: str,
    chunks: list[RetrievalResult],
    memos: list[StanceMemo],
) -> tuple[str, str]:
    """Return (system, user) strings for the grounding/adjudication agent."""
    memos_block = "\n\n---\n\n".join(
        f"[{m.stance.upper()}]\n"
        f"Thesis: {m.thesis}\n"
        f"Supporting arguments: {m.supporting_arguments}\n"
        f"Attacks on rivals: {m.attack_on_rivals}\n"
        f"Citations: {m.citations}\n"
        f"Uncertainty: {m.uncertainty_notes}"
        for m in memos
    )
    user = (
        f"Question: {question}\n\n"
        f"Retrieved evidence:\n\n{_format_chunks(chunks)}\n\n"
        f"Stance memos:\n\n{memos_block}"
    )
    return _GROUNDING_SYSTEM, user
