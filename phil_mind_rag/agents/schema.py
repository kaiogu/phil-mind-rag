"""Pydantic contracts between agents — do not change mid-implementation."""

from __future__ import annotations

from pydantic import BaseModel


class StanceMemo(BaseModel):
    stance: str  # "materialist" | "idealist" | "dualist"
    thesis: str  # one-sentence position statement
    supporting_arguments: list[str]  # strongest arguments for the stance
    attack_on_rivals: list[str]  # strongest objections to opposing stances
    confidence: float  # 0.0–1.0
    uncertainty_notes: str  # what the agent is unsure about
    citations: list[str]  # chunk IDs from the retrieval result set


class Claim(BaseModel):
    text: str
    stance: str
    supported: bool
    source_chunk_id: str | None  # None if unsupported
    note: str  # grounding agent's annotation


class SynthesisReport(BaseModel):
    question: str
    areas_of_disagreement: list[str]
    strongest_arguments: dict[str, str]  # stance → best supported argument
    unsupported_claims: list[Claim]
    synthesis: str  # adjudication / unresolved remainder
    source_chunks_used: list[str]  # all chunk IDs cited across all memos
