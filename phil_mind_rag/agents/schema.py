"""Pydantic contracts between agents — do not change mid-implementation."""

from __future__ import annotations

from pydantic import BaseModel


class EvidenceClaim(BaseModel):
    text: str
    citations: list[str]


class StanceMemo(BaseModel):
    stance: str  # "materialist" | "idealist" | "dualist"
    thesis: str  # one-sentence position statement
    supporting_claims: list[EvidenceClaim]
    rival_critiques: list[EvidenceClaim]
    confidence: float  # 0.0–1.0
    uncertainty_notes: str  # what the agent is unsure about


class Claim(BaseModel):
    text: str
    stance: str
    supported: bool
    citations: list[str]
    source_chunk_id: str | None  # None if unsupported
    note: str  # grounding agent's annotation


class SynthesisReport(BaseModel):
    question: str
    areas_of_disagreement: list[str]
    strongest_arguments: dict[str, str]  # stance → best supported argument
    supported_claims: list[Claim]
    unsupported_claims: list[Claim]
    synthesis: str  # adjudication / unresolved remainder
    decisive_chunks: list[str]
    source_chunks_used: list[str]  # all chunk IDs cited across all memos
