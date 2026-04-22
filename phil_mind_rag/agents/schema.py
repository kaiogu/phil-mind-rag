"""Pydantic contracts between agents — do not change mid-implementation."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    evidence_chunk_ids: list[str] = Field(default_factory=list)


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


class SourceCandidate(BaseModel):
    title: str
    source_type: str  # paper | book | blog | article | video | other
    authors: list[str]
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    abstract: str = ""
    citation_count: int | None = None
    source_url: str | None = None
    download_url: str | None = None
    access_status: str = "unknown"  # open | paywalled | copyrighted | unknown
    access_note: str | None = None


class SourceRecommendation(BaseModel):
    title: str
    source_type: str
    rationale: str
    priority: int  # 1 (highest) to 5 (lowest)
    relevance_to_question: str
    suggested_use: str
    doi: str | None = None
    source_url: str | None = None
    download_url: str | None = None
    access_status: str = "unknown"
    acquisition_note: str | None = None


class SourceDiscoveryReport(BaseModel):
    field: str
    question: str
    search_query: str
    recommendations: list[SourceRecommendation]
    gaps_or_followups: list[str]


class PaperCandidate(BaseModel):
    title: str
    authors: list[str]
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    abstract: str = ""
    citation_count: int | None = None
    pdf_url: str | None = None
    source_url: str | None = None


class PaperRecommendation(SourceRecommendation):
    pdf_url: str | None = None


class PaperDiscoveryReport(SourceDiscoveryReport):
    recommendations: list[PaperRecommendation]
