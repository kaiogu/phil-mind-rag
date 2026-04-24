"""Grounding / adjudication agent node function."""

from __future__ import annotations

from typing import TYPE_CHECKING

from phil_mind_rag.agents._llm import generate_structured
from phil_mind_rag.agents.prompts import grounding_prompt
from phil_mind_rag.agents.schema import Claim, StanceMemo, SynthesisReport

if TYPE_CHECKING:
    from openai import OpenAI

    from phil_mind_rag.agents.state import AgentState


def run_grounding(
    state: AgentState, client: OpenAI, model: str | tuple[str, ...]
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
    report = _merge_deterministic_audit(report, memos, len(state["chunks"]))
    return {"report": report}


def _merge_deterministic_audit(
    report: SynthesisReport,
    memos: list[StanceMemo],
    chunk_count: int,
) -> SynthesisReport:
    valid_chunk_ids = {f"chunk_{i}" for i in range(chunk_count)}
    deterministic_flags = _audit_claim_citations(memos, valid_chunk_ids)
    unsupported = _dedupe_claims([*report.unsupported_claims, *deterministic_flags])
    unsupported_keys = {
        (claim.text, claim.stance, tuple(claim.citations)) for claim in unsupported
    }
    supported = [
        claim
        for claim in _dedupe_claims(report.supported_claims)
        if (claim.text, claim.stance, tuple(claim.citations)) not in unsupported_keys
    ]

    report_chunks = [
        chunk_id
        for chunk_id in report.source_chunks_used
        if chunk_id in valid_chunk_ids
    ]
    memo_chunks = sorted(
        {
            citation
            for memo in memos
            for claim in (*memo.supporting_claims, *memo.rival_critiques)
            for citation in claim.citations
            if citation in valid_chunk_ids
        }
    )
    decisive_chunks = [
        chunk_id for chunk_id in report.decisive_chunks if chunk_id in valid_chunk_ids
    ]

    return report.model_copy(
        update={
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "source_chunks_used": sorted(set([*report_chunks, *memo_chunks])),
            "decisive_chunks": sorted(set(decisive_chunks)),
        }
    )


def _audit_claim_citations(
    memos: list[StanceMemo], valid_chunk_ids: set[str]
) -> list[Claim]:
    findings: list[dict[str, object]] = []
    for memo in memos:
        for claim in (*memo.supporting_claims, *memo.rival_critiques):
            if not claim.citations:
                findings.append(
                    {
                        "text": claim.text,
                        "stance": memo.stance,
                        "supported": False,
                        "citations": [],
                        "source_chunk_id": None,
                        "note": "Claim provides no chunk citations.",
                    }
                )
                continue

            invalid = [
                citation
                for citation in claim.citations
                if citation not in valid_chunk_ids
            ]
            if invalid:
                findings.append(
                    {
                        "text": claim.text,
                        "stance": memo.stance,
                        "supported": False,
                        "citations": claim.citations,
                        "source_chunk_id": None,
                        "note": f"Claim cites unknown chunk IDs: {', '.join(invalid)}.",
                    }
                )
    return _dedupe_claims_dicts(findings)


def _dedupe_claims(claims: list[Claim]) -> list[Claim]:
    seen: set[tuple[str, str, tuple[str, ...], bool]] = set()
    deduped: list[Claim] = []
    for claim in claims:
        key = (
            claim.text,
            claim.stance,
            tuple(claim.citations),
            claim.supported,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped


def _dedupe_claims_dicts(claims: list[dict[str, object]]) -> list[Claim]:
    models = [Claim.model_validate(claim) for claim in claims]
    return _dedupe_claims(models)
