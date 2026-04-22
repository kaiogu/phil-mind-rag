"""Markdown formatting helpers for the Gradio frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phil_mind_rag.agents.schema import (
        ArgumentMap,
        ArgumentMapClaim,
        SourceDiscoveryReport,
        StanceMemo,
        SynthesisReport,
        VerifiedClaim,
    )
    from phil_mind_rag.retrieval.store import RetrievalResult

LIBRARY_COLUMNS = [
    "Title",
    "Author",
    "Source File",
    "Chunks",
    "Chunk Size",
    "Chunk Overlap",
    "Chunker",
    "Embedding Model",
    "Ingested At",
]


def format_stance_memo(memo: StanceMemo) -> str:
    lines = [f"**Thesis:** {memo.thesis}", f"\n**Confidence:** {memo.confidence:.0%}"]
    lines.append("\n**Supporting Claims:**")
    for claim in memo.supporting_claims:
        citations = ", ".join(claim.citations) if claim.citations else "none"
        lines.append(f"- {claim.text} _[{citations}]_")
    lines.append("\n**Critiques of Rivals:**")
    for claim in memo.rival_critiques:
        citations = ", ".join(claim.citations) if claim.citations else "none"
        lines.append(f"- {claim.text} _[{citations}]_")
    lines.append(f"\n**Uncertainty:** {memo.uncertainty_notes}")
    return "\n".join(lines)


def format_grounding(report: SynthesisReport) -> str:
    lines = []
    if report.supported_claims:
        lines.append("**Supported Claims:**\n")
        for claim in report.supported_claims:
            citations = ", ".join(claim.citations) if claim.citations else "none"
            lines.append(
                "✓ "
                f"**[{claim.stance}]** {claim.text} _[{citations}]_\n\n"
                f"   _{claim.note}_"
            )
    if not report.unsupported_claims:
        if lines:
            lines.append(
                "\n\n_All remaining cited claims are supported by the "
                "retrieved evidence._"
            )
            return "\n\n".join(lines)
        return "_All cited claims are supported by the retrieved evidence._"
    lines.append("\n\n**Flagged Claims:**\n")
    for claim in report.unsupported_claims:
        citations = ", ".join(claim.citations) if claim.citations else "none"
        lines.append(
            f"✗ **[{claim.stance}]** {claim.text} _[{citations}]_\n\n   _{claim.note}_"
        )
    return "\n\n".join(lines)


def format_verified_claims(verified_claims: list[VerifiedClaim]) -> str:
    if not verified_claims:
        return "_No claim-verification audit was produced._"

    label_symbols = {
        "supported": "✓",
        "unsupported": "✗",
        "ambiguous": "!",
    }
    lines = ["**Claim Verification Audit:**"]
    for verified in verified_claims:
        citations = ", ".join(verified.citations) if verified.citations else "none"
        repaired = (
            f" Repaired: {', '.join(verified.repaired_citations)}."
            if verified.repaired_citations
            else ""
        )
        symbol = label_symbols.get(verified.label, "?")
        source = verified.claim.source.replace("_", " ")
        stance = f" / {verified.claim.stance}" if verified.claim.stance else ""
        lines.append(
            f"{symbol} **{verified.label}** ({source}{stance}) "
            f"{verified.claim.text} _[{citations}]_\n\n"
            f"   _{verified.note}{repaired}_"
        )
    return "\n\n".join(lines)


def format_synthesis(report: SynthesisReport) -> str:
    lines = ["**Areas of Disagreement:**"]
    for area in report.areas_of_disagreement:
        lines.append(f"- {area}")
    if report.strongest_arguments:
        lines.append("\n**Strongest Supported Arguments:**")
        for stance, arg in report.strongest_arguments.items():
            lines.append(f"- **{stance.capitalize()}:** {arg}")
    if report.decisive_chunks:
        lines.append(f"\n**Decisive Evidence:** {', '.join(report.decisive_chunks)}")
    lines.append(f"\n**Synthesis:**\n\n{report.synthesis}")
    return "\n".join(lines)


def format_argument_map(argument_map: ArgumentMap | None) -> str:
    if argument_map is None:
        return "_No argument map was produced._"

    lines = ["**Argument Map:**", f"\n**Question:** {argument_map.question}"]
    if argument_map.disagreement_axes:
        lines.append("\n**Disagreement Axes:**")
        lines.extend(f"- {axis}" for axis in argument_map.disagreement_axes)
    for stance in argument_map.stances:
        lines.append(f"\n**{stance.stance.capitalize()} Position:** {stance.thesis}")
        if stance.strongest_argument:
            lines.append(f"- Strongest argument: {stance.strongest_argument}")
        lines.append("- Supporting claims:")
        lines.extend(
            f"  - {format_map_claim(claim)}" for claim in stance.supporting_claims
        )
        lines.append("- Objections:")
        lines.extend(f"  - {format_map_claim(claim)}" for claim in stance.objections)
    if argument_map.decisive_chunks:
        lines.append(
            f"\n**Decisive Evidence:** {', '.join(argument_map.decisive_chunks)}"
        )
    return "\n".join(lines)


def format_map_claim(claim: ArgumentMapClaim) -> str:
    citations = ", ".join(claim.citations) if claim.citations else "none"
    if claim.supported is True:
        support = "supported"
    elif claim.supported is False:
        support = "flagged"
    else:
        support = "not adjudicated"
    note = f" — {claim.note}" if claim.note else ""
    return f"{claim.text} _[{citations}; {support}]_{note}"


def format_sources(chunks: list[RetrievalResult]) -> str:
    if not chunks:
        return ""
    lines = ["**Retrieved sources:**\n"]
    for i, result in enumerate(chunks):
        src = result.metadata.get("source", "?")
        section = result.metadata.get("section", "?")
        snippet = result.text[:200].replace("\n", " ")
        lines.append(
            f"**chunk_{i}** — {src} / {section} "
            f"(score: {result.score:.3f})\n> {snippet}…"
        )
    return "\n\n".join(lines)


def format_discovery_report(
    report: SourceDiscoveryReport,
    *,
    web_search_enabled: bool,
) -> str:
    lines = [f"**Search Query:** {report.search_query}", "", "**Recommendations:**"]
    for recommendation in report.recommendations:
        lines.append(
            f"- **[{recommendation.priority}] {recommendation.title}** "
            f"({recommendation.source_type})"
        )
        lines.append(f"  {recommendation.rationale}")
        lines.append(f"  Relevance: {recommendation.relevance_to_question}")
        lines.append(f"  Suggested use: {recommendation.suggested_use}")
        lines.append(f"  Access: {recommendation.access_status}")
        if recommendation.doi:
            lines.append(f"  DOI: {recommendation.doi}")
        if recommendation.acquisition_note:
            lines.append(f"  Acquisition note: {recommendation.acquisition_note}")
        if recommendation.source_url:
            lines.append(f"  URL: {recommendation.source_url}")
    if report.gaps_or_followups:
        lines.append("")
        lines.append("**Gaps / Follow-Ups:**")
        for gap in report.gaps_or_followups:
            lines.append(f"- {gap}")
    if not web_search_enabled:
        lines.append("")
        lines.append(
            "_Normal web search is not configured. Enable the OpenAI web-search "
            "provider to add books, blogs, and other web results alongside "
            "OpenAlex._"
        )
    return "\n".join(lines)


def format_acquisition_results(results: list) -> str:
    lines = ["**Acquisition Results:**"]
    for result in results:
        if result.status == "already-indexed":
            lines.append(f"- Already indexed **{result.title}**: {result.skip_reason}")
            continue
        if result.success:
            extra = (
                f" — ingested {result.ingested_chunks} chunks"
                if result.ingested_chunks is not None
                else ""
            )
            lines.append(f"- Downloaded **{result.title}**{extra}.")
            continue
        if result.skipped:
            lines.append(f"- Skipped **{result.title}**: {result.skip_reason}")
            continue
        lines.append(f"- Failed **{result.title}**: {result.error}")
    return "\n".join(lines)
