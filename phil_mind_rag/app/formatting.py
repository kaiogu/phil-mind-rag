"""Markdown formatting helpers for the Gradio frontend."""

from __future__ import annotations

import html as _html
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


_STANCE_COLORS: dict[str, tuple[str, str]] = {
    "materialist": ("#4a90d9", "#e8f1fb"),
    "idealist": ("#7b61ff", "#f0edff"),
    "dualist": ("#00897b", "#e0f4f1"),
}
_DEFAULT_STANCE_COLOR: tuple[str, str] = ("#6b7280", "#f3f4f6")


def _claim_chip_html(claim: ArgumentMapClaim) -> str:
    citations = ", ".join(claim.citations) if claim.citations else "none"
    if claim.supported is True:
        badge_color, symbol, bg = "#4caf50", "✓", "#f0fdf4"
    elif claim.supported is False:
        badge_color, symbol, bg = "#f59e0b", "⚠", "#fffbeb"
    else:
        badge_color, symbol, bg = "#9e9e9e", "–", "#f9fafb"
    note_html = (
        f'<span style="color:#666;font-size:0.82em">'
        f" — {_html.escape(claim.note)}</span>"
        if claim.note
        else ""
    )
    return (
        f'<div style="background:{bg};border-left:3px solid {badge_color};'
        f'border-radius:4px;padding:5px 8px;margin:4px 0;font-size:0.88em">'
        f"{_html.escape(claim.text)}"
        f'<span style="color:#666;font-size:0.82em"> [{_html.escape(citations)}]</span>'
        f'<span style="background:{badge_color};color:white;border-radius:3px;'
        f'padding:1px 5px;font-size:0.8em;margin-left:4px">{symbol}</span>'
        f"{note_html}</div>"
    )


def format_argument_map_html(argument_map: ArgumentMap | None) -> str:
    if argument_map is None:
        return "<p><em>No argument map was produced.</em></p>"

    axes_html = "".join(
        f'<span style="background:#f0f0f0;border-radius:4px;padding:2px 8px;'
        f'margin:2px;display:inline-block;font-size:0.9em">'
        f"{_html.escape(a)}</span>"
        for a in argument_map.disagreement_axes
    )

    cards_html = ""
    for stance_data in argument_map.stances:
        border, bg = _STANCE_COLORS.get(
            stance_data.stance.lower(), _DEFAULT_STANCE_COLOR
        )
        strongest_html = (
            f'<p style="font-size:0.85em;color:#555;margin:4px 0 8px">'
            f"<b>Strongest argument:</b> "
            f"{_html.escape(stance_data.strongest_argument)}</p>"
            if stance_data.strongest_argument
            else ""
        )
        supporting_html = (
            "".join(_claim_chip_html(c) for c in stance_data.supporting_claims)
            or "<p style='color:#aaa;font-size:0.85em'>No supporting claims.</p>"
        )
        objections_html = (
            "".join(_claim_chip_html(c) for c in stance_data.objections)
            or "<p style='color:#aaa;font-size:0.85em'>No objections recorded.</p>"
        )

        cards_html += (
            f'<div style="border:2px solid {border};border-radius:8px;'
            f'padding:12px;background:{bg}">'
            f'<b style="color:{border};font-size:1.05em">'
            f"{_html.escape(stance_data.stance.capitalize())}</b>"
            f'<p style="font-style:italic;font-size:0.9em;margin:6px 0">'
            f"{_html.escape(stance_data.thesis)}</p>"
            f"{strongest_html}"
            f'<p style="font-size:0.85em;font-weight:bold;margin:8px 0 4px">'
            f"Supporting Claims</p>"
            f"{supporting_html}"
            f'<p style="font-size:0.85em;font-weight:bold;margin:8px 0 4px">'
            f"Objections to Rivals</p>"
            f"{objections_html}"
            f"</div>"
        )

    decisive_html = (
        f'<p style="font-size:0.85em;color:#555;margin-top:8px">'
        f"<b>Decisive evidence:</b> "
        f"{_html.escape(', '.join(argument_map.decisive_chunks))}</p>"
        if argument_map.decisive_chunks
        else ""
    )

    return (
        f'<div style="font-family:sans-serif;max-width:960px;padding:8px">'
        f'<div style="margin-bottom:12px">'
        f"<b>Areas of disagreement:</b> {axes_html}</div>"
        f'<div style="display:grid;grid-template-columns:'
        f'repeat(auto-fit,minmax(260px,1fr));gap:12px">'
        f"{cards_html}</div>"
        f'<div style="margin-top:14px;border-top:1px solid #e0e0e0;padding-top:10px">'
        f"<b>Synthesis:</b> {_html.escape(argument_map.synthesis)}</div>"
        f"{decisive_html}</div>"
    )


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
        state = getattr(result, "state", result.status).replace("_", " ")
        if result.status == "already-indexed":
            lines.append(
                f"- Already indexed **{result.title}** ({state}): {result.skip_reason}"
            )
            continue
        if result.success:
            extra = (
                f" — ingested {result.ingested_chunks} chunks"
                if result.ingested_chunks is not None
                else ""
            )
            lines.append(f"- Downloaded **{result.title}** ({state}){extra}.")
            continue
        if result.skipped:
            lines.append(
                f"- Skipped **{result.title}** ({state}): {result.skip_reason}"
            )
            continue
        lines.append(f"- Failed **{result.title}** ({state}): {result.error}")
    return "\n".join(lines)
