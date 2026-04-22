"""Prompt formatting helpers for corpus discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phil_mind_rag.agents.schema import SourceCandidate


def format_candidates(candidates: list[SourceCandidate]) -> str:
    """Format source candidates for structured source-selection prompts."""
    blocks = []
    for i, candidate in enumerate(candidates, start=1):
        authors = ", ".join(candidate.authors) if candidate.authors else "Unknown"
        citation_count = (
            str(candidate.citation_count)
            if candidate.citation_count is not None
            else "unknown"
        )
        blocks.append(
            "\n".join(
                [
                    f"[source_{i}] {candidate.title}",
                    f"Type: {candidate.source_type}",
                    f"Authors: {authors}",
                    "Year: "
                    f"{candidate.year if candidate.year is not None else 'unknown'}",
                    f"Venue: {candidate.venue or 'unknown'}",
                    f"DOI: {candidate.doi or 'unknown'}",
                    f"Citations: {citation_count}",
                    f"Source URL: {candidate.source_url or 'none'}",
                    f"Download URL: {candidate.download_url or 'none'}",
                    f"Access: {candidate.access_status}",
                    f"Access note: {candidate.access_note or 'none'}",
                    f"Abstract: {candidate.abstract or 'none'}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)
