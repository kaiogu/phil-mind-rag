"""Prompt builders for stance agents and the grounding agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phil_mind_rag.agents.schema import StanceMemo
    from phil_mind_rag.retrieval.store import RetrievalResult

_STANCE_SYSTEM = """# Identity
You are a philosopher of mind representing the {stance} position.

# Task
{instruction}
Answer the research question by building the strongest charitable memo you can
for this stance using only the retrieved evidence.

# Rules
- Use only the retrieved evidence. Do not use outside knowledge.
- Every supporting claim and every rival critique must cite one or more chunk
  IDs that directly support the claim.
- Never invent chunk IDs or cite chunks that only weakly or indirectly relate
  to the claim.
- Prefer fewer, stronger claims over many weak or repetitive claims.
- Engage rival positions seriously before criticizing them.
- If the evidence is thin or mixed, make the memo narrower and record the
  limitation in `uncertainty_notes`.

# Output Requirements
- Return strict JSON matching the provided schema.
- Do not include any prose outside the JSON.
"""

_GROUNDING_SYSTEM = """# Identity
You are an impartial philosophical adjudicator.

# Task
Compare the stance memos against the retrieved evidence and produce a grounded
adjudication report.

# Rules
- Use only the retrieved evidence and the stance memos.
- Audit each claim against its cited chunk IDs.
- Prefer marking a claim unsupported over guessing.
- Flag missing citations, invalid citations, source gaps, and equivocations.
- Identify genuine disagreements between the positions, not generic summaries.
- Include only chunk IDs that appear in the retrieved evidence.

# Output Requirements
- Populate both `supported_claims` and `unsupported_claims` when warranted.
- `decisive_chunks` should include only chunks that materially affect the
  adjudication.
- `source_chunks_used` should include all chunk IDs actually relied on.
- Return strict JSON matching the provided schema.
- Do not include any prose outside the JSON.
"""


def _format_chunks(chunks: list[RetrievalResult]) -> str:
    parts = []
    for i, c in enumerate(chunks):
        chunk_id = c.metadata.get("chunk_id", f"chunk_{i}")
        src = c.metadata.get("source", "?")
        sec = c.metadata.get("section", "?")
        parts.append(f"[{chunk_id}] ({src} — {sec})\n{c.text}")
    return "\n\n---\n\n".join(parts)


def stance_prompt(
    stance: str,
    instruction: str,
    question: str,
    chunks: list[RetrievalResult],
) -> tuple[str, str]:
    """Return (system, user) strings for a stance agent."""
    system = _STANCE_SYSTEM.format(stance=stance, instruction=instruction)
    user = (
        "<question>\n"
        f"{question}\n"
        "</question>\n\n"
        "<retrieved_evidence>\n"
        f"{_format_chunks(chunks)}\n"
        "</retrieved_evidence>"
    )
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
        f"Supporting claims:\n{_format_claims(m.supporting_claims)}\n"
        f"Critiques of rivals:\n{_format_claims(m.rival_critiques)}\n"
        f"Uncertainty: {m.uncertainty_notes}"
        for m in memos
    )
    user = (
        "<question>\n"
        f"{question}\n"
        "</question>\n\n"
        "<retrieved_evidence>\n"
        f"{_format_chunks(chunks)}\n"
        "</retrieved_evidence>\n\n"
        "<stance_memos>\n"
        f"{memos_block}\n"
        "</stance_memos>"
    )
    return _GROUNDING_SYSTEM, user


def _format_claims(claims: list) -> str:
    if not claims:
        return "- none"
    return "\n".join(
        f"- {claim.text} (citations: {', '.join(claim.citations) or 'none'})"
        for claim in claims
    )
