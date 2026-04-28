"""Tests for app formatting helpers."""

from phil_mind_rag.agents.schema import ArgumentMap, ArgumentMapClaim, ArgumentMapStance
from phil_mind_rag.app.formatting import format_argument_map_html


def _map(
    *,
    question: str = "What is consciousness?",
    axes: list[str] | None = None,
    synthesis: str = "The dispute remains open.",
    decisive_chunks: list[str] | None = None,
    stances: list[ArgumentMapStance] | None = None,
) -> ArgumentMap:
    return ArgumentMap(
        question=question,
        disagreement_axes=axes or ["reduction", "qualia"],
        stances=stances
        or [
            ArgumentMapStance(
                stance="materialist",
                thesis="Consciousness is physical.",
                strongest_argument="Neural correlates are well established.",
                supporting_claims=[
                    ArgumentMapClaim(
                        text="Brain states track conscious states.",
                        citations=["chunk_0"],
                        supported=True,
                        note="Directly grounded.",
                    )
                ],
                objections=[
                    ArgumentMapClaim(
                        text="Idealists lack empirical footing.",
                        citations=["chunk_1"],
                        supported=False,
                    )
                ],
            )
        ],
        decisive_chunks=["chunk_0"] if decisive_chunks is None else decisive_chunks,
        synthesis=synthesis,
    )


class TestFormatArgumentMapHtml:
    def test_returns_placeholder_for_none(self) -> None:
        result = format_argument_map_html(None)
        assert "<em>" in result
        assert "No argument map" in result

    def test_contains_disagreement_axes(self) -> None:
        html = format_argument_map_html(_map(axes=["reduction", "qualia"]))
        assert "reduction" in html
        assert "qualia" in html

    def test_contains_stance_name(self) -> None:
        html = format_argument_map_html(_map())
        assert "Materialist" in html

    def test_contains_thesis(self) -> None:
        html = format_argument_map_html(_map())
        assert "Consciousness is physical." in html

    def test_contains_strongest_argument(self) -> None:
        html = format_argument_map_html(_map())
        assert "Neural correlates are well established." in html

    def test_contains_supporting_claim_text(self) -> None:
        html = format_argument_map_html(_map())
        assert "Brain states track conscious states." in html

    def test_supported_claim_gets_green_badge(self) -> None:
        html = format_argument_map_html(_map())
        assert "#4caf50" in html  # green badge for supported=True

    def test_flagged_claim_gets_amber_badge(self) -> None:
        html = format_argument_map_html(_map())
        assert "#f59e0b" in html  # amber badge for supported=False

    def test_contains_synthesis(self) -> None:
        html = format_argument_map_html(_map(synthesis="Remains contested."))
        assert "Remains contested." in html

    def test_contains_decisive_chunks(self) -> None:
        html = format_argument_map_html(_map(decisive_chunks=["chunk_42"]))
        assert "chunk_42" in html

    def test_html_escapes_user_content(self) -> None:
        map_with_xss = _map(synthesis='<script>alert("xss")</script>')
        html = format_argument_map_html(map_with_xss)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_empty_decisive_chunks_omits_section(self) -> None:
        html = format_argument_map_html(_map(decisive_chunks=[]))
        assert "Decisive evidence" not in html

    def test_none_adjudicated_claim_gets_gray_badge(self) -> None:
        stance = ArgumentMapStance(
            stance="idealist",
            thesis="Mind is primary.",
            supporting_claims=[
                ArgumentMapClaim(
                    text="Qualia are irreducible.", citations=[], supported=None
                )
            ],
            objections=[],
        )
        html = format_argument_map_html(_map(stances=[stance]))
        assert "#9e9e9e" in html  # gray for not adjudicated
