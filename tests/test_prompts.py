"""Tests for RAGPrompt."""

from phil_mind_rag.generation.prompts import RAGPrompt
from phil_mind_rag.retrieval.store import RetrievalResult


def _result(
    text: str, source: str = "nagel.pdf", section: str = "Intro"
) -> RetrievalResult:
    return RetrievalResult(
        text=text, score=0.9, metadata={"source": source, "section": section}
    )


class TestRAGPrompt:
    def test_prompt_contains_user_question(self) -> None:
        built = RAGPrompt().build("What is the hard problem?", [_result("ctx")])
        assert "What is the hard problem?" in built

    def test_prompt_contains_context_text(self) -> None:
        built = RAGPrompt().build("Q?", [_result("Important philosophical passage.")])
        assert "Important philosophical passage." in built

    def test_prompt_contains_source_citation(self) -> None:
        built = RAGPrompt().build(
            "Q?", [_result("text", source="nagel.pdf", section="Section 2")]
        )
        assert "nagel.pdf" in built
        assert "Section 2" in built

    def test_prompt_includes_all_contexts(self) -> None:
        contexts = [
            _result("First passage."), _result("Second passage.", section="Conclusion")
        ]
        built = RAGPrompt().build("Q?", contexts)
        assert "First passage." in built
        assert "Second passage." in built

    def test_prompt_includes_system_instruction(self) -> None:
        built = RAGPrompt().build("Q?", [_result("ctx")])
        assert "Philosophy of Mind" in built

    def test_custom_system_text_is_used(self) -> None:
        prompt = RAGPrompt(system="Custom system.")
        built = prompt.build("Q?", [_result("ctx")])
        assert "Custom system." in built
        assert "Philosophy of Mind" not in built

    def test_missing_metadata_uses_question_mark_fallback(self) -> None:
        result = RetrievalResult(text="ctx", score=0.5, metadata={})
        built = RAGPrompt().build("Q?", [result])
        assert "?" in built  # fallback for missing source/section
