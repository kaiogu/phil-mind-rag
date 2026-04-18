"""Gradio frontend for the Philosophy of Mind multi-agent RAG system."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

from phil_mind_rag.agents.graph import AnalysisResult, run_analysis
from phil_mind_rag.config import Settings, get_settings
from phil_mind_rag.pipeline import RAGPipeline

if TYPE_CHECKING:
    from phil_mind_rag.agents.schema import (  # SynthesisReport used in format helpers
        StanceMemo,
        SynthesisReport,
    )
    from phil_mind_rag.retrieval.store import RetrievalResult

logger = logging.getLogger(__name__)

# Lazy singletons — built once on first use.
_pipeline: RAGPipeline | None = None
_settings: Settings | None = None


def _get_settings() -> Settings:
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = get_settings()
    return _settings


def _get_pipeline() -> RAGPipeline:
    global _pipeline  # noqa: PLW0603
    if _pipeline is None:
        _pipeline = RAGPipeline(_get_settings())
    return _pipeline


# --- Format helpers -------------------------------------------------------


def _format_stance_memo(memo: StanceMemo) -> str:
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


def _format_grounding(report: SynthesisReport) -> str:
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


def _format_synthesis(report: SynthesisReport) -> str:
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


def _format_sources(chunks: list[RetrievalResult]) -> str:
    if not chunks:
        return ""
    lines = ["**Retrieved sources:**\n"]
    for i, r in enumerate(chunks):
        src = r.metadata.get("source", "?")
        section = r.metadata.get("section", "?")
        snippet = r.text[:200].replace("\n", " ")
        lines.append(
            f"**chunk_{i}** — {src} / {section} (score: {r.score:.3f})\n> {snippet}…"
        )
    return "\n\n".join(lines)


# --- Library & upload callbacks (unchanged) --------------------------------

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


def handle_refresh() -> list[list[object]]:
    records = _get_pipeline().list_documents()
    return [
        [
            r.title,
            r.author,
            r.source,
            r.chunk_count,
            r.chunk_size,
            r.chunk_overlap,
            r.chunker.split(".")[-1],
            r.embedding_model,
            r.ingested_at[:19].replace("T", " "),
        ]
        for r in records
    ]


def handle_extract_metadata(file: str | None) -> tuple[str, str]:
    if file is None:
        return "", ""
    try:
        meta = _get_pipeline().extract_metadata(Path(file))
        return meta.title or "", meta.author or ""
    except Exception:
        logger.exception("Metadata extraction failed for %s", file)
        return "", ""


def handle_upload(
    file: str | None,
    title: str | None,
    author: str | None,
):
    if file is None:
        yield "No file uploaded."
        return

    path = Path(file)
    pipeline = _get_pipeline()

    try:
        yield f"⏳ **{path.name}** — Parsing PDF..."
        document = pipeline.parse(path)

        yield f"⏳ **{path.name}** — Chunking sections..."
        chunks = pipeline.chunk(document)

        if not chunks:
            yield f"⚠️ **{path.name}** — No chunks produced."
            return

        yield f"⏳ **{path.name}** — Embedding {len(chunks)} chunks..."
        embeddings = pipeline.embed(chunks)

        yield f"⏳ **{path.name}** — Storing in vector database..."
        pipeline.store(chunks, embeddings)

        pipeline.register(path, len(chunks), title, author)

        yield f"✅ Ingested **{path.name}** — {len(chunks)} chunks indexed."
    except ValueError as exc:
        yield f"Validation error: {exc}"
    except Exception:
        logger.exception("Ingestion failed for %s", path)
        yield "An unexpected error occurred during ingestion."


# --- Analysis callback ----------------------------------------------------


def handle_analysis(
    question: str,
) -> tuple[str, str, str, str, str, str, str]:
    """Run multi-agent analysis.

    Returns
    (baseline, materialist, idealist, dualist, grounding, synthesis, sources).
    """
    if not question.strip():
        empty = "Please enter a question."
        return empty, empty, empty, empty, empty, empty, ""

    try:
        pipeline = _get_pipeline()
        settings = _get_settings()
        result: AnalysisResult = run_analysis(question, pipeline, settings)
        return (
            result.baseline_answer,
            _format_stance_memo(result.materialist_memo),
            _format_stance_memo(result.idealist_memo),
            _format_stance_memo(result.dualist_memo),
            _format_grounding(result.report),
            _format_synthesis(result.report),
            _format_sources(result.chunks),
        )
    except ValueError as exc:
        err = f"Input error: {exc}"
        return err, err, err, err, err, err, ""
    except Exception:
        logger.exception("Analysis failed")
        err = "An unexpected error occurred during analysis."
        return err, err, err, err, err, err, ""


# --- UI -------------------------------------------------------------------


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(title="Philosophy of Mind — Multi-Agent RAG") as app:
        gr.Markdown(
            "# Philosophy of Mind — Multi-Agent RAG\n"
            "Upload academic papers (PDF), then ask a question to receive "
            "three grounded stance memos and an adjudicated synthesis report."
        )

        with gr.Tab("Ask"):
            question_input = gr.Textbox(
                label="Question",
                placeholder="e.g. What is the hard problem of consciousness?",
                lines=2,
            )
            ask_btn = gr.Button("Analyse", variant="primary")

            gr.Markdown("### Single-Agent Baseline")
            baseline_output = gr.Markdown(label="Baseline answer")

            gr.Markdown("### Stance Memos")
            with gr.Tabs():
                with gr.Tab("Materialist"):
                    mat_output = gr.Markdown(label="Materialist memo")
                with gr.Tab("Idealist"):
                    ide_output = gr.Markdown(label="Idealist memo")
                with gr.Tab("Dualist"):
                    dua_output = gr.Markdown(label="Dualist memo")

            gr.Markdown("### Grounding")
            grounding_output = gr.Markdown(label="Grounding")

            gr.Markdown("### Synthesis")
            synthesis_output = gr.Markdown(label="Synthesis")

            gr.Markdown("### Sources")
            sources_output = gr.Markdown(label="Sources")

            ask_btn.click(
                fn=handle_analysis,
                inputs=question_input,
                outputs=[
                    baseline_output,
                    mat_output,
                    ide_output,
                    dua_output,
                    grounding_output,
                    synthesis_output,
                    sources_output,
                ],
            )

        with gr.Tab("Upload"):
            title_input = gr.Textbox(
                label="Title (optional)",
                placeholder="e.g. Facing Up to the Problem of Consciousness",
                value="",
            )
            author_input = gr.Textbox(
                label="Author (optional)",
                placeholder="e.g. David Chalmers",
                value="",
            )
            file_input = gr.File(
                label="Upload a PDF paper",
                file_types=[".pdf"],
                type="filepath",
            )
            file_input.change(
                fn=handle_extract_metadata,
                inputs=file_input,
                outputs=[title_input, author_input],
            )
            upload_btn = gr.Button("Ingest")
            upload_output = gr.Markdown(label="Status")

            upload_btn.click(
                fn=handle_upload,
                inputs=[file_input, title_input, author_input],
                outputs=upload_output,
            )

        with gr.Tab("Library"):
            gr.Markdown("### Ingested Papers")
            refresh_btn = gr.Button("Refresh")
            library_table = gr.Dataframe(
                headers=LIBRARY_COLUMNS,
                datatype=[
                    "str",
                    "str",
                    "str",
                    "number",
                    "number",
                    "number",
                    "str",
                    "str",
                    "str",
                ],
                value=handle_refresh,
                interactive=False,
                wrap=True,
            )
            refresh_btn.click(fn=handle_refresh, inputs=None, outputs=library_table)

    return app


def main() -> None:
    """Entry point for `python -m phil_mind_rag.app.ui`."""
    logging.basicConfig(level=logging.INFO)
    settings = _get_settings()
    app = create_app()
    app.launch(server_port=settings.gradio_server_port)


if __name__ == "__main__":
    main()
