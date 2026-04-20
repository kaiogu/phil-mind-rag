"""Gradio frontend for the Philosophy of Mind multi-agent RAG system."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
from openai import OpenAI

from phil_mind_rag.agents.graph import AnalysisResult, run_analysis
from phil_mind_rag.agents.paper_tools import (
    DownloadJob,
    discover_sources,
    download_sources,
)
from phil_mind_rag.agents.schema import SourceRecommendation
from phil_mind_rag.agents.source_search import (
    OpenAIWebSearchProvider,
    default_source_providers,
)
from phil_mind_rag.config import Settings, get_settings
from phil_mind_rag.pipeline import RAGPipeline

if TYPE_CHECKING:
    from phil_mind_rag.agents.schema import (  # SynthesisReport used in format helpers
        SourceDiscoveryReport,
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


def _format_discovery_report(
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


def _build_download_jobs(
    recommendations: list[SourceRecommendation],
) -> list[DownloadJob]:
    return [
        DownloadJob(
            title=recommendation.title,
            source_url=recommendation.source_url,
            download_url=recommendation.download_url,
            access_status=recommendation.access_status,
            access_note=recommendation.acquisition_note,
        )
        for recommendation in recommendations
    ]


def _format_acquisition_results(results: list) -> str:
    lines = ["**Acquisition Results:**"]
    for result in results:
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


def handle_discover_sources(
    field: str,
    question: str,
    search_query: str,
) -> tuple[str, list[dict[str, object]]]:
    if not field.strip() or not question.strip():
        return "Please enter both a field and a discovery question.", []

    settings = _get_settings()
    providers = default_source_providers(settings)
    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    provider_names = {provider.__class__.__name__ for provider in providers}

    try:
        report = discover_sources(
            field=field,
            question=question,
            providers=providers,
            client=client,
            model=settings.openai_chat_model,
            search_query=search_query or None,
        )
        return (
            _format_discovery_report(
                report,
                web_search_enabled=OpenAIWebSearchProvider.__name__ in provider_names,
            ),
            [recommendation.model_dump() for recommendation in report.recommendations],
        )
    except ValueError as exc:
        return f"Input error: {exc}", []
    except Exception:
        logger.exception("Source discovery failed")
        return "An unexpected error occurred during source discovery.", []


def handle_acquire_sources(
    recommendations_data: list[dict[str, object]] | None,
) -> str:
    if not recommendations_data:
        return "No discovered sources are available yet."

    recommendations = [
        SourceRecommendation.model_validate(item) for item in recommendations_data
    ]
    jobs = _build_download_jobs(recommendations)
    settings = _get_settings()
    pipeline = _get_pipeline()

    try:
        results = download_sources(
            jobs=jobs,
            output_dir=settings.source_download_dir,
            pipeline=pipeline,
        )
        return _format_acquisition_results(results)
    except ValueError as exc:
        return f"Input error: {exc}"
    except Exception:
        logger.exception("Source acquisition failed")
        return "An unexpected error occurred during source acquisition."


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

        with gr.Tab("Add Sources"):
            gr.Markdown("### Discover Sources")
            field_input = gr.Textbox(
                label="Field",
                placeholder="e.g. philosophy of mind",
                value="",
            )
            discovery_question_input = gr.Textbox(
                label="Discovery Question",
                placeholder=(
                    "e.g. What are the most important sources on the hard "
                    "problem of consciousness?"
                ),
                lines=2,
            )
            search_query_input = gr.Textbox(
                label="Search Query (optional)",
                placeholder="e.g. hard problem consciousness seminal papers books",
                value="",
            )
            discover_btn = gr.Button("Discover Sources", variant="primary")
            discovery_output = gr.Markdown(label="Discovery Results")
            discovered_sources_state = gr.State(value=[])
            acquire_btn = gr.Button("Acquire Open Sources")
            acquisition_output = gr.Markdown(label="Acquisition Status")

            discover_btn.click(
                fn=handle_discover_sources,
                inputs=[field_input, discovery_question_input, search_query_input],
                outputs=[discovery_output, discovered_sources_state],
            )
            acquire_btn.click(
                fn=handle_acquire_sources,
                inputs=discovered_sources_state,
                outputs=acquisition_output,
            )

            gr.Markdown("### Manual PDF Upload")
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
            upload_btn = gr.Button("Ingest PDF")
            upload_output = gr.Markdown(label="Upload Status")

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
