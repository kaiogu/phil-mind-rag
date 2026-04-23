"""Gradio frontend for the Philosophy of Mind multi-agent RAG system."""

from __future__ import annotations

import logging

import gradio as gr

from phil_mind_rag.app.callbacks import (
    handle_acquire_sources,
    handle_analysis,
    handle_discover_sources,
    handle_extract_metadata,
    handle_refresh,
    handle_upload,
)
from phil_mind_rag.app.formatting import LIBRARY_COLUMNS
from phil_mind_rag.app.state import get_settings


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
                value=[],
                interactive=False,
                wrap=True,
            )
            refresh_btn.click(fn=handle_refresh, inputs=None, outputs=library_table)

    return app


def main() -> None:
    """Entry point for `python -m phil_mind_rag.app.ui`."""
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    app = create_app()
    app.launch(server_port=settings.gradio_server_port)


if __name__ == "__main__":
    main()
