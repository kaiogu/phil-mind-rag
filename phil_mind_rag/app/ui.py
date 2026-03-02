"""Gradio frontend for the Philosophy of Mind RAG system."""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from phil_mind_rag.config import get_settings
from phil_mind_rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Lazy singleton so the pipeline is only built once.
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline  # noqa: PLW0603
    if _pipeline is None:
        _pipeline = RAGPipeline(get_settings())
    return _pipeline


# --- Callbacks ----------------------------------------------------------


def handle_upload(file: str | None) -> str:
    """Ingest an uploaded PDF."""
    if file is None:
        return "No file uploaded."

    path = Path(file)
    try:
        count = _get_pipeline().ingest(path)
        return f"Ingested **{path.name}** — {count} chunks indexed."
    except ValueError as exc:
        return f"Validation error: {exc}"
    except Exception:
        logger.exception("Ingestion failed for %s", path)
        return "An unexpected error occurred during ingestion."


def handle_query(question: str) -> str:
    """Answer a question against the indexed papers."""
    if not question.strip():
        return "Please enter a question."

    try:
        return _get_pipeline().query(question)
    except ValueError as exc:
        return f"Input error: {exc}"
    except Exception:
        logger.exception("Query failed")
        return "An unexpected error occurred while generating the answer."


# --- UI -----------------------------------------------------------------


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(title="Philosophy of Mind RAG") as app:
        gr.Markdown(
            "# Philosophy of Mind RAG\n"
            "Upload academic papers (PDF) and ask questions about them."
        )

        with gr.Tab("Upload"):
            file_input = gr.File(
                label="Upload a PDF paper",
                file_types=[".pdf"],
                type="filepath",
            )
            upload_btn = gr.Button("Ingest")
            upload_output = gr.Markdown(label="Status")

            upload_btn.click(
                fn=handle_upload,
                inputs=file_input,
                outputs=upload_output,
            )

        with gr.Tab("Ask"):
            question_input = gr.Textbox(
                label="Question",
                placeholder="e.g. What is the hard problem of consciousness?",
                lines=2,
            )
            ask_btn = gr.Button("Ask")
            answer_output = gr.Markdown(label="Answer")

            ask_btn.click(
                fn=handle_query,
                inputs=question_input,
                outputs=answer_output,
            )

    return app


def main() -> None:
    """Entry point for `python -m phil_mind_rag.app.ui`."""
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    app = create_app()
    app.launch(server_port=settings.gradio_server_port)


if __name__ == "__main__":
    main()
