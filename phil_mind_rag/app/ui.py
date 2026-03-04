"""Gradio frontend for the Philosophy of Mind RAG system."""

from __future__ import annotations

import logging
from collections.abc import Generator
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

LIBRARY_COLUMNS = [
    "Title", "Author", "Source File", "Chunks",
    "Chunk Size", "Chunk Overlap", "Chunker", "Embedding Model", "Ingested At",
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
            r.chunker.split(".")[-1],                    # class name only
            r.embedding_model,
            r.ingested_at[:19].replace("T", " "),        # "2026-03-02 14:05:00"
        ]
        for r in records
    ]


def handle_extract_metadata(file: str | None) -> tuple[str, str]:
    """Return (title, author) extracted from the uploaded PDF."""
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
) -> Generator[str, None, None]:
    """Ingest an uploaded PDF, yielding status updates to the UI."""
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


def handle_query(question: str) -> tuple[str, str]:
    """Answer a question and return (answer_markdown, sources_markdown)."""
    if not question.strip():
        return "Please enter a question.", ""

    try:
        answer, sources = _get_pipeline().query_with_sources(question)
        sources_md = _format_sources(sources)
        return answer, sources_md
    except ValueError as exc:
        return f"Input error: {exc}", ""
    except Exception:
        logger.exception("Query failed")
        return "An unexpected error occurred while generating the answer.", ""


def _format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = ["**Retrieved sources:**\n"]
    for i, r in enumerate(sources, 1):
        src = r.metadata.get("source", "?")
        section = r.metadata.get("section", "?")
        score = r.score
        snippet = r.text[:200].replace("\n", " ")
        lines.append(
            f"{i}. **{src} — {section}** (score: {score:.3f})\n"
            f"   > {snippet}…"
        )
    return "\n\n".join(lines)


# --- UI -----------------------------------------------------------------


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(title="Philosophy of Mind RAG") as app:
        gr.Markdown(
            "# Philosophy of Mind RAG\n"
            "Upload academic papers (PDF) and ask questions about them."
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

        with gr.Tab("Ask"):
            question_input = gr.Textbox(
                label="Question",
                placeholder="e.g. What is the hard problem of consciousness?",
                lines=2,
            )
            ask_btn = gr.Button("Ask")
            answer_output = gr.Markdown(label="Answer")
            sources_output = gr.Markdown(label="Sources")

            ask_btn.click(
                fn=handle_query,
                inputs=question_input,
                outputs=[answer_output, sources_output],
            )

        with gr.Tab("Library"):
            gr.Markdown("### Ingested Papers")
            refresh_btn = gr.Button("Refresh")
            library_table = gr.Dataframe(
                headers=LIBRARY_COLUMNS,
                datatype=[
                    "str", "str", "str",
                    "number", "number", "number",
                    "str", "str", "str",
                ],
                value=handle_refresh,   # called on page load
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
