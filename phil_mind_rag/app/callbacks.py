"""Gradio callback handlers for the application."""

from __future__ import annotations

import logging
from pathlib import Path

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
from phil_mind_rag.app.formatting import (
    format_acquisition_results,
    format_argument_map,
    format_discovery_report,
    format_grounding,
    format_sources,
    format_stance_memo,
    format_synthesis,
    format_verified_claims,
)
from phil_mind_rag.app.state import get_pipeline, get_settings

logger = logging.getLogger(__name__)


def handle_refresh() -> list[list[object]]:
    records = get_pipeline().list_documents()
    return [
        [
            record.title,
            record.author,
            record.source,
            record.chunk_count,
            record.chunk_size,
            record.chunk_overlap,
            record.chunker.split(".")[-1],
            record.embedding_model,
            record.ingested_at[:19].replace("T", " "),
        ]
        for record in records
    ]


def handle_extract_metadata(file: str | None) -> tuple[str, str]:
    if file is None:
        return "", ""
    try:
        meta = get_pipeline().extract_metadata(Path(file))
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
    pipeline = get_pipeline()

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

    settings = get_settings()
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
            format_discovery_report(
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
    settings = get_settings()
    pipeline = get_pipeline()

    try:
        results = download_sources(
            jobs=jobs,
            output_dir=settings.source_download_dir,
            pipeline=pipeline,
        )
        return format_acquisition_results(results)
    except ValueError as exc:
        return f"Input error: {exc}"
    except Exception:
        logger.exception("Source acquisition failed")
        return "An unexpected error occurred during source acquisition."


def _build_download_jobs(
    recommendations: list[SourceRecommendation],
) -> list[DownloadJob]:
    return [
        DownloadJob(
            title=recommendation.title,
            source_url=recommendation.source_url,
            download_url=recommendation.download_url,
            doi=recommendation.doi,
            access_status=recommendation.access_status,
            access_note=recommendation.acquisition_note,
        )
        for recommendation in recommendations
    ]


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
        pipeline = get_pipeline()
        settings = get_settings()
        result: AnalysisResult = run_analysis(question, pipeline, settings)
        return (
            result.baseline_answer,
            format_stance_memo(result.materialist_memo),
            format_stance_memo(result.idealist_memo),
            format_stance_memo(result.dualist_memo),
            "\n\n".join(
                [
                    format_grounding(result.report),
                    format_verified_claims(result.verified_claims),
                ]
            ),
            "\n\n".join(
                [
                    format_synthesis(result.report),
                    format_argument_map(result.argument_map),
                ]
            ),
            format_sources(result.chunks),
        )
    except ValueError as exc:
        err = f"Input error: {exc}"
        return err, err, err, err, err, err, ""
    except Exception:
        logger.exception("Analysis failed")
        err = "An unexpected error occurred during analysis."
        return err, err, err, err, err, err, ""
