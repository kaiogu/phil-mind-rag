"""Gradio callback handlers for the application."""

from __future__ import annotations

import logging
from pathlib import Path

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
from phil_mind_rag.providers import generation_client, generation_models

logger = logging.getLogger(__name__)

type DeepResearchOutputs = tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
]


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
    client = generation_client(settings)
    provider_names = {provider.__class__.__name__ for provider in providers}

    try:
        report = discover_sources(
            field=field,
            question=question,
            providers=providers,
            client=client,
            model=generation_models(settings),
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


def handle_deep_research(
    field: str,
    question: str,
    search_query: str,
):
    if not field.strip() or not question.strip():
        message = "Please enter both a field and a research question."
        yield _deep_research_snapshot(
            status=_format_workflow_status(
                discovery=("failed", message),
                acquisition=("skipped", "Discovery did not run."),
                analysis=("skipped", "Analysis did not run."),
            ),
            discovery=message,
            acquisition="",
            provenance="",
        )
        return

    pipeline = get_pipeline()
    settings = get_settings()
    existing_sources_before = {record.source for record in pipeline.list_documents()}

    discovery_markdown = ""
    acquisition_markdown = ""
    provenance_markdown = ""

    yield _deep_research_snapshot(
        status=_format_workflow_status(
            discovery=("running", "Searching and ranking candidate sources."),
            acquisition=("pending", "Waiting for discovery results."),
            analysis=("pending", "Waiting for corpus preparation."),
        )
    )

    recommendations: list[SourceRecommendation] = []
    discovery_error: str | None = None
    try:
        providers = default_source_providers(settings)
        client = generation_client(settings)
        provider_names = {provider.__class__.__name__ for provider in providers}
        report = discover_sources(
            field=field,
            question=question,
            providers=providers,
            client=client,
            model=generation_models(settings),
            search_query=search_query or None,
        )
        recommendations = report.recommendations
        discovery_markdown = format_discovery_report(
            report,
            web_search_enabled=OpenAIWebSearchProvider.__name__ in provider_names,
        )
        discovery_status = (
            "completed",
            f"Ranked {len(recommendations)} candidate sources.",
        )
    except ValueError as exc:
        discovery_error = f"Input error: {exc}"
        discovery_markdown = discovery_error
        discovery_status = ("failed", discovery_error)
    except Exception:
        logger.exception("Deep research discovery failed")
        discovery_error = "An unexpected error occurred during source discovery."
        discovery_markdown = discovery_error
        discovery_status = ("failed", discovery_error)

    acquisition_results = []
    new_sources: set[str] = set()
    if recommendations:
        yield _deep_research_snapshot(
            status=_format_workflow_status(
                discovery=discovery_status,
                acquisition=("running", "Downloading and ingesting open sources."),
                analysis=("pending", "Waiting for acquisition to finish."),
            ),
            discovery=discovery_markdown,
        )

        try:
            acquisition_results = download_sources(
                jobs=_build_download_jobs(recommendations),
                output_dir=settings.source_download_dir,
                pipeline=pipeline,
            )
            acquisition_markdown = format_acquisition_results(acquisition_results)
            new_sources = {
                result.path.name
                for result in acquisition_results
                if result.success and result.status == "downloaded" and result.path
            }
            downloaded_count = sum(
                1 for result in acquisition_results if result.status == "downloaded"
            )
            skipped_count = sum(1 for result in acquisition_results if result.skipped)
            failed_count = sum(
                1 for result in acquisition_results if result.status == "failed"
            )
            acquisition_status = (
                "completed",
                " ".join(
                    [
                        f"Downloaded {downloaded_count} source(s).",
                        f"Skipped {skipped_count}.",
                        f"Failed {failed_count}.",
                    ]
                ),
            )
        except ValueError as exc:
            acquisition_markdown = f"Input error: {exc}"
            acquisition_status = ("failed", acquisition_markdown)
        except Exception:
            logger.exception("Deep research acquisition failed")
            acquisition_markdown = (
                "An unexpected error occurred during source acquisition."
            )
            acquisition_status = ("failed", acquisition_markdown)
    else:
        if discovery_error is None:
            acquisition_markdown = "No recommended sources were available to acquire."
            acquisition_reason = "Discovery returned no recommended sources."
        else:
            acquisition_markdown = (
                "Source acquisition was skipped because discovery failed."
            )
            acquisition_reason = "Discovery failed, so no acquisition jobs were run."
        acquisition_status = ("skipped", acquisition_reason)

    documents_after_acquisition = pipeline.list_documents()
    available_sources = {record.source for record in documents_after_acquisition}
    if not available_sources:
        provenance_markdown = "No library sources are available for analysis."
        yield _deep_research_snapshot(
            status=_format_workflow_status(
                discovery=discovery_status,
                acquisition=acquisition_status,
                analysis=(
                    "skipped",
                    "The library is empty, so analysis could not run.",
                ),
            ),
            discovery=discovery_markdown,
            acquisition=acquisition_markdown,
            provenance=provenance_markdown,
        )
        return

    yield _deep_research_snapshot(
        status=_format_workflow_status(
            discovery=discovery_status,
            acquisition=acquisition_status,
            analysis=("running", "Running grounded multi-agent analysis."),
        ),
        discovery=discovery_markdown,
        acquisition=acquisition_markdown,
    )

    try:
        result = run_analysis(question, pipeline, settings)
        provenance_markdown = _format_source_usage(
            chunks=result.chunks,
            new_sources=new_sources,
            existing_sources=existing_sources_before,
        )
        analysis_status = (
            "completed",
            f"Retrieved {len(result.chunks)} chunk(s) for the final answer.",
        )
        yield _deep_research_snapshot(
            status=_format_workflow_status(
                discovery=discovery_status,
                acquisition=acquisition_status,
                analysis=analysis_status,
            ),
            discovery=discovery_markdown,
            acquisition=acquisition_markdown,
            provenance=provenance_markdown,
            baseline=result.baseline_answer,
            materialist=format_stance_memo(result.materialist_memo),
            idealist=format_stance_memo(result.idealist_memo),
            dualist=format_stance_memo(result.dualist_memo),
            grounding="\n\n".join(
                [
                    format_grounding(result.report),
                    format_verified_claims(result.verified_claims),
                ]
            ),
            synthesis="\n\n".join(
                [
                    format_synthesis(result.report),
                    format_argument_map(result.argument_map),
                ]
            ),
            sources=format_sources(result.chunks),
        )
    except ValueError as exc:
        error = f"Input error: {exc}"
        yield _deep_research_snapshot(
            status=_format_workflow_status(
                discovery=discovery_status,
                acquisition=acquisition_status,
                analysis=("failed", error),
            ),
            discovery=discovery_markdown,
            acquisition=acquisition_markdown,
            provenance=provenance_markdown,
            baseline=error,
            materialist=error,
            idealist=error,
            dualist=error,
            grounding=error,
            synthesis=error,
        )
    except Exception:
        logger.exception("Deep research analysis failed")
        error = "An unexpected error occurred during analysis."
        yield _deep_research_snapshot(
            status=_format_workflow_status(
                discovery=discovery_status,
                acquisition=acquisition_status,
                analysis=("failed", error),
            ),
            discovery=discovery_markdown,
            acquisition=acquisition_markdown,
            provenance=provenance_markdown,
            baseline=error,
            materialist=error,
            idealist=error,
            dualist=error,
            grounding=error,
            synthesis=error,
        )


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


def _deep_research_snapshot(
    *,
    status: str = "",
    discovery: str = "",
    acquisition: str = "",
    provenance: str = "",
    baseline: str = "",
    materialist: str = "",
    idealist: str = "",
    dualist: str = "",
    grounding: str = "",
    synthesis: str = "",
    sources: str = "",
) -> DeepResearchOutputs:
    return (
        status,
        discovery,
        acquisition,
        provenance,
        baseline,
        materialist,
        idealist,
        dualist,
        grounding,
        synthesis,
        sources,
    )


def _format_workflow_status(
    *,
    discovery: tuple[str, str],
    acquisition: tuple[str, str],
    analysis: tuple[str, str],
) -> str:
    lines = ["**Workflow Status**"]
    lines.append(_format_workflow_status_line("Discover sources", *discovery))
    lines.append(_format_workflow_status_line("Acquire and ingest", *acquisition))
    lines.append(_format_workflow_status_line("Grounded analysis", *analysis))
    return "\n".join(lines)


def _format_workflow_status_line(name: str, state: str, detail: str) -> str:
    return f"- **{name}:** {state.replace('_', ' ')}. {detail}"


def _format_source_usage(
    *,
    chunks: list,
    new_sources: set[str],
    existing_sources: set[str],
) -> str:
    if not chunks:
        return "_No sources were retrieved for the final answer._"

    retrieved_sources = {
        str(chunk.metadata.get("source", "?"))
        for chunk in chunks
        if chunk.metadata.get("source")
    }
    new_used = sorted(source for source in retrieved_sources if source in new_sources)
    existing_used = sorted(
        source for source in retrieved_sources if source in existing_sources
    )
    other_used = sorted(retrieved_sources - set(new_used) - set(existing_used))

    if new_used and existing_used:
        scope = "Both newly added sources and existing library sources were used."
    elif new_used:
        scope = "Only newly added sources were used."
    elif existing_used:
        scope = "Only existing library sources were used."
    else:
        scope = "Retrieved sources could not be matched to this run's library history."

    lines = ["**Corpus Usage**", scope]
    if new_used:
        lines.append(f"Newly added: {', '.join(new_used)}")
    if existing_used:
        lines.append(f"Existing library: {', '.join(existing_used)}")
    if other_used:
        lines.append(f"Unclassified: {', '.join(other_used)}")
    return "\n".join(lines)
