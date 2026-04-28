"""Fetch and ingest open-access corpus papers for the Philosophy of Mind RAG demo.

Only papers with access_status "open_author_html" or "repository_open" in
data/corpus_sources.json are processed. Paywalled and copyrighted sources are
skipped.

Copyright notes:
- Chalmers "Facing Up to the Problem of Consciousness" (1995): author-hosted
  HTML at consc.net. Source URL included in chunk metadata for attribution.
  For research/demo use only.
- Block "Troubles with Functionalism" (1978): University of Minnesota Digital
  Conservancy open repository. Repository terms apply — for research/demo use.

Usage:
    uv run python scripts/setup_corpus.py [--dry-run]
"""

from __future__ import annotations

import argparse
import html.parser
import json
import logging
import sys
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_JSON = PROJECT_ROOT / "data" / "corpus_sources.json"
OPEN_STATUSES = {"open_author_html", "repository_open"}
_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (academic research bot; phil-mind-rag)"}


# ---------------------------------------------------------------------------
# HTML source ingestion (Chalmers)
# ---------------------------------------------------------------------------


def _partition_html_to_parsed_document(url: str, source_name: str):  # noqa: ANN201
    """Fetch HTML from *url* and build a ParsedDocument using unstructured."""
    from unstructured.partition.html import partition_html

    from phil_mind_rag.ingestion.parser import ParsedDocument, Section

    logger.info("Fetching HTML from %s", url)
    req = urllib.request.Request(url, headers=_REQUEST_HEADERS)  # noqa: S310
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        html_bytes = resp.read()

    elements = partition_html(text=html_bytes.decode("utf-8", errors="replace"))

    sections: list[Section] = []
    current_title = "Introduction"
    current_texts: list[str] = []

    for el in elements:
        category = el.category  # type: ignore[attr-defined]
        text = str(el).strip()
        if not text:
            continue
        if category == "Title":
            if current_texts:
                sections.append(
                    Section(
                        title=current_title,
                        text="\n".join(current_texts),
                        metadata={
                            "source": source_name,
                            "section": current_title,
                            "source_url": url,
                        },
                    )
                )
                current_texts = []
            current_title = text
        else:
            current_texts.append(text)

    if current_texts:
        sections.append(
            Section(
                title=current_title,
                text="\n".join(current_texts),
                metadata={
                    "source": source_name,
                    "section": current_title,
                    "source_url": url,
                },
            )
        )

    if not sections:
        raise RuntimeError(f"No sections extracted from {url}")

    return ParsedDocument(
        source=source_name,
        sections=sections,
        metadata={"source_url": url},
    )


def _ingest_html_source(source_entry: dict, pipeline) -> int:  # noqa: ANN001
    source_name = source_entry["id"] + ".html"
    doc = _partition_html_to_parsed_document(source_entry["source_url"], source_name)

    chunks = pipeline.chunk(doc)
    if not chunks:
        raise RuntimeError("No chunks produced from HTML document")

    embeddings = pipeline.embed(chunks)
    pipeline.store(chunks, embeddings)
    # register() only uses the path's name and stem — file need not exist
    pipeline.register(
        Path(source_name),
        len(chunks),
        title=source_entry["title"],
        author=", ".join(source_entry["authors"]),
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# PDF source ingestion (Block)
# ---------------------------------------------------------------------------


class _PdfLinkFinder(html.parser.HTMLParser):
    """Collect href values that look like PDF download links."""

    def __init__(self) -> None:
        super().__init__()
        self.pdf_hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href") or ""
        if href.lower().endswith(".pdf") or "bitstream" in href.lower():
            self.pdf_hrefs.append(href)


def _find_pdf_url(page_url: str) -> str:
    """Follow redirects from *page_url* and return the first PDF download URL."""
    req = urllib.request.Request(page_url, headers=_REQUEST_HEADERS)  # noqa: S310
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        final_url = resp.url
        html_content = resp.read().decode("utf-8", errors="replace")

    finder = _PdfLinkFinder()
    finder.feed(html_content)

    if not finder.pdf_hrefs:
        raise RuntimeError(f"No PDF download link found on {final_url}")

    href = finder.pdf_hrefs[0]
    return href if href.startswith("http") else urljoin(final_url, href)


def _ingest_pdf_source(source_entry: dict, pipeline) -> int:  # noqa: ANN001
    page_url = source_entry["source_url"]
    logger.info("Locating PDF from repository page %s", page_url)
    pdf_url = _find_pdf_url(page_url)
    logger.info("Downloading PDF from %s", pdf_url)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pdf = Path(tmp_dir) / (source_entry["id"] + ".pdf")
        req = urllib.request.Request(pdf_url, headers=_REQUEST_HEADERS)  # noqa: S310
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            tmp_pdf.write_bytes(resp.read())
        logger.info("Downloaded %.1f KB", tmp_pdf.stat().st_size / 1024)

        return pipeline.ingest(
            tmp_pdf,
            title=source_entry["title"],
            author=", ".join(source_entry["authors"]),
        )


# ---------------------------------------------------------------------------
# Corpus JSON helpers
# ---------------------------------------------------------------------------


def _is_already_registered(source_entry: dict, pipeline) -> bool:
    source_name = (
        source_entry["id"] + ".html"
        if source_entry["access_status"] == "open_author_html"
        else source_entry["id"] + ".pdf"
    )
    return pipeline._registry.get(source_name) is not None  # noqa: SLF001


def _update_ingestion_status(corpus_json: Path, source_id: str, status: str) -> None:
    data = json.loads(corpus_json.read_text(encoding="utf-8"))
    for src in data["sources"]:
        if src["id"] == source_id:
            src["ingestion_status"] = status
            break
    corpus_json.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List open-access candidates without fetching or ingesting",
    )
    args = parser.parse_args()

    data = json.loads(CORPUS_JSON.read_text(encoding="utf-8"))
    candidates = [s for s in data["sources"] if s["access_status"] in OPEN_STATUSES]

    if not candidates:
        logger.info("No open-access candidates found in %s", CORPUS_JSON)
        return 0

    if args.dry_run:
        print("\nOpen-access candidates (dry run — no ingestion):")
        for src in candidates:
            status = src.get("ingestion_status", "unknown")
            print(f"  [{src['id']}] {src['title']}")
            print(f"    access:  {src['access_status']}")
            print(f"    url:     {src['source_url']}")
            print(f"    status:  {status}")
        print(f"\nRun without --dry-run to ingest {len(candidates)} source(s).")
        return 0

    sys.path.insert(0, str(PROJECT_ROOT))
    from phil_mind_rag.app.state import get_settings
    from phil_mind_rag.pipeline import RAGPipeline

    settings = get_settings()
    pipeline = RAGPipeline(settings)

    errors = 0
    for src in candidates:
        if _is_already_registered(src, pipeline):
            logger.info("Already registered: %s — skipping", src["id"])
            continue

        try:
            access = src["access_status"]
            if access == "open_author_html":
                n_chunks = _ingest_html_source(src, pipeline)
            elif access == "repository_open":
                n_chunks = _ingest_pdf_source(src, pipeline)
            else:
                logger.warning("Unhandled access_status '%s' for %s", access, src["id"])
                continue

            logger.info("Ingested '%s' — %d chunks", src["id"], n_chunks)
            _update_ingestion_status(CORPUS_JSON, src["id"], "ingested")

        except Exception:
            logger.exception("Failed to ingest '%s'", src["id"])
            errors += 1

    return errors


if __name__ == "__main__":
    sys.exit(main())
