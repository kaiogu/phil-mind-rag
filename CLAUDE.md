# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the app
python main.py          # Launches Gradio at http://localhost:7860

# Ingest open-access corpus papers (Chalmers, Block)
uv run python scripts/setup_corpus.py          # fetch and ingest
uv run python scripts/setup_corpus.py --dry-run  # list candidates only

# Run evaluation (requires at least one ingested doc)
python scripts/run_eval.py

# Lint
uv run ruff check .

# Type-check
uv run ty check

# Tests (all)
uv run pytest

# Single test file
uv run pytest tests/test_pipeline.py -v

# Single test by name
uv run pytest tests/test_security.py::test_sanitise_query_too_long -v
```

## Architecture

The system is a five-layer RAG pipeline, all wired together **only** in `phil_mind_rag/pipeline.py`. Every layer sits behind an interface so backends can be swapped without touching the rest of the code.

```
PDF → parse → chunk → embed → store → retrieve → generate → Gradio UI
```

### Layers

| Module | Purpose |
|---|---|
| `ingestion/parser.py` | `UnstructuredPDFParser` — layout-aware PDF → `ParsedDocument` (list of `Section`s) |
| `ingestion/chunker.py` | `SectionAwareChunker` — splits sections into `Chunk`s, never crossing section boundaries |
| `ingestion/metadata_extractor.py` | `MetadataExtractor` — extracts title/author from PDF metadata or via LLM |
| `ingestion/registry.py` | `DocumentRegistry` — JSON file (`data/registry.json`) tracking ingested docs |
| `retrieval/store.py` | `BaseVectorStore` ABC + `ChromaVectorStore` impl; cosine similarity via ChromaDB |
| `retrieval/retriever.py` | `VectorRetriever` — calls embed_fn, queries the store, returns `RetrievalResult` list |
| `generation/llm.py` | `OpenAILLM` — chat completion wrapper |
| `generation/prompts.py` | `RAGPrompt.build()` — assembles the context + question prompt |
| `eval/evaluator.py` | RAGAS evaluation harness (faithfulness, relevancy, precision, recall) |
| `pipeline.py` | `RAGPipeline` — the only place components are instantiated and wired |
| `app/ui.py` | Gradio frontend: Upload, Ask, Library tabs |
| `security.py` | `sanitise_query()` (prompt-injection detection) + `validate_document()` (type/size checks) |
| `config.py` | `Settings` (Pydantic) — all config from env vars / `.env` |

### Key data types

- `Section(title, text, metadata)` — one structural section of a parsed PDF
- `ParsedDocument(source, sections)` — full parsed document
- `Chunk(text, metadata)` — retrieval unit; metadata carries `source` and `section`
- `RetrievalResult(text, score, metadata)` — a matched chunk with cosine similarity score

### Data directory

- `data/chroma/` — ChromaDB persisted vector index (git-ignored)
- `data/registry.json` — tracks ingested docs (untracked, auto-created)
- `data/eval_set.json` — 12 hand-authored Q&A pairs for RAGAS evaluation (Nagel bat paper)
- `data/raw/` and `data/processed/` — PDF working directories

## Configuration

All settings are in `config.py` / `.env`. Required: `OPENAI_API_KEY`. See `.env.example` for the full list. The `Settings` object is passed into `RAGPipeline`; components are never instantiated with hardcoded config.

## Commit convention

Conventional Commits with an emoji prefix: `<emoji> <type>[(<scope>)]: <description>`. The commit-msg hook in `.githooks/` enforces this. Activate with:

```bash
git config core.hooksPath .githooks
```

Common types: `✨ feat`, `🐛 fix`, `♻️ refactor`, `🧪 test`, `📦 build`, `👷 ci`, `📝 docs`.

## Testing

Tests live in `tests/`. Shared fixtures (sample `Section`, `ParsedDocument`, `Chunk`, `RetrievalResult`) are in `tests/conftest.py`. All tests are pure-unit (no ChromaDB, no OpenAI calls) — external dependencies are replaced with fakes or simple stubs in individual test files.
