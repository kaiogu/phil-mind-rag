---
title: Philosophy of Mind RAG
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Philosophy of Mind RAG

A Retrieval-Augmented Generation (RAG) system for querying academic papers in Philosophy of Mind. Upload PDFs, then ask natural-language questions — the system retrieves relevant passages and generates grounded answers via OpenAI.

## Evaluation Results

Evaluated on 12 questions about Thomas Nagel's *What Is It Like to Be a Bat?* using RAGAS metrics.

| Metric | Score |
|---|---|
| Faithfulness | — |
| Answer Relevancy | — |
| Context Precision | — |
| Context Recall | — |

> **Note:** Run `python scripts/run_eval.py` after ingesting the corpus to populate these numbers.

## Features

- **PDF ingestion** — parse, chunk, embed, and index papers with real-time progress updates
- **Semantic search** — vector retrieval via ChromaDB using OpenAI embeddings
- **LLM-generated answers** — responses grounded in retrieved paper excerpts, with citations
- **Source panel** — displays the retrieved chunks that informed each answer
- **Gradio web UI** — browser-based interface with Upload, Ask, and Library tabs
- **Security** — prompt-injection detection, file-type/size validation
- **Evaluation** — RAGAS metrics (faithfulness, answer relevancy, context precision/recall)

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- An OpenAI API key

### Install

```bash
uv sync
```

### Configure

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-...
```

### Run

```bash
python main.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

### Run Evaluation

After ingesting at least one document:

```bash
python scripts/run_eval.py
```

## Usage

1. **Upload tab** — select a PDF and click **Ingest**. Progress is shown step-by-step (parse → chunk → embed → store).
2. **Ask tab** — type a question and click **Ask** to get an answer sourced from the indexed papers. Retrieved source passages appear below the answer.
3. **Library tab** — view all ingested documents.

## Configuration

All settings are loaded from environment variables (or `.env`). Defaults:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(required)_ | OpenAI API key |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-5-mini` | Chat model for generation |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | `phil_mind_papers` | ChromaDB collection |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Token overlap between chunks |
| `MAX_DOCUMENT_SIZE_MB` | `50` | Maximum PDF size |
| `GRADIO_SERVER_PORT` | `7860` | UI port |

## Project Structure

```
phil_mind_rag/
├── app/
│   └── ui.py           # Gradio frontend
├── ingestion/
│   ├── parser.py       # PDF → structured sections (Unstructured)
│   └── chunker.py      # Section-aware text chunking
├── retrieval/
│   ├── store.py        # ChromaDB vector store
│   └── retriever.py    # Top-k semantic retrieval
├── generation/
│   ├── llm.py          # OpenAI chat completion
│   └── prompts.py      # RAG prompt template
├── eval/
│   └── evaluator.py    # RAGAS evaluation harness
├── pipeline.py         # Orchestrates all components
├── config.py           # Pydantic settings
└── security.py         # Input sanitisation & validation
data/
└── eval_set.json       # 12-question eval set (Nagel bat paper)
scripts/
└── run_eval.py         # CLI evaluation runner
```

## Development

```bash
# Lint
uv run ruff check .

# Type-check
uv run mypy .

# Tests
uv run pytest
```

## Contributing

This project follows [Conventional Commits](https://www.conventionalcommits.org/) with an emoji prefix.

**Format:** `<emoji> <type>[(<scope>)]: <short description>`

| Emoji | Type | Use for |
|---|---|---|
| ✨ | `feat` | New feature |
| 🐛 | `fix` | Bug fix |
| 📝 | `docs` | Documentation |
| 🎨 | `style` | Formatting, no logic change |
| ♻️ | `refactor` | Code restructure, no behaviour change |
| 🧪 | `test` | Tests |
| 📦 | `build` | Dependencies, build system |
| 👷 | `ci` | CI/CD pipelines |
| 🔧 | `chore` | Maintenance, tooling |
| ⚡ | `perf` | Performance improvement |
| ⏪ | `revert` | Revert a previous commit |

**Examples:**

```
✨ feat(ui): add retrieved sources panel below answer
🐛 fix(eval): update evaluator to RAGAS 0.4.x API
📝 docs: add eval results table to README
📦 build: add Dockerfile for HuggingFace Spaces
```

### Install the commit-msg hook

```bash
git config core.hooksPath .githooks
```

The hook rejects commits that don't match the format above. A GitHub Actions workflow runs the same check on every push and pull request.

## Tech Stack

- [Unstructured](https://github.com/Unstructured-IO/unstructured) — PDF parsing
- [ChromaDB](https://www.trychroma.com/) — local vector store
- [OpenAI](https://platform.openai.com/) — embeddings & chat completion
- [Gradio](https://gradio.app/) — web UI
- [RAGAS](https://docs.ragas.io/) — RAG evaluation metrics
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) — configuration

## Deployment

The app ships as a Docker image for [HuggingFace Spaces](https://huggingface.co/spaces) (Docker SDK).

Set `OPENAI_API_KEY` as a Space secret in the HF Space settings. The `Dockerfile` installs all system dependencies (poppler, tesseract) required by Unstructured.

---

## Design Notes

### What I built

A production-quality RAG system for philosophy of mind papers. The architecture is component-based: each layer (parse → chunk → embed → store → retrieve → generate) is behind a plain Python interface, making it easy to swap backends without touching the rest of the code.

### Key decisions

**Unstructured for parsing.** Academic PDFs have complex layouts (multi-column, footnotes, headers). Unstructured's layout-aware parser extracts sections with structure preserved, which improves chunk coherence. The trade-off is system dependencies (poppler, tesseract), addressed by Docker deployment.

**Section-aware chunking.** Chunks stay within section boundaries rather than splitting blindly at token limits. This means retrieved chunks are semantically coherent — a chunk from the "Introduction" is less likely to mix arguments from different sections.

**RAGAS evaluation.** Chose RAGAS because it measures what matters: does the answer faithfully reflect the retrieved context (faithfulness), and did the retrieval surface the right passages (context precision/recall)? A 12-question eval set was hand-authored against Nagel's *What Is It Like to Be a Bat?* to give reproducible, inspectable results.

**Source citations in UI.** The retrieved chunks that ground each answer are displayed separately from the answer text. This makes the RAG mechanism visible during demos and lets users verify claims against the source.

### What I'd improve next

- **Hybrid retrieval**: add BM25 alongside dense vectors for better coverage on exact-match queries (author names, technical terms).
- **Re-ranking**: a cross-encoder re-ranker after the initial retrieval step would improve precision.
- **Streaming answers**: stream the LLM response token-by-token via the OpenAI streaming API for better perceived latency.
- **Multi-document eval**: expand the eval set to cover more papers and add cross-paper reasoning questions.
