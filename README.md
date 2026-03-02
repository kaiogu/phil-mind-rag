# Philosophy of Mind RAG

A Retrieval-Augmented Generation (RAG) system for querying academic papers in Philosophy of Mind. Upload PDFs, then ask natural-language questions — the system retrieves relevant passages and generates grounded answers via OpenAI.

## Features

- **PDF ingestion** — parse, chunk, embed, and index papers with real-time progress updates
- **Semantic search** — vector retrieval via ChromaDB using OpenAI embeddings
- **LLM-generated answers** — responses grounded in retrieved paper excerpts
- **Gradio web UI** — browser-based interface with Upload and Ask tabs
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

## Usage

1. **Upload tab** — select a PDF and click **Ingest**. Progress is shown step-by-step (parse → chunk → embed → store).
2. **Ask tab** — type a question and click **Ask** to get an answer sourced from the indexed papers.

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

## Tech Stack

- [Unstructured](https://github.com/Unstructured-IO/unstructured) — PDF parsing
- [ChromaDB](https://www.trychroma.com/) — local vector store
- [OpenAI](https://platform.openai.com/) — embeddings & chat completion
- [Gradio](https://gradio.app/) — web UI
- [RAGAS](https://docs.ragas.io/) — RAG evaluation metrics
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) — configuration
