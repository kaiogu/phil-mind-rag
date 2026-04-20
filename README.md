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

A learning and portfolio project building a RAG system that evolves into a multi-agent reasoning architecture over philosophy-of-mind papers. The goal is to understand — deeply, not just use — retrieval-augmented generation, vector search, LLM orchestration, and multi-agent coordination.

This README explains what the system does, why each piece works the way it does, what you should actually read and understand in the code, and where the project is going.

---

## What it does (current)

Upload a philosophy PDF. Ask a question. The system retrieves the most relevant passages and generates a grounded answer with source citations.

Under the hood: the PDF is parsed into structural sections, split into overlapping chunks, embedded into a vector space, and stored in ChromaDB. At query time, the question is embedded and compared against all stored chunks via cosine similarity. The top-k most similar chunks are passed to an LLM with a prompt that constrains it to answer only from the retrieved context.

## Multi-agent layer

Three stance agents — materialist, idealist, dualist — each read the same retrieved passages and write a structured memo arguing their philosophical position. A grounding agent reads all three memos, checks every claim against the source chunks, and produces a synthesis report. See [`docs/design.md`](docs/design.md) for the full architecture.

---

## Architecture

### Current: single-pipeline RAG

```
PDF
 │
 ▼
Parser          Unstructured — layout-aware, preserves document structure
 │
 ▼
Chunker         Section-aware: chunks never cross section boundaries
 │
 ▼
Embedder        OpenAI text-embedding-3-small — 1536-dim dense vectors
 │
 ▼
Vector Store    ChromaDB — HNSW index, cosine similarity
 │
 ▼  (at query time)
Retriever       Embed query → top-k nearest neighbours
 │
 ▼
LLM             GPT-4o-mini — constrained to retrieved context
 │
 ▼
Gradio UI
```

The key architectural decision: every layer is behind a plain Python interface (`BaseVectorStore`, `BaseRetriever`, `BaseLLM`). `pipeline.py` is the only file that touches all of them. Swapping ChromaDB for Pinecone, or OpenAI for Cohere, means changing one file.

### Multi-agent with LangGraph

```
User question
      │
      ▼
  Retrieval         shared top-k chunks
      │
      ├──► Materialist agent  ──► StanceMemo
      ├──► Idealist agent     ──► StanceMemo    (parallel)
      └──► Dualist agent      ──► StanceMemo
                                      │
                                      ▼
                             Grounding agent  ──►  SynthesisReport
                                      │
                                      ▼
                                  Gradio UI
```

LangGraph models this as a directed graph where nodes are agent calls and edges are control flow decisions. The parallel fan-out and fan-in pattern is a core LangGraph primitive.

A plain Python orchestration baseline implements the same flow without LangGraph, so framework value can be compared against a minimal control-flow implementation before adding CrewAI or another framework.

---

## The theory behind each layer

### Chunking

**Why it matters:** LLMs have context windows. You cannot pass a whole book. You must split text into retrievable units, retrieve only what's relevant, and pass that. The quality of chunking directly determines retrieval quality.

**What section-aware chunking does:** A naive chunker splits at fixed token counts and doesn't care about document structure — a chunk might contain the end of one section and the start of another. A section-aware chunker splits *within* sections, so each chunk is semantically coherent. For academic papers with distinct Introduction / Argument / Objection sections, this matters.

**Read:** `phil_mind_rag/ingestion/chunker.py` — especially how section boundaries are preserved and how `chunk_overlap` creates a sliding window so context isn't lost at chunk edges.

### Vector embeddings and similarity search

**What embeddings are:** A text embedding model maps a string to a point in high-dimensional space such that semantically similar strings land near each other. `text-embedding-3-small` produces 1536-dimensional vectors. "What is consciousness?" and "The hard problem of mind" will be close; "What is consciousness?" and "The GDP of France" will be far.

**Cosine similarity:** The distance metric used here. Two vectors are similar if they point in the same direction, regardless of magnitude. This is standard for text because embedding magnitude doesn't carry semantic meaning.

**HNSW index:** ChromaDB uses Hierarchical Navigable Small World graphs for approximate nearest-neighbour search. Exact search over millions of vectors is O(n) — too slow. HNSW is O(log n) with a small accuracy trade-off. You won't need to implement this, but understanding why approximate search exists matters.

**Read:** `phil_mind_rag/retrieval/store.py` — `ChromaVectorStore.query()` shows how a query embedding is sent to ChromaDB and how the returned distances (cosine distance → similarity score) are interpreted.

### Retrieval-Augmented Generation (RAG)

**The problem RAG solves:** LLMs hallucinate when asked about things outside their training data, or when precise sourcing matters. RAG constrains the LLM to reason from a specific retrieved context — it becomes a reading comprehension task, not a memory recall task.

**The retrieval-generation contract:** The prompt explicitly instructs the LLM to answer only from the provided passages and to say it doesn't know if the answer isn't there. `phil_mind_rag/generation/prompts.py` is this contract in code.

**Faithfulness vs relevancy:** A faithful answer uses only retrieved content. A relevant answer actually addresses the question. These are different failure modes. You can be faithful but irrelevant (retrieved the wrong chunks) or relevant but unfaithful (LLM added information not in the context).

**Read:** `phil_mind_rag/generation/prompts.py` and `phil_mind_rag/pipeline.py` (`query_with_sources`).

### Multi-agent coordination

**Why multiple agents?** A single LLM asked "compare materialism and idealism" will produce a balanced summary that represents no position strongly. Stance agents are constrained to argue *for* their position as charitably as possible — they produce stronger, more specific arguments. The grounding agent then has real disagreement to adjudicate, not mush.

**The anti-prompt-theater principle:** Three agents with different system prompts but identical retrieval, tools, and memory is mostly cosmetic. Real multi-agent differentiation means agents differ in what they retrieve, how they use tools, what schema they output, or how they critique. v1 uses the same retrieval but different output schemas and critique responsibilities — a minimal but real differentiation.

**LangGraph:** Models agent pipelines as directed graphs. Nodes are functions (agent calls, tool calls, conditional logic). Edges are transitions. State flows through the graph and is updated at each node. This makes the control flow explicit and inspectable rather than implicit in a chain of function calls.

**Read:** [`docs/design.md`](docs/design.md) for the full multi-agent design, then the LangGraph docs on [state machines](https://langchain-ai.github.io/langgraph/concepts/) before touching implementation.

### Evaluation (RAGAS)

**Why evals matter:** Without metrics, you can't tell if a change to chunking, retrieval, or prompting made things better or worse. "It seems better" is not a signal you can act on.

**RAGAS metrics:**
- **Faithfulness** — does the answer contain only claims supported by the retrieved context?
- **Answer relevancy** — does the answer actually address the question?
- **Context precision** — of the retrieved chunks, what fraction were actually useful?
- **Context recall** — of the relevant information that exists in the corpus, what fraction was retrieved?

**LLM-as-judge:** RAGAS uses an LLM to evaluate LLM outputs. This is circular if the same model family generates and judges, so cross-model evaluation (generate with one family, judge with another) is the mitigation.

**Read:** `phil_mind_rag/eval/evaluator.py` and `data/eval_set.json`.

---

## Code worth reading directly

These are the files where the real decisions live. Read them in this order:

| File | What to understand |
|---|---|
| `phil_mind_rag/config.py` | How Pydantic Settings works — env vars → typed config with zero boilerplate |
| `phil_mind_rag/ingestion/chunker.py` | Section-aware chunking logic, the sliding window, why `chunk_overlap` exists |
| `phil_mind_rag/retrieval/store.py` | `BaseVectorStore` ABC pattern, how ChromaDB stores and queries, cosine distance → similarity conversion |
| `phil_mind_rag/pipeline.py` | How all components are wired together; the only file with global knowledge of the system |
| `phil_mind_rag/generation/prompts.py` | The RAG prompt contract — how context is injected and what constraints are placed on the LLM |
| `phil_mind_rag/security.py` | Prompt injection detection and document validation — boundary security in an LLM app |
| `phil_mind_rag/eval/evaluator.py` | RAGAS integration — what metrics exist and how they're computed |
| `docs/design.md` | The multi-agent architecture design — read before touching agent code |

### What not to read first

- `phil_mind_rag/app/ui.py` — Gradio wiring, not where the interesting decisions are
- `phil_mind_rag/ingestion/parser.py` — thin wrapper over Unstructured, not much to learn here
- `tests/` — useful for understanding expected behaviour, but not architectural

---

## Tech stack

| Tool | Why |
|---|---|
| [Unstructured](https://github.com/Unstructured-IO/unstructured) | Layout-aware PDF parsing — preserves section structure that naive parsers discard |
| [ChromaDB](https://www.trychroma.com/) | Local persistent vector store with HNSW — no infrastructure needed for development |
| [OpenAI](https://platform.openai.com/) | `text-embedding-3-small` for embeddings, `gpt-4o-mini` for generation |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Explicit state-machine orchestration for multi-agent pipelines |
| [RAGAS](https://docs.ragas.io/) | RAG evaluation metrics — faithfulness, relevancy, context precision/recall |
| [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | Typed configuration from environment variables |
| [Gradio](https://gradio.app/) | Browser UI — low friction, good enough for portfolio and iteration |
| [uv](https://github.com/astral-sh/uv) | Fast Python package manager |
| [ruff](https://docs.astral.sh/ruff/) | Linter and formatter |
| [ty](https://github.com/astral-sh/ty) | Type checker (Astral, replaces mypy) |

---

## Roadmap

### v1 — current (single-pipeline RAG)
- [x] PDF ingestion: parse, chunk, embed, store
- [x] Semantic retrieval with ChromaDB
- [x] LLM-generated answers grounded in retrieved context
- [x] Source citation panel in UI
- [x] RAGAS evaluation harness
- [x] Security: prompt injection detection, file validation
- [x] Document registry

### v2 — multi-agent (in progress)
- [x] LangGraph orchestration layer
- [x] Plain Python orchestration baseline for framework comparison
- [x] Materialist, Idealist, Dualist stance agents
- [x] Grounding / adjudication agent
- [x] `StanceMemo` and `SynthesisReport` Pydantic schemas
- [x] Gradio UI exposing the memo + adjudication pipeline
- [x] Deterministic grounding-fidelity eval scaffolding
- [x] Semantic support eval hook beyond chunk-ID validity — KGU-113
- [ ] Live end-to-end validation with real corpus and paid model calls

### v3 — corpus pipeline
- [x] Source discovery helpers with OpenAlex and OpenAI web search — partial KGU-104
- [x] Source download and optional auto-ingestion helpers — partial KGU-105
- [ ] DOI / Unpaywall / arXiv / Semantic Scholar fallback acquisition order
- [ ] Registry-level ingestion status and dedup tracking
- [ ] Eval-generation agent (LLM-authored, corpus-pinned) — KGU-106

### v4 — framework comparison and deeper evals
- [ ] CrewAI reimplementation for framework comparison — KGU-114
- [ ] Stance-biased reranking
- [ ] Cross-model evaluation (generate with one family, judge with another)
- [ ] Full eval suite across all three question types

---

## Running it

```bash
# Install
uv sync

# Configure
cp .env.example .env
# add OPENAI_API_KEY to .env

# Run
python main.py         # http://localhost:7860

# Evaluate (after ingesting at least one doc)
python scripts/run_eval.py

# Live paid-provider smoke test for source discovery
uv run pytest --verbose -m paid_api tests/test_paid_api_smoke.py

# Lint / type-check / test
uv run ruff check .
uv run ty check
uv run pytest
```

---

## Commit convention

Conventional Commits with emoji prefix: `<emoji> <type>[(<scope>)]: <description>`

Install the hook: `git config core.hooksPath .githooks`

| Emoji | Type |
|---|---|
| ✨ | `feat` |
| 🐛 | `fix` |
| ♻️ | `refactor` |
| 🧪 | `test` |
| 📦 | `build` |
| 👷 | `ci` |
| 📝 | `docs` |
| 🔧 | `chore` |
| ⚡ | `perf` |
| ⏪ | `revert` |
