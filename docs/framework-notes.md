# Framework Notes

This repo is easier to learn if each framework is tied to the exact job it does
here. These notes avoid generic descriptions and point to the implementation.

## RAG

RAG is not one library in this repo. It is the system shape:

```text
parse PDF -> chunk text -> embed chunks -> store vectors -> retrieve chunks -> generate grounded answer
```

The integration point is `phil_mind_rag/pipeline.py`. Read `RAGPipeline` as the
map of the basic system. Its public methods are useful learning checkpoints:

- `ingest()`: all ingestion steps in order.
- `retrieve()`: retrieval without generation, useful for evals and debugging.
- `answer_from_contexts()`: generation from already retrieved chunks.
- `query_with_sources()`: retrieval plus generation.

Tests: `tests/test_pipeline.py`.

## Chroma

Chroma is the local vector database. It stores chunk text, metadata, IDs, and
embeddings, then returns nearest chunks for a query embedding.

Implementation: `phil_mind_rag/retrieval/store.py`.

What to notice:

- `BaseVectorStore` defines the abstraction the rest of the app depends on.
- `ChromaVectorStore.add_chunks()` persists chunk records and embeddings.
- `ChromaVectorStore.query()` turns Chroma results into `RetrievalResult`
  objects.
- The app treats Chroma distance as a similarity-like score for display and
  ranking.

Tests: `tests/test_store.py`, `tests/test_retriever.py`.

## OpenAI

OpenAI appears in two separate roles:

- Embeddings in `pipeline.py`, via `OpenAI.embeddings.create()`.
- Chat generation in `generation/llm.py` and structured agent calls in
  `agents/_llm.py`.

The important design point is that model calls sit behind small wrappers or
callable boundaries, so tests can use fakes instead of network calls.

Tests: `tests/test_pipeline.py`, `tests/test_prompts.py`, `tests/test_agents.py`.

## Pydantic And Pydantic Settings

Pydantic Settings powers typed configuration in `phil_mind_rag/config.py`.
Regular Pydantic models define the contracts between agents in
`phil_mind_rag/agents/schema.py`.

What to notice:

- Settings keep model names, paths, chunk sizes, and limits out of global code.
- Agent schemas make LLM output testable because downstream code expects
  structured fields, not loose prose.
- Eval-generation schemas make generated eval rows auditable and chunk-pinned.

Tests: `tests/conftest.py`, `tests/test_agents.py`,
`tests/test_eval_generation.py`.

## LangGraph

LangGraph is the primary multi-agent orchestrator. It is used in
`phil_mind_rag/agents/graph.py`.

The graph is intentionally simple:

```text
START -> retrieve -> run_stances -> grounding -> END
```

State shape lives in `phil_mind_rag/agents/state.py`. Each node receives state
and returns updates. The stance node uses a thread pool to run the three stance
agents concurrently.

What LangGraph contributes here:

- explicit node/edge control flow
- a typed shared state shape
- a clear fan-out/fan-in orchestration model
- a concrete comparison point against plain Python orchestration

Tests: `tests/test_agents.py`.

## CrewAI

CrewAI is not the primary runtime path. It is an optional comparison adapter in
`phil_mind_rag/agents/crewai.py`.

Use it to study role/task orchestration as a contrast with LangGraph. Do not
treat it as a production dependency of the app unless a future issue
productizes the comparison runner.

Related tests: `tests/test_agents.py` covers import/lazy behavior around the
optional path.

## RAGAS

RAGAS is used as an evaluation wrapper for answer/context quality in
`phil_mind_rag/eval/evaluator.py`.

The wrapper focuses on four basic metrics:

- faithfulness
- answer relevancy
- context precision
- context recall

The deterministic evals are separate and often more useful for CI:

- `eval/retrieval_evaluator.py`: source/chunk precision and recall.
- `eval/agent_evaluator.py`: citation validity, stance coverage,
  disagreement coverage, adjudication coverage, and optional semantic support.

Tests: `tests/test_evaluator.py`, `tests/test_retrieval_evaluator.py`,
`tests/test_agent_evaluator.py`.

## Gradio

Gradio is only the UI layer. The current split is:

- `app/ui.py`: layout and event wiring.
- `app/callbacks.py`: button handlers and workflow calls.
- `app/formatting.py`: Markdown/table formatting.
- `app/state.py`: lazy app-level `Settings` and `RAGPipeline`.

Read this after the core RAG and agent flow. The UI is useful for seeing the
system, but it is not where the main RAG ideas live.

Tests: `tests/test_ui.py`.

## uv, Ruff, ty, pytest

These are the repo maintenance tools:

- `uv`: dependency and command runner.
- `ruff`: formatter and linter.
- `ty`: type checker.
- `pytest`: deterministic test runner.

Normal local gate:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```
