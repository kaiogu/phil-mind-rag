# Learning Guide

This repo is both a portfolio project and a learning project. Read it as a RAG
system that grew an auditable multi-agent analysis layer, not as a finished
research product.

North star:

> Auditable philosophy-of-mind argument analysis over a curated corpus.

Start with the maps:

- `docs/architecture.md`: what exists and how modules connect.
- `docs/framework-notes.md`: what each framework contributes in this repo.
- `docs/design.md`: the agent design brief and product principles.

## Mental Model

The core RAG path is:

1. Validate and parse a PDF.
2. Split it into section-aware chunks.
3. Embed each chunk.
4. Store chunks, metadata, IDs, and embeddings in Chroma.
5. Embed a user question.
6. Retrieve nearest chunks.
7. Ask the LLM to answer from those chunks.

The multi-agent path adds:

1. Retrieve base chunks for the question.
2. Retrieve stance-specific chunks using deterministic query expansion.
3. Build evidence packs with stable global `chunk_N` IDs.
4. Generate materialist, idealist, and dualist structured memos.
5. Ground and synthesize those memos against retrieved evidence.
6. Extract and verify claims.
7. Project the result into an argument map.

The UI path is intentionally separate:

1. `app/ui.py` lays out Gradio components.
2. `app/callbacks.py` handles button actions.
3. `app/state.py` creates settings and the pipeline lazily.
4. `app/formatting.py` turns structured outputs into Markdown/table views.

## Reading Path

### 1. Project Story

Read:

- `README.md`
- `docs/architecture.md`
- `docs/framework-notes.md`
- `docs/design.md`

Learn:

- what is implemented
- what is experimental
- what is only planned
- why the project is more than a generic PDF chatbot

### 2. Configuration

Read:

- `phil_mind_rag/config.py`
- `tests/conftest.py`

Learn:

- how environment variables become typed settings
- where model names, paths, chunk sizes, and limits come from
- how tests avoid depending on your real `.env`

Focused check:

```bash
uv run ty check
```

### 3. Pipeline Wiring

Read:

- `phil_mind_rag/pipeline.py`
- `tests/test_pipeline.py`

Learn:

- why `RAGPipeline` is the integration point
- how parse/chunk/embed/store/retrieve/generate connect
- why `retrieve()` and `answer_from_contexts()` exist separately
- how evals and agents reuse the same pipeline pieces

Focused check:

```bash
uv run pytest tests/test_pipeline.py -v
```

### 4. Ingestion

Read:

- `phil_mind_rag/ingestion/parser.py`
- `phil_mind_rag/ingestion/chunker.py`
- `phil_mind_rag/ingestion/metadata_extractor.py`
- `phil_mind_rag/ingestion/registry.py`

Learn:

- how PDFs become parsed sections
- how section-aware chunks preserve local meaning
- why overlap protects context at chunk boundaries
- how the local document registry tracks ingested files

Focused checks:

```bash
uv run pytest tests/test_chunker.py tests/test_metadata_extractor.py tests/test_registry.py -v
```

### 5. Retrieval

Read:

- `phil_mind_rag/retrieval/store.py`
- `phil_mind_rag/retrieval/retriever.py`

Learn:

- what the vector-store abstraction hides
- how Chroma stores chunks and embeddings
- how query embeddings become nearest-neighbor results
- how retrieval output is shaped for prompts, agents, UI, and evals

Focused checks:

```bash
uv run pytest tests/test_store.py tests/test_retriever.py -v
```

### 6. Generation

Read:

- `phil_mind_rag/generation/prompts.py`
- `phil_mind_rag/generation/llm.py`
- `tests/test_prompts.py`

Learn:

- the grounded-answer prompt contract
- how chunks are injected into prompts
- where OpenAI chat calls are wrapped
- what tests fake instead of calling the network

Focused check:

```bash
uv run pytest tests/test_prompts.py -v
```

### 7. Security Boundaries

Read:

- `phil_mind_rag/security.py`
- `tests/test_security.py`

Learn:

- how query sanitization works
- how uploaded document extension and size are checked
- why this is a basic boundary, not a complete PDF/document security system

Focused check:

```bash
uv run pytest tests/test_security.py -v
```

### 8. Agent Contracts

Read:

- `phil_mind_rag/agents/schema.py`
- `tests/test_agents.py`
- `tests/test_agent_evaluator.py`

Learn:

- why structured outputs matter
- what `StanceMemo`, `SynthesisReport`, `VerifiedClaim`, and `ArgumentMap`
  represent
- how chunk citations are represented
- how deterministic evals inspect the agent outputs

### 9. Agent Evidence And Behavior

Read:

- `phil_mind_rag/agents/evidence.py`
- `phil_mind_rag/agents/prompts.py`
- `phil_mind_rag/agents/stance.py`
- `phil_mind_rag/agents/grounding.py`
- `phil_mind_rag/agents/_llm.py`

Learn:

- how stance-specific query expansion works
- how evidence packs preserve stable global chunk IDs
- how stance prompts differ
- how structured output is requested from OpenAI
- how grounding combines LLM judgment with deterministic citation checks

Focused check:

```bash
uv run pytest tests/test_agents.py -v
```

### 10. Orchestration

Read in order:

1. `phil_mind_rag/agents/plain.py`
2. `phil_mind_rag/agents/graph.py`
3. `phil_mind_rag/agents/crewai.py`

Learn:

- how the flow looks without a framework
- how LangGraph turns it into explicit state-machine nodes and edges
- why CrewAI is optional comparison code, not the primary app path

Current LangGraph flow:

```text
question -> retrieve evidence -> run stance memos -> grounding -> post-processing
```

Post-processing includes baseline answer generation, claim verification, and
argument-map construction.

### 11. Claims And Argument Maps

Read:

- `phil_mind_rag/agents/claim_extraction.py`
- `phil_mind_rag/agents/claim_verification.py`
- `phil_mind_rag/agents/argument_map.py`
- `tests/test_agent_report.py`
- `tests/test_agents.py`

Learn:

- how structured memos and reports become atomic claims
- why structurally valid citations are not automatically semantic proof
- how conservative citation repair works
- how the argument map is built deterministically from existing outputs

### 12. Corpus Discovery And Acquisition

Read:

- `phil_mind_rag/agents/source_search.py`
- `phil_mind_rag/corpus/discovery.py`
- `phil_mind_rag/corpus/acquisition.py`
- `phil_mind_rag/corpus/resolvers.py`
- `phil_mind_rag/corpus/formatting.py`
- `phil_mind_rag/agents/paper_tools.py`
- `tests/test_paper_tools.py`

Learn:

- how providers return candidate sources
- how candidates are ranked by an LLM into recommendations
- how acquisition jobs track lifecycle state
- how direct URL, Unpaywall, arXiv, and Semantic Scholar fallback resolution works
- why `agents/paper_tools.py` remains as a compatibility facade

Focused check:

```bash
uv run pytest tests/test_paper_tools.py -v
```

### 13. Evals

Read:

- `phil_mind_rag/eval/evaluator.py`
- `phil_mind_rag/eval/retrieval_evaluator.py`
- `phil_mind_rag/eval/agent_evaluator.py`
- `phil_mind_rag/eval/reporting.py`
- `phil_mind_rag/eval/agent_report.py`
- `phil_mind_rag/agents/eval_generation.py`
- `scripts/run_eval.py`
- `scripts/run_agent_eval.py`
- `data/eval_set.json`

Learn:

- what RAGAS measures
- why retrieval evals need expected source/chunk IDs
- why deterministic evals are useful in CI
- how generated eval questions should stay pinned to chunks
- why semantic support needs a judge beyond citation validity

Focused checks:

```bash
uv run pytest tests/test_evaluator.py tests/test_retrieval_evaluator.py tests/test_agent_evaluator.py tests/test_eval_generation.py -v
uv run python scripts/run_agent_eval.py
```

### 14. UI Last

Read:

- `phil_mind_rag/app/state.py`
- `phil_mind_rag/app/formatting.py`
- `phil_mind_rag/app/callbacks.py`
- `phil_mind_rag/app/ui.py`
- `tests/test_ui.py`

Learn:

- how Gradio events call into the system
- how the UI exposes ingestion, discovery, acquisition, analysis, and library
  state
- why formatting belongs outside callbacks
- why `ui.py` should stay mostly layout and event wiring

Focused check:

```bash
uv run pytest tests/test_ui.py -v
```

## Study Loop

Use this loop for each layer:

1. Read the implementation file.
2. Read the matching tests.
3. Run the focused tests.
4. Explain the module in your own words.
5. Write down one question before moving on.

If you get lost, return to:

- `phil_mind_rag/pipeline.py` for core RAG.
- `phil_mind_rag/agents/graph.py` for agent orchestration.
- `docs/architecture.md` for the whole-system map.

## Best First Week

1. Day 1: `config.py`, `pipeline.py`, and `docs/architecture.md`.
2. Day 2: ingestion and retrieval.
3. Day 3: generation and prompts.
4. Day 4: agent schemas, stance prompts, and grounding.
5. Day 5: LangGraph orchestration and plain Python comparison.
6. Day 6: evals.
7. Day 7: UI and corpus acquisition.
