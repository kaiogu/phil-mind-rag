# Philosophy of Mind RAG — Multi-Agent Design Brief

> Resolves KGU-93. This document is the authoritative design reference for v1.

---

## What this is

A RAG-backed multi-agent system over philosophy-of-mind sources. Three stance agents each represent a philosophical position and produce a structured memo in response to a user question. A grounding/adjudication agent compares the memos against retrieved evidence, penalises unsupported claims, and produces a final synthesis report.

This is not a debate simulator. The point is auditable, evidence-grounded philosophical reasoning — not stylised roleplay.

---

## Anti-prompt-theater constraint

The three stance agents must differ in at least one of:

- retrieval ranking / source emphasis
- output schema constraints
- critique responsibilities
- memory and coordination topology

**Current differentiation:** agents share the base retrieval set, but each
stance also receives stance-specific evidence from deterministic query
expansion and evidence-pack reranking. They still share the same tool surface
and memory model, so future work can deepen differentiation further through
claim-level tools, critic loops, and uncertainty calibration.

---

## Agent roster (v1)

### Stance agents

| Agent | Role |
|---|---|
| **Materialist** | Foregrounds physicalist, functionalist, and neuroscientific arguments. Builds the strongest charitable case for materialism. |
| **Idealist** | Foregrounds consciousness-first, anti-physicalist, and phenomenological arguments. Builds the strongest charitable case for idealism. |
| **Dualist** | Foregrounds property dualism and substance dualism arguments. Engages both the materialist and idealist critiques charitably. |

### Coordination agent

| Agent | Role |
|---|---|
| **Grounding / Adjudicator** | Makes no metaphysical commitments. Compares stance memos against retrieved sources. Flags unsupported leaps, equivocations, and source gaps. Produces the final synthesis report. |

### Deferred agents

Panpsychist, Illusionist, Neutral Monist, Historian/Taxonomy, Red-team Critic — all post-v1.

---

## Coordination pattern

Parallel memo generation followed by grounded adjudication. **No open-ended debate.**

```
User question
     │
     ▼
Retrieval layer  ──►  top-k chunks (shared across all stance agents)
     │
     ├──► Materialist agent  ──► StanceMemo
     ├──► Idealist agent     ──► StanceMemo      (parallel)
     └──► Dualist agent      ──► StanceMemo
                                      │
                                      ▼
                             Grounding agent  ──►  SynthesisReport
                                      │
                                      ▼
                                 Gradio UI
```

Stance agents run in parallel. The grounding agent runs only after all three memos are complete.

---

## Report schema

Formalised as Pydantic models. These are the contracts between agents — changing them mid-implementation is expensive.

```python
from pydantic import BaseModel

class EvidenceClaim(BaseModel):
    text: str
    citations: list[str]


class StanceMemo(BaseModel):
    stance: str                      # "materialist" | "idealist" | "dualist"
    thesis: str                      # one-sentence position statement
    supporting_claims: list[EvidenceClaim]
    rival_critiques: list[EvidenceClaim]
    confidence: float                # 0.0–1.0
    uncertainty_notes: str           # what the agent is unsure about


class Claim(BaseModel):
    text: str
    stance: str
    supported: bool
    citations: list[str]
    source_chunk_id: str | None      # None if unsupported
    note: str                        # grounding agent's annotation


class SynthesisReport(BaseModel):
    question: str
    areas_of_disagreement: list[str]
    strongest_arguments: dict[str, str]   # stance → best supported argument
    supported_claims: list[Claim]
    unsupported_claims: list[Claim]
    synthesis: str                        # adjudication / unresolved remainder
    decisive_chunks: list[str]
    source_chunks_used: list[str]         # all chunk IDs cited across all memos
```

The live analysis result also includes deterministic post-processing artifacts:
`VerifiedClaim` entries for citation audit and an `ArgumentMap` projection that
groups each stance's thesis, supporting claims, objections, adjudication notes,
and strongest argument. These live in `phil_mind_rag/agents/schema.py`.

---

## Retrieval strategy (v1)

- Shared corpus across all agents (existing ChromaDB index)
- Shared broad retrieval: top-k chunks returned for the original user query
- Stance-specific retrieval: each stance also retrieves with deterministic query
  expansion terms for materialist, idealist, and dualist evidence
- The system builds one global chunk-ID space across base and stance-specific
  retrieval results so `chunk_N` citations stay unambiguous during grounding
- Each stance receives the base chunks plus its own stance-specific chunks, then
  applies evidence-pack reranking before prompt construction
- Reranking uses explicit term sets for materialist, idealist, and dualist
  emphasis while preserving global `chunk_N` citation IDs for grounding
- Each generated `StanceMemo` records `evidence_chunk_ids`, the ordered chunk IDs
  shown to that stance agent

---

## Orchestration framework

**v1 primary: LangGraph** — explicit state machine, inspectable control flow, good observability.

**Implemented comparison baseline: plain Python orchestration** — same retrieve → parallel stances → grounding flow without a graph framework. This gives a concrete baseline for comparing boilerplate, debug ergonomics, and graph-framework value before adding another full agent framework.

**CrewAI comparison path:** optional task/role orchestration implementation,
tracked by KGU-114, lives in `phil_mind_rag/agents/crewai.py` and is lazy-loaded
so the primary app does not require CrewAI. Comparison dimensions:
1. Explicitness of state and control flow
2. Inspectability of multi-agent coordination
3. Tool use, memory, retry handling
4. Debug and eval ergonomics
5. Portfolio legibility vs production maintainability

Do not start with CrewAI. Start with architecture and evaluation criteria, implement LangGraph cleanly first.

---

## Corpus pipeline

The existing `RAGPipeline` (parse → chunk → embed → store) is the retrieval layer. It does not change for v1.

Corpus-building helpers extend it experimentally:

- Source providers discover candidate sources through OpenAlex, Semantic
  Scholar, and optional OpenAI web search.
- `corpus/discovery.py` ranks candidates into source recommendations.
- `corpus/acquisition.py` turns recommendations into lifecycle-tracked
  download jobs/results.
- `corpus/resolvers.py` resolves direct PDFs with direct URL → Unpaywall →
  arXiv → Semantic Scholar fallback order.
- Successful downloads can ingest via existing `RAGPipeline.ingest()`.

The acquisition workflow exists, but broad web acquisition is still
experimental product surface. Curated corpus work remains the stronger default
path for a useful portfolio artifact.

---

## Evaluation plan

### Eval generation (KGU-106)

An eval-generation agent reads source chunks and produces question sets pinned to source chunk IDs. The structured contracts live in `phil_mind_rag/agents/eval_generation.py`. Three question types:

| Type | What it tests |
|---|---|
| Factual retrieval | Does the system surface the right passages? |
| Stance-divergence | Do stance agents give genuinely different, position-consistent answers? |
| Grounding-fidelity probes | Are claims traceable to the retrieved source? |

All generated questions must reference a source chunk ID. No free-floating LLM-prior questions.

### Eval dimensions (v1)

| Dimension | Method |
|---|---|
| Grounding fidelity | Automatic: are cited chunk IDs real and do they support the claim? |
| Position fidelity | LLM-as-judge with cross-model evaluation (generate with one family, judge with another) |
| Disagreement quality | Are genuine cruxes surfaced, not generic summaries? |
| Adjudication quality | Does the grounding agent penalise unsupported claims consistently? |
| Framework quality | Manual: inspectability, debuggability, maintainability |

The deterministic scaffolding for these checks lives in
`phil_mind_rag/eval/agent_evaluator.py`. It verifies structural grounding,
stance coverage, disagreement coverage, and whether the grounding report
actually adjudicates stance-memo claims. It also exposes a separate semantic
support dimension: callers can pass a claim/chunk judge to score whether valid
citations actually support the claim, instead of treating valid chunk IDs as
semantic entailment.

Deterministic claim extraction lives in
`phil_mind_rag/agents/claim_extraction.py`. It converts structured stance memos,
grounding reports, and optional baseline answers into `AtomicClaim` records so
claim audit and verification can operate over one common shape.

Claim verification lives in `phil_mind_rag/agents/claim_verification.py`. It
checks citation structure, optionally repairs missing or invalid citations using
conservative lexical matching, and can call an optional semantic judge. Without a
semantic judge, structurally valid claims are labeled `ambiguous` rather than
treated as semantically proven. Live analysis runs this audit after baseline
answer generation in each orchestration path and displays the result in the UI
Grounding panel.

Argument-map construction lives in `phil_mind_rag/agents/argument_map.py`. It
does not call the model; it projects existing stance memos and grounding claims
into a readable disagreement map shown in the UI Synthesis panel.

Human spot-check required before any generated eval set is committed as ground truth.

---

## UI

Keep Gradio. Redesign the interface to expose the pipeline, not just the final answer:

- Stance memo panel: show all three memos side-by-side or in tabs
- Grounding panel: show which claims are supported vs flagged
- Synthesis panel: final adjudication
- Source panel: retrieved chunks (already exists)

The process is the product for a portfolio artifact. Showing memo-writing and adjudication is more compelling than a single answer box.

---

## MVP definition

| In scope | Out of scope |
|---|---|
| 3 stance agents + 1 grounding agent | Additional stances (panpsychist, illusionist, etc.) |
| Parallel memo generation | Open-ended debate / turn-based dialogue |
| Structured `StanceMemo` + `SynthesisReport` output | Free-form agent prose as the integration contract |
| LangGraph orchestration + optional CrewAI comparison path | Production CrewAI migration |
| Base retrieval + stance-specific evidence packs | Fully autonomous source acquisition during analysis |
| Redesigned Gradio UI | Native app / PWA |
| Grounding-fidelity eval | Full eval suite |
| Existing corpus (Nagel + manual uploads) | Fully automatic unsupervised corpus growth |

MVP ships when: a user can ask a philosophy-of-mind question, receive three grounded stance memos and a synthesis report, and verify each claim against the source chunks in the UI.

---

## Open questions (post-v1)

- Stance-specific retrieval: how much does it improve position fidelity over
  the original shared-retrieval baseline?
- Cross-paper reasoning: can the system surface disagreements across authors, not just within one paper?
- Memory: should stance agents accumulate positional memory across a session?
- CrewAI comparison: which framework wins on inspectability and eval ergonomics?
