# Curated Corpus

The current product direction is an auditable philosophy-of-mind research
assistant over a curated corpus. The curated corpus is tracked as metadata first,
not as committed PDFs.

## Committed vs Local Data

Committed:

- `data/corpus_sources.json`: canonical source metadata and access notes.
- `data/eval_set.json`: small eval set, currently Nagel-focused.

Local runtime state, ignored by git:

- `data/raw/`: manually added or downloaded PDFs and source files.
- `data/processed/`: future parsed/chunked intermediate artifacts.
- `data/chroma/`: Chroma vector database.
- `data/registry.json`: local ingestion registry.
- `eval_runs/`: generated eval reports.

Do not commit copyrighted PDFs, parsed full text, Chroma databases, or registry
files from your local machine.

## Metadata Schema

`data/corpus_sources.json` has this shape:

```json
{
  "schema_version": 1,
  "description": "...",
  "sources": [
    {
      "id": "chalmers_facing_up",
      "title": "Facing Up to the Problem of Consciousness",
      "authors": ["David J. Chalmers"],
      "year": 1995,
      "source_type": "journal_article",
      "venue": "Journal of Consciousness Studies",
      "citation": "...",
      "doi": null,
      "source_url": "https://consc.net/papers/facing.html",
      "access_status": "open_author_html",
      "license_status": "author_web_copy",
      "local_filename": null,
      "ingestion_status": "candidate",
      "stance_tags": ["dualist", "anti_reductionist"],
      "school_tags": ["property_dualism"],
      "eval_roles": ["cross_paper_comparison"],
      "priority": 1,
      "acquisition_note": "..."
    }
  ]
}
```

Field meanings:

| Field | Meaning |
|---|---|
| `id` | Stable source ID used by docs, evals, and future corpus tools. |
| `source_type` | Bibliographic form, such as `journal_article`, `book_chapter`, or `book`. |
| `access_status` | Practical access category for ingestion planning. |
| `license_status` | Legal/copyright caution independent of technical availability. |
| `local_filename` | Expected local artifact name only when a lawful local file exists. |
| `ingestion_status` | Current repo status, such as `candidate` or `local_sample`. |
| `stance_tags` | Philosophical role tags used for source balancing and future retrieval diagnostics. |
| `school_tags` | Broader school/tradition tags. |
| `eval_roles` | Eval slices this source should support. |
| `priority` | Lower number means more central to the starter corpus. |
| `acquisition_note` | Human-readable legal/access instructions. |

## Starter Corpus

The initial metadata set covers:

- Nagel: subjective character and reductionism challenge.
- Chalmers: hard problem and nonreductive explanation.
- Jackson: knowledge argument and qualia.
- Churchland: eliminative materialism.
- Dennett: anti-qualia/illusionist-adjacent critique.
- Block: anti-functionalist qualia arguments.
- Putnam: functionalism and multiple realizability.
- Searle: Chinese room and anti-computationalism.
- Tye: representationalism.
- Crick/Koch: neuroscience anchor for consciousness research.

This is intentionally balanced across anti-reductionist, dualist,
functionalist, materialist, representationalist, and neuroscience sources.

## Rebuild Local Ingestion State

From a clean checkout:

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Configure OpenAI:

   ```bash
   cp .env.example .env
   # set OPENAI_API_KEY in .env
   ```

3. Place lawful source files under `data/raw/`.

   Recommended naming convention: use the source ID from
   `data/corpus_sources.json`, for example `data/raw/nagel_bat.pdf`.

4. Start the app and ingest through the UI:

   ```bash
   python main.py
   ```

   Use the Add Sources tab for manual upload, or use Discover/Acquire for
   experimental open-source acquisition.

5. Confirm local registry state:

   ```bash
   python main.py
   # open the Library tab and click Refresh
   ```

The registry stores local ingestion facts: source filename, title, author, chunk
count, chunker settings, embedding model, and ingestion timestamp. It is a
runtime index, not canonical corpus metadata.

## Legal Safety Rules

- Metadata is safe to commit.
- Full-text files are local unless license and repository policy are explicitly
  safe for redistribution.
- Paywalled or copyrighted sources can be included as metadata candidates, but
  the repo must not include their PDFs.
- Repository-hosted files may still have terms of use or copyright limits; record
  that in `license_status` and `acquisition_note`.
- Prefer author-hosted HTML or clearly licensed open copies for automated
  ingestion.

## How This Supports Future Evals

Future cross-paper eval rows should use source IDs from
`data/corpus_sources.json` in `expected_sources`. Once source files are ingested,
retrieval eval rows should also pin `expected_chunk_ids`.

The committed `data/eval_set.json` now contains:

- the original Nagel smoke-eval rows
- cross-paper comparison rows
- source-attribution rows
- stance-divergence rows
- grounding-fidelity traps
- unanswerable/adversarial rows

Rows prefixed with `corpus_` are manually authored starter rows. They use the
chunk-pinned schema but leave `expected_chunk_ids` empty until the corresponding
sources are lawfully ingested and stable chunk IDs can be recorded.
