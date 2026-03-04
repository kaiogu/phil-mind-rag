# Manual / Integration Testing Backlog

These items require either a live API key, a real browser session, or
integration with Gradio that isn't practical to automate right now.

---

## Gradio UI (`phil_mind_rag/app/ui.py`)

All callbacks are tested indirectly through `test_pipeline.py`, but the
Gradio wiring itself needs manual verification:

- [ ] **Upload tab** — upload `data/raw/nagel_bat.pdf`, confirm progress
      messages appear step-by-step (parse → chunk → embed → store).
- [ ] **Upload tab** — verify metadata auto-fill (title/author) fires on
      file selection and populates the text boxes.
- [ ] **Ask tab** — ask a question, confirm answer appears and the Sources
      panel populates below it with file name, section, score, and snippet.
- [ ] **Ask tab** — ask with an empty input, confirm graceful error message.
- [ ] **Library tab** — click Refresh after ingestion, confirm the new row
      appears in the table.
- [ ] **Library tab** — confirm table is populated on page load (not just
      after Refresh).

---

## `UnstructuredPDFParser` (real PDF)

`UnstructuredPDFParser` is excluded from unit tests because it shells out
to poppler/tesseract. Run manually:

```bash
python - <<'EOF'
from pathlib import Path
from phil_mind_rag.ingestion.parser import UnstructuredPDFParser
doc = UnstructuredPDFParser().parse(Path("data/raw/nagel_bat.pdf"))
print(f"{len(doc.sections)} sections extracted")
for s in doc.sections[:3]:
    print(f"  [{s.title}] {s.text[:80]!r}")
EOF
```

Expected: several sections with non-empty text.

---

## `ChromaVectorStore` (integration)

The vector store is mocked in unit tests. To verify the real persistence
layer, run the pipeline end-to-end:

```bash
python scripts/run_eval.py --eval-set data/eval_set.json --top-k 5
```

This requires `OPENAI_API_KEY` in `.env` and at least one ingested document.

---

## `scripts/run_eval.py` (full CLI)

- [ ] Run with an empty vector store — confirm it exits with a clear error.
- [ ] Run after ingesting `nagel_bat.pdf` — confirm RAGAS scores print.
- [ ] Verify scores are plausible (faithfulness > 0.5, recall > 0.4).

---

## HuggingFace Spaces deployment

- [ ] Build the Docker image locally: `docker build -t phil-mind-rag .`
- [ ] Run it: `docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... phil-mind-rag`
- [ ] Confirm the UI loads at `http://localhost:7860`.
- [ ] Deploy to HF Spaces and set `OPENAI_API_KEY` as a Space secret.
