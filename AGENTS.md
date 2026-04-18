# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `phil_mind_rag/`. Keep pipeline wiring in `phil_mind_rag/pipeline.py`; most other modules should stay focused on one layer: `ingestion/`, `retrieval/`, `generation/`, `agents/`, `eval/`, and `app/`. Tests live in `tests/` with shared fixtures in `tests/conftest.py`. Utility scripts belong in `scripts/`. Design notes live in `docs/`. Local runtime data is under `data/` (`raw/`, `processed/`, `chroma/`, `registry.json`); treat generated contents there as local state unless explicitly curated like `data/eval_set.json`.

## Build, Test, and Development Commands
Use `uv` for environment and command execution.

- `uv sync` installs runtime and dev dependencies.
- `python main.py` starts the Gradio app on `http://localhost:7860`.
- `python scripts/run_eval.py` runs the RAG evaluation harness.
- `uv run pytest` runs the full test suite.
- `uv run pytest tests/test_pipeline.py -v` runs one file.
- `uv run ruff check .` lints and checks import order.
- `uv run ruff format .` formats code to the repository standard.
- `uv run ty check` runs static type checks.

## Coding Style & Naming Conventions
Target Python `3.12` and follow Ruff’s `88`-character line length. Use 4-space indentation, type annotations on public functions, and small modules with explicit interfaces. Prefer `snake_case` for functions, variables, and modules; `PascalCase` for classes and Pydantic models; `UPPER_SNAKE_CASE` for constants. Keep imports sorted and move annotation-only imports under `TYPE_CHECKING` unless runtime evaluation requires otherwise.

## Testing Guidelines
Tests use `pytest` with verbose output and short tracebacks. Add unit tests alongside behavior changes, especially for chunking, retrieval, prompts, security, and agent graph logic. Follow the existing pattern `tests/test_<module>.py` and name test functions `test_<behavior>()`. Keep tests deterministic and replace external services with fakes or stubs.

## Commit & Pull Request Guidelines
Commits follow emoji-prefixed Conventional Commits, for example `🐛 fix(agents): break circular import on startup` or `👷 ci(pre-commit): add pytest to pre-commit hook`. Keep scopes narrow and descriptions imperative. Before opening a PR, run `uv run ruff check .`, `uv run ty check`, and `uv run pytest`. PRs should explain the behavioral change, note any config or data impacts, link the relevant issue when available, and include screenshots only for UI changes.

## Configuration & Security Tips
Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Do not commit secrets or local vector-store artifacts. Changes touching `phil_mind_rag/security.py` or file upload flows should include tests for prompt-injection or document-validation regressions.
