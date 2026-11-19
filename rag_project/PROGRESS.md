# Progress log

## Decisions
- 2026-11-19: Python pinned to 3.12 (system 3.14 is too new for torch/chromadb wheels).
- 2026-11-19: PRs opened with `gh` (user installs and authenticates it).
- 2026-11-19: PII check is scoped: no surnames or CustomerIds in dataset-derived docs; across the
  whole corpus no CustomerIds and no whole-word surnames, except a logged allowlist of collisions
  with ordinary words in external sources.
- 2026-11-19: arXiv documents keep metadata, abstract and intro/results/conclusion (truncated), not
  full text, with a link to the paper.
- `core/` shared package added to the layout; `DocumentMetadata` has an extra `format` field
  (csv/json/html/pdf/generated) so the run report can count formats.

## Stage 0: scaffold and environment (branch `stage-0-scaffold`)
Done:
- uv project, Python 3.12 pin, CPU-only torch, `uv.lock`, dev/docs groups.
- `config.yaml`, typed settings (YAML < `.env` < env), logging, exceptions, domain models.
- LLM adapters (Gemini, Groq, DeepSeek), `FallbackLLM`, `build_llm`; e5 embedder; `factory.py`.
- 25 offline unit tests (97% coverage), integration smoke tests for each LLM provider and e5.
- ruff, mypy (strict), pre-commit (ruff), poe tasks, README v1.

Verified: `uv sync`, `uv run pytest`, `ruff check`, `ruff format --check`, `mypy` clean; e5 integration
test passes (384-dim vector).

Pending / open:
- LLM integration tests are skipped until `GOOGLE_API_KEY` (and optionally Groq/DeepSeek keys) are in `.env`.
- `gh` must be installed and authenticated before the PR can be opened.
