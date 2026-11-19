# RAG project: customer churn in European retail banking

A RAG system (LangChain + ChromaDB + multilingual-e5 + Gradio) on customer retention and churn in
French, German and Spanish retail banking. It turns the Kaggle *Bank Customer Churn* CSV into
computed text documents and combines them with public sources (papers, Wikipedia, ECB/Eurostat/World
Bank data, EU law, regulators) to show that RAG beats the same LLM without retrieval.

> **Status:** Stages 0 (scaffold) and 1 (data pipeline) done. Later stages (index, evaluation, app,
> docs) are tracked in [PROGRESS.md](PROGRESS.md); sections below marked *Stage n* are filled in then.
> The project is self-contained: it does not import from the churn-prediction project.

## Architecture

```mermaid
flowchart LR
    S[Sources: CSV, API, HTML, PDF] --> P[Data pipeline]
    P --> D[data/processed Markdown]
    D --> I[Chunking + embeddings]
    I --> V[(ChromaDB)]
    Q[Question] --> R[Retriever]
    V --> R
    R --> L[LLM]
    L --> U[Gradio app]
```

## Links
- App: *Stage 4*
- Quarto site: *Stage 6*
- Repository: <https://github.com/Adra642/Proyecto-IA> (folder `rag_project/`)

## Prerequisites
- Python 3.12 (pinned in `.python-version`; uv downloads it if missing), git
- [uv](https://docs.astral.sh/uv/)
  - Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Quarto (only for the docs, *Stage 6*)

## Setup

Linux (bash):
```bash
git clone git@github.com:Adra642/Proyecto-IA.git && cd Proyecto-IA/rag_project
uv sync                  # creates .venv and installs the locked dependencies
cp .env.example .env     # then fill in the keys
```

Windows (PowerShell):
```powershell
git clone git@github.com:Adra642/Proyecto-IA.git; cd Proyecto-IA\rag_project
uv sync
Copy-Item .env.example .env
```

Activating `.venv` is optional because `uv run` uses it (`source .venv/bin/activate` on Linux,
`.venv\Scripts\Activate.ps1` on Windows; if PowerShell blocks scripts, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`). Fallback without uv:
`python -m venv .venv`, activate it, `pip install -e ".[dev]"`; never outside `.venv`.

## Configuration

Defaults are in `config.yaml`; environment variables and `.env` override them. Only the selected
LLM provider's key is needed.

| Variable | Purpose | Where to get it |
|---|---|---|
| `LLM_PROVIDER` | `gemini` (default), `groq` or `deepseek`; overrides `llm.provider` | n/a |
| `GOOGLE_API_KEY` | Gemini | <https://aistudio.google.com/apikey> |
| `GROQ_API_KEY` | Groq | <https://console.groq.com/keys> |
| `DEEPSEEK_API_KEY` | DeepSeek (pay-as-you-go) | <https://platform.deepseek.com> |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Kaggle dataset download, only if anonymous access fails | Kaggle account settings, API token |
| `HF_TOKEN` | Hugging Face Space deployment (*Stage 4*) | <https://huggingface.co/settings/tokens> |

`llm.fallback_order` (e.g. `[gemini, groq, deepseek]`) makes the app try the next provider when
the active one is rate-limited. Nested values can be overridden with `__`, e.g.
`LLM__MODELS__GROQ=...`. Other `config.yaml` options (sources, chunk size, k) arrive in later stages.

## Run the data pipeline

```bash
uv run python -m data_pipeline.run_all                 # all enabled sources
uv run python -m data_pipeline.run_all --sources a1,a3 # only some sources (a2 is an alias of a1)
uv run python -m data_pipeline.run_all --refresh       # bypass the raw download cache
```

Per source: fetch (polite, cached, retried) -> clean (HTML to Markdown, unicode/whitespace
normalization, de-duplication, tiny documents dropped) -> write. A failing source or page is
logged and reported but never stops the others.

| Output | What |
|---|---|
| `data/processed/<source>/<doc_id>.md` | one Markdown document per item, YAML front matter = the Section 5.3 metadata |
| `data/processed/stats.json` | every number used in the derived churn documents, keyed by dotted name |
| `data/SOURCES.md` | generated table: source, formats, languages, retrieval date, license |
| `logs/run_report.json`, `logs/pipeline.log` | documents/words per source, failures, PII check (git-ignored) |
| `data/raw/` | cache of raw downloads (git-ignored); delete it or use `--refresh` to re-download |

Idempotence: `retrieved_at` comes from the cache manifest and files are written in a stable order,
so running the pipeline twice with the same cache gives byte-identical `data/processed/`.
`--refresh` intentionally updates live data (ECB rates, Eurostat, ...).

Sources (`config.yaml` -> `sources`): A1/A2 Kaggle churn CSV and derived analytics, A3 arXiv,
A4 Wikipedia, A5 ECB Data Portal, A6 EU law (EUR-Lex with Cellar fallback), B1 Banco de España,
B2 Eurostat, B3 World Bank, B5 ECB HFCS, B7 credit registers (CIRBE, SCHUFA, FICP), B8 Banque de
France / BaFin / Bundesbank. Each has an `enabled` flag and its own options (queries, pages, series).

Privacy: `RowNumber`, `CustomerId` and `Surname` are dropped before any document is created.
After every run a scoped check scans `data/processed/` (no CustomerId anywhere, no surname in
documents derived from the dataset; surnames in external text are logged as collisions).

## Build the index, run the evaluation and the app
*Stages 2 to 4.*

## Tests and quality
```bash
uv run pytest                    # unit tests, offline (coverage report included)
uv run pytest -m integration     # real LLM / embedding calls and live fetches; needs keys, downloads the e5 model
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pre-commit install        # run from the git root with -c rag_project/.pre-commit-config.yaml
```
The hooks live in `.pre-commit-config.yaml` (ruff only; mypy runs through `poe typecheck`).
From the repo root: `uv run --project rag_project pre-commit install -c rag_project/.pre-commit-config.yaml`.

## Task shortcuts (identical on Linux and Windows)
`uv run poe pipeline | test | test-integration | lint | typecheck | export-reqs`; `index`, `eval`,
`app` and `docs` are defined but their modules arrive in later stages.

## Deployment, Quarto docs
*Stages 4 and 6.*

## Project structure
```
rag_project/
├── config.yaml, pyproject.toml, uv.lock, .python-version, .env.example
├── core/            settings (pydantic-settings), logging, exceptions, domain models
├── rag/             llm.py (LLM adapters + fallback), embeddings.py (e5 adapter)
├── factory.py       composition root: wires components from settings
├── data_pipeline/
│   ├── fetchers/    one module per source + base.py (template, registry)
│   ├── http.py, cache.py, resilience.py   polite client, raw cache, retry/rate limit
│   ├── clean.py, build_docs.py            cleaning steps; pandas -> churn documents + stats
│   ├── writer.py, pii_check.py            output files, SOURCES.md; privacy scan
│   └── run_all.py   orchestrator + run report
├── eval/, app/, docs/   later stages
├── data/            raw/ (git-ignored cache), processed/ (committed documents)
└── tests/           unit/ (offline), integration/ (marked), fixtures/
```
`core/` is a small shared package so `data_pipeline` and `rag` never import each other.

Design patterns in use so far:
- Adapter + Factory: `LLMClient` / `build_llm`, `Embedder` / `build_embedder`.
- Decorator: `FallbackLLM`; retry with backoff and per-host rate limiting (`resilience.py`).
- Template Method + Registry: `BaseFetcher.fetch()` implements cache, error isolation and logging
  once; subclasses implement `_items`, `_download`, `_parse`; `@register_fetcher("a4")` registers them.
- Chain of fallbacks: EUR-Lex HTML, then Cellar XHTML, then Cellar PDF.
- Pipeline of small pure functions: the cleaning steps and the statistics in `build_docs.py`.
- Dependency injection with a composition root (`factory.py`); typed settings and domain models.

## Troubleshooting
- *Missing API key:* the error names the variable; set it in `.env` or the environment.
- *Rate limits:* set `llm.fallback_order` or switch `LLM_PROVIDER`.
- *Kaggle:* the public dataset downloads anonymously with `kagglehub`. If it ever needs auth, set
  `KAGGLE_USERNAME` / `KAGGLE_KEY`, or put a raw CSV URL in `sources.a1.fallback_url`.
- *EUR-Lex blocking:* EUR-Lex answers bots with an empty challenge page; the fetcher detects it
  and falls back to the Publications Office (Cellar) XHTML, then PDF. Failures show in the run report.
- *HTTP 429 / slow first run:* Wikimedia and arXiv rate-limit; the client retries with backoff.
  arXiv honours its 15 s crawl delay, so the first run takes several minutes; later runs use the cache.
- *A source is empty in the report:* see `logs/pipeline.log` and the `errors` of that source in
  `logs/run_report.json`; re-run it alone with `--sources`.
- Empty index, Space cold start: *added with their stages.*

## Data attribution and licenses
Sources and licenses are listed in `data/SOURCES.md` (generated by the pipeline, *Stage 1*).

## Team
*To be filled in.*
