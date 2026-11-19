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

## Stage 1: automated data pipeline (branch `stage-1-pipeline`, based on `stage-0-scaffold`)
Done:
- Infrastructure: polite `HttpClient` (User-Agent, robots.txt, >=1 s per host, retry/backoff), `RawCache`
  with manifest, `BaseFetcher` (Template Method) + `@register_fetcher` registry.
- 11 sources, all verified live: A1/A2 Kaggle CSV + derived analytics (66 docs, `stats.json`), A3 arXiv
  (19 papers: abstract + intro/results/conclusion excerpts), A4 Wikipedia (16), A5 ECB (13 series notes),
  A6 EU law (7, EUR-Lex bot challenge detected, Cellar XHTML fallback), B1 Banco de Espana (7, ES),
  B2 Eurostat (15), B3 World Bank (12), B5 ECB HFCS (3), B7 credit registers (3: ES/DE/FR),
  B8 Banque de France / BaFin / Bundesbank (7: FR/DE).
- `clean.py` (HTML to Markdown, normalize, dedupe, min length), `build_docs.py` (pure stats functions,
  segment cards, summaries, dictionary, findings with Complain leakage check), `writer.py`, `pii_check.py`, `run_all.py`.
- Corpus: 168 documents, about 385k words, formats csv/json/html/pdf/generated, languages en/es/fr/de.

Verified (commands run):
- `uv run python -m data_pipeline.run_all` (cold, then cached): 11 sources ok, 0 fetch errors in the last run.
- Second and third runs: `diff -r` of `data/processed/` and `data/SOURCES.md` is empty (byte-identical).
- PII: 0 CustomerId hits anywhere; 0 surname hits in derived documents; an independent grep for distinctive
  surnames returned nothing. 165 external surname collisions (common words/authors' names) are logged in `logs/run_report.json`.
- `uv run pytest`: 104 unit tests, 96% coverage on `core`, `rag`, `data_pipeline`; ruff and mypy clean.
- `uv run pytest -m integration tests/integration/test_fetchers_live.py`: 11/11 sources return documents.
- A failing source does not stop the run: unit test with one crashing and one failing fetcher.

Findings in the data (computed by the pipeline): overall churn 20.4%; Germany 32.4%; Complain mirrors Exited
(99.8% of churners complained, correlation 0.996: target leakage); satisfaction score is flat (19.6% to 21.8%).

Problems found and fixed during the live runs:
- `export.arxiv.org/pdf` is disallowed by robots.txt: PDFs now come from `arxiv.org/pdf` (Crawl-delay 15 s respected).
- Wikimedia returned HTTP 429: Wikipedia interval raised to 2 s, global retries 5 with base 2 s; "SCHUFA" is titled "Schufa".
- `<body class="layout-no-sidebars">` (Banque de France) matched the sidebar rule and emptied the page: containers are now exempt.
- PII scan: front matter excluded; derived-doc check is case-sensitive (surnames like "Band" and "See" are ordinary words in our templates).

Open issues / for the team:
- LLM integration smoke tests (Stage 0) still need `GOOGLE_API_KEY` in `.env`; not run.
- Kaggle dataset license is not verified (listed as such in `SOURCES.md`); scraped regulator pages: reuse terms not verified.
- B5 HFCS: the ECB does not expose the statistical tables at a stable URL, so the corpus has the HFCS overview page and two research PDFs (first 60k characters), not wealth-by-age tables. Replace or add a source if the team finds a direct table.
- arXiv: 19 unique papers (4 queries overlap), not 25; the "results" excerpt of some papers is picked by a heading heuristic and can be imperfect.
- User-Agent is generic (`rag-project-student-bot`); Wikimedia asks for contact details, the team may add a contact URL in `config.yaml`.
- B4/B6 (optional sources) not implemented.
- A first cold run takes about 8 minutes (arXiv crawl delay); later runs take seconds.
