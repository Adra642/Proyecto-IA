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

- 2026-11-22: **[DECISION] app hosting moved from Hugging Face Spaces to Google Cloud Run** (user decision). The HF Spaces docs say Gradio and Docker Spaces need a paid plan, except 2 ZeroGPU Spaces; I had wrongly said CPU Basic was free. Plan sections 5.1 and Stage 4 updated.

- 2026-11-23: Quarto site (Stage 6) will be published on **GitHub Pages** (user decision). The live app URL goes into the README in this stage.

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
- 13 sources, all verified live (B4 and B6 added in a follow-up commit): A1/A2 Kaggle CSV + derived analytics (66 docs, `stats.json`), A3 arXiv
  (19 papers: abstract + intro/results/conclusion excerpts), A4 Wikipedia (16), A5 ECB (13 series notes),
  A6 EU law (7, EUR-Lex bot challenge detected, Cellar XHTML fallback), B1 Banco de Espana (7, ES),
  B2 Eurostat (15), B3 World Bank (12), B5 ECB HFCS (3), B7 credit registers (3: ES/DE/FR),
  B8 Banque de France / BaFin / Bundesbank (7: FR/DE).
- `clean.py` (HTML to Markdown, normalize, dedupe, min length), `build_docs.py` (pure stats functions,
  segment cards, summaries, dictionary, findings with Complain leakage check), `writer.py`, `pii_check.py`, `run_all.py`.
- B4 ECB press releases (3 docs from the RSS feed) and B6 EBA consumer pages/guidelines (7 docs, incl. the loan-origination PDF).
- Corpus: 178 documents, about 399k words, formats csv/json/html/pdf/generated, languages en/es/fr/de.

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
- B4 only sees what is in the ECB RSS feed (about 15 recent items, 3 are HTML press releases); the 29 Oct 2026 monetary policy decision was no longer in it and the ECB listing pages are JavaScript-rendered. B6: the EBA Consumer Trends Report page could not be located (guessed URLs gave 404), so it is not in the corpus.
- A first cold run takes about 8 minutes (arXiv crawl delay); later runs take seconds.

### Follow-up: LLM model names (2026-11-19)
- Gemini, called live with the team's key: `gemini-2.5-flash` and `gemini-2.5-flash-lite` are retired for new users (404), `gemini-3.7-flash` and `gemini-3.8-flash` timed out (504) in our test, `gemini-3.5-flash`, `gemini-3.6-flash` and `gemini-3.5-flash-lite` work. On a RAG-style prompt latency was about 1 s (3.5-flash-lite) versus 5 to 13 s (3.5-flash, 3.6-flash); all gave the correct cited answer. Default is `gemini-3.5-flash-lite` **[DECISION confirmed by the user on 2026-11-19; a stronger model can still be set in `config.yaml`]**.
- **Caveat:** the Gemini 3.5 models ignore `temperature` (langchain-google-genai warns "uses fixed sampling defaults"). The plan's `temperature=0` for a fair, reproducible comparison cannot be enforced for Gemini; mitigation: cache evaluation answers (key includes provider and model) and score the cached answers.
- Groq `llama-3.3-70b-versatile` is still a production model in the Groq docs; DeepSeek docs now list `deepseek-flash` and `deepseek-v4-pro` and no longer mention `deepseek-chat`, so the default is `deepseek-flash`. Both checked against the docs only (no keys), not called.
- Smoke tests: Gemini passes live; Groq and DeepSeek are skipped (no keys). The smoke test now skips providers whose key is empty.

## Stage 2: chunking, embeddings, vector DB (branch `stage-2-chunking-index`, based on `stage-0-scaffold` = Stage 1)
Done:
- `rag/chunking.py`: Strategy (`PassthroughChunker`, `MarkdownRecursiveChunker`, `FixedSizeChunker`) + `RoutingChunker` + `build_chunker`; atomic types (`dataset_stats`, `macro_data`) are never split.
- `rag/vectorstore.py` (Repository over Chroma, cosine, metadata filters), `rag/index.py` (incremental by fingerprint, `--rebuild`, `--query`), `core/corpus.py`, `factory.py` wiring.
- `eval/questions.json` (12 questions: 5 D, 2 L, 3 R, 2 U, with gold source ids and `stats.json` keys for gold numbers), `eval/retrieval_metrics.py`, `eval/chunking_experiment.py`.
- Index built: 178 documents, 6,309 chunks at 600/90; cold build 292 s on CPU; re-sync of unchanged corpus about 1 s.

Verified:
- "churn rate in Germany": `kaggle_summary_geography` rank 1, `kaggle_segment_geography_germany` rank 2 (top 3 as required). Spanish and German queries return the BdE complaints page and the SCHUFA page; the geography filter works (`tests/integration/test_index_acceptance.py`, 5 passed).
- Chunk-size experiment (`eval/chunking_results.csv`, `.png`, per question `chunking_detail.csv`): hit@3 = hit@5 = 1.00 for every variant; hit@1 0.80 (300), 0.90 (600, 1000, fixed 600); MRR 0.883, 0.933, 0.933, 0.933. Average top-5 context 2,154 / 2,795 / 3,341 chars. D1 is rank 3 in all variants (the geography summary is outranked by two similar cards); R2 is rank 2 only at 300.
- The 600-character variant took 55 min wall clock but its own timer shows 304 s of embedding: the laptop was suspended; my first explanation (CPU throttling) was wrong.

Stage 1 bug fixed here: single numeric segments were labelled "4.0" (pandas upcast), giving `kaggle_segment_num_of_products_4.0` and `num_of_products.4.0.*` keys in `stats.json`; labels are now plain text ("4"), docs and stats regenerated.

[DECISION] chunk size: 600 characters / 90 overlap, **confirmed by the user on 2026-11-20** (best quality at the smallest context, tied with 1000).

Open issues / limits:
- The experiment separates sizes weakly: 10 questions (one = 0.1), and the atomic documents are not affected by size. Stage 3's larger evaluation and the ablation in Stage 5 will give more signal.
- Header-aware chunking ties with fixed-size chunking on retrieval metrics here; its benefit is the header path kept as citation context.
- Semantic chunking (optional) not implemented.
- The two literature gold answers in `eval/questions.json` are drafts: a human must verify them against the PDFs (Stage 3).
- Chroma index is not committed (`chroma_db/` is git-ignored); Stage 4 decides between a prebuilt index and building on first start.

## Stage 3: baseline vs RAG and evaluation (branch `stage-3-evaluation`, from `main` with Stages 0-2)
Done:
- `rag/chain.py` (`RagChain.answer_no_rag` / `answer_rag`), `rag/prompts.py` (versioned templates), `rag/retrievers.py` (vector retriever), `core.models.RagAnswer`.
  RAG prompt: answer only from the context, cite `[doc_id]`, say so when insufficient, answer in the question's language, never reveal personal identifiers. Privacy guard: requests for a surname or a specific customer record are refused without calling the LLM; an empty retrieval answers "I do not know" without calling it.
- `ThrottledLLM` and `RetryingLLM` decorators (min interval 1 s, 3 retries with 10/20/40 s backoff), applied to every provider before the fallback chain.
- `eval/questions.json`: 14 questions (D 6, L 2, R 4, U 2), `eval/generate_gold.py` (gold numbers from `stats.json` and from regexes over the ingested texts; best segment computed for D4/D5; rewrites questions idempotently), `eval/numbers.py` (tolerant number parsing: `100 000`, `100,000`, `0,2`, `32.4%`).
- `eval/run_eval.py` + `eval/answer_cache.py` (cache in `eval/cache/`, committed; key includes provider:model, prompts version, chunking and embedding settings, corpus hash), `eval/score.py` (hit@k, numeric match, citation check, abstain and PII-leak checks, manual sheet that preserves entries, summaries, examples), `eval/literature_verification.md`, `eval/examples.md`.

Verified:
- `uv run python -m eval.run_eval`: 28 answers, 0 failed (one HTTP 429 recovered by the backoff). Median latency 1.0 s in both modes.
- Automatic results (`eval/auto_summary.csv`): numeric match D 0.83 (RAG) vs 0.00 (No RAG), R 0.75 vs 0.25; hit@4 and citations 0.83 to 1.00; L questions: both papers retrieved and cited; U questions 1.00 in both modes.
- Live tests (`tests/integration/test_chain_live.py`): exact figure with citation, PII refusal without a retrieval, baseline has no sources; the plain Germany question is an expected failure (`xfail`).
- 184 unit tests pass, ruff and mypy clean.

Questions changed after the first run (plan rule: replace questions the baseline already answers):
- R1 and R2 (deposit coverage EUR 100 000, interchange caps 0.2 %/0.3 %): the baseline answered both correctly. Replaced by the fixed EUR 0.05 option of Article 3(2)(b) of Regulation 2015/751 and the latest Eurostat internet-banking share for Germany. The old R2 stays as **R4** (baseline right, RAG initially wrong).
- D3 reworded to ask for the exact shares (the baseline already got the qualitative conclusion); D4/D5 gold is now the best segment, so the numeric check matches what is asked; **D6** added (plain "churn rate in Germany").

Problems found and what was done:
- Chunk header paths repeated the document's long H1 (EU laws: about 150 characters) in every chunk's embedding text; the H1 is now dropped (`CHUNKER_VERSION = 2`, part of the index fingerprint, so the index rebuilt itself). R4 improved; R1 did not.
- A shorter "Source:" line in the churn cards (without the country list) was tried and reverted: no measurable gain (Germany card rank 5 to 4, another question 2 to 3).
- Leak check flagged an answer that merely repeated the customer id from the question: it now ignores identifiers present in the question. Number parser read "0.324" as 324: fixed.

Open issues / for the team:
- **Manual scoring pending** (`eval/manual_scoring.csv`, 28 rows, scale 0/1/2 + citation Y/N), then `uv run python -m eval.score --summary`.
- **Human verification pending** of the L1 and L2 gold answers in `eval/literature_verification.md` (PDF sentences are quoted there).
- The three RAG failures are kept as diagnostics for Stage 5: D6 (hybrid/BM25 and a geography filter), R1 (neighbour-chunk or rerank), R4 (source filtering). The baseline prompt allows "I do not know", so U does not separate the modes and the baseline does not hallucinate numbers; a baseline prompt without that permission would likely look worse.
- One cached run per question and Gemini 3.5 ignores `temperature` (the D5 citation flipped between two runs with identical retrieval).
- `eval/chunking_results.csv` was produced before the header-path change and before the question changes (the 600-character decision stands, to be re-measured in the Stage 5 ablation).
- Token counts are not recorded (the LLM wrapper returns text only).
- The evaluation set has 14 questions instead of 12 (three diagnostics).

## Stage 4: web app and Cloud Run deployment (branch `stage-4-app-cloudrun`, from `main` with Stages 0-3)
Done:
- `app/handlers.py` (`AppService`: modes, one retry on rate limit, friendly Spanish errors for missing key, rate limit and provider failure, question length guard; source and status formatting), `app/ui.py` (Gradio Blocks: Chat, Comparar, Acerca de; examples from `eval/questions.json`), `app/app.py` (composition, index sync at start, `PORT`).
- `deploy/Dockerfile` (uv, CPU torch, e5 model and Chroma index baked in), `.dockerignore`, `deploy/cloudbuild.yaml` (30 min timeout).
- Plan and reference updated for Cloud Run; README sections "Run the app" and "Deployment".
- Gradio 6.32 added with `uv add`.

Verified: see the PR (unit tests, local run through `gradio_client`: RAG gives 32.4% for Germany with citations and sources; No RAG says it does not know; image build with Podman).

Open / pending (needs the team):
- **Cloud Run deployment itself needs your Google Cloud project, billing account, secret and budget alert**; the public-URL acceptance (another device, private window) is checked after the first deploy.
- `deploy/cloudbuild.yaml` and the Cloud Shell command are untested against a real project.
- *RAG avanzado* and the filter controls are Stage 5.

## Stage 5: advanced RAG (branch `stage-5-advanced-rag`, from `main` with Stages 0-4)
Done:
- `rag/lexical.py` (BM25 + `matches_where`), `rag/retrievers.py` (`HybridRetriever` weighted RRF, `RerankingRetriever`), `rag/rerank.py` (FlashRank), `rag/router.py` (rules: small talk skips retrieval; source-type and single-country filter; fallback to an unfiltered search when empty; user filters override), `RagChain.answer_advanced`, `RagAnswer.mode = rag_advanced` + `route`, `Container.retriever(kind)`.
- App: mode *RAG avanzado*, filter dropdowns (source type, geography, language), three-column *Comparar* tab, route shown in the status line.
- `eval/ablation.py` -> `eval/ablation.csv`, `eval/ablation_detail.csv`; `run_eval`/`score` handle `rag_advanced` (cache key includes retrieval settings and router version).
- Config: `retrieval.candidates/vector_weight/bm25_weight/rrf_k/rerank_enabled/rerank_model`.

Verified: 223 unit tests, 95% coverage, ruff and mypy clean; `tests/integration/test_advanced_live.py` 3 passed on the real index; ablation and a full `run_eval` (42 answers, 0 failed).

Results (see README table): hybrid raises hit@1 0.667 to 0.833, hit@4 0.917 to 1.000, MRR 0.794 to 0.903 and fixes D6 (rank 5 to 1; wrong to right answer). Filters and reranking add nothing on top of hybrid on our 12 questions; MiniLM-L-12 (hit@1 0.667) and MultiBERT (0.417, hit@4 0.5) are worse. **Decision taken by me, not confirmed: reranking stays implemented and measured but disabled by default** (`rerank_enabled: false`); the plan lists reranking but not that it must be on, and enabling a worse configuration for the demo made no sense. Tell me if you want it on.

Open / pending:
- **Manual scoring** (`eval/manual_scoring.csv` now has 42 rows: three modes) and the L1/L2 literature check are still pending (user chose to do them at the end).
- Answer level is one run per question and Gemini 3.5 ignores `temperature`: R4 numeric went 1.0 to 0.5 and D1 citation 1 to 0 in advanced mode, probably noise (retrieval is better in both); D6, R1 and R4 are not fully solved (R1 needs neighbour chunks).
- Semantic chunking and an LLM router were not built (cut order); the router is rule-based and English/Spanish/French/German keywords only.
- FlashRank downloads its model on first use; the Docker image does not bake it because reranking is off.
- `eval/chunking_results.csv` was re-measured on 2026-11-23 (see the follow-up below).

### Stage 5 follow-up: smaller image and faster rebuilds (2026-11-23, user request)
Measured with `podman build` on the laptop:
- Image size **4.67 GB to 2.76 GB**. Causes found: `uv` left a 1.74 GB wheel cache in the layer (`--no-cache` fixes it); the data-pipeline packages (pymupdf, kagglehub, bs4, lxml, markdownify) are now a `pipeline` dependency group left out of the image (venv 2166 MB to 2030 MB; pandas stays because Gradio needs it). The e5 model (470 MB, fp32) and torch (805 MB) are untouched on purpose (changing them would change the embeddings).
- Layer order is dependencies, then model + index, then the app. Rebuild after changing only `app/`: dependencies and the index are `Using cache`, 1 s instead of re-embedding for about 208 s.
- The cold-build total time was **not measured** (the laptop clock jumped during the run); the index step took 208 s as before.
- `cloudbuild.yaml`: pulls `:latest` and uses `--cache-from` (BuildKit inline cache), pushes the SHA and `latest` tags, `_CONTEXT` substitution so `gcloud builds submit rag_project ...` uploads only the project folder. Default machine type kept (free minutes). **Not yet tried on Cloud Build:** check that the second build there shows cache hits.
- Dropped the multi-stage idea: with `--no-cache` it saves about 50 MB and would stop the inline cache from covering the dependency and index layers.
- Not done (user chose safe): deleting torch `test`/`include` folders (about 170 MB).
- The slim image answers correctly (RAG and RAG avanzado) with the key passed as an environment variable.

## Stage 6: Quarto memoir and publication (branch `stage-6-quarto`, from `main` with Stages 0-5)
Done:
- `docs/` Quarto website in Spanish: `index`, `01-motivacion-datos`, `02-pipeline`, `03-chunking-vectordb`, `04-evaluacion`, `05-rag-avanzado`, `06-app-despliegue`, `07-equipo-referencias`, `_quarto.yml` (freeze auto), `_helpers.py` (CSV/JSON readers, Markdown tables, code excerpts cut from the sources with `ast`, so nothing heavy is imported), `styles.css`, `img/app_inicio.png` (screenshot of the live app).
- Content comes from the repository: source catalog and dictionary from `data/`, chunk-size table and plot from `eval/chunking_results.csv`, evaluation tables and before/after examples from `eval/results.json` and `auto_summary.csv`, ablation from `eval/ablation.csv`; Mermaid diagrams for the pipeline, the architecture and the advanced mode.
- `.github/workflows/rag_project-docs.yml` (repo root, filtered to `rag_project/**`): renders with Quarto 1.10.19 and deploys to GitHub Pages on `main`; pull requests only render.
- README: links and a "Quarto documentation site" section.

Verified: `uv run quarto render docs` renders all 8 pages with no Python errors (checked the HTML for tracebacks and empty tables); one rendered page inspected in a screenshot. The workflow file parses as YAML; **it has not run yet** (it needs the PR merged and Pages enabled).

Needs the team (cannot be done by me):
- **Enable Pages:** repo *Settings, Pages, Source = GitHub Actions*; then check the first run of `rag_project-docs` and open the URL.
- **Fill in** the team table (`docs/07-equipo-referencias.qmd`: names, roles, contributions) and the **Loop URL** in `docs/index.qmd`; post the site URL in Loop.
- **Manual scoring** (still pending): the evaluation page shows a "pendiente" note until `eval/manual_scoring.csv` has scores, then fills in its own table.
- Only one app screenshot (the empty *Chat* tab): a headless browser cannot type a question. Add screenshots of an answered question, the *Fuentes* accordion and the *Comparar* tab to `docs/img/` and reference them in `06-app-despliegue.qmd` (optional).
- Link checks: the site links to the app, repo and project 1 folder (`bank-churn-prediction`); the Pages URL assumes the repo is `Adra642/Proyecto-IA` and public (it is).

### Chunk-size experiment re-measured (2026-11-23, branch `chunking-remeasure`)
Re-ran `uv run python -m eval.chunking_experiment --with-fixed` with chunker version 2 and the 12 retrieval questions (the first run used 10 questions and the old header paths).
- hit@5 = 1.00 for all variants. hit@1 / MRR: 300 chars 0.667 / 0.781, **600 chars 0.667 / 0.794**, 1000 chars 0.750 / 0.836, fixed 600 0.750 / 0.836. Average top-5 context: 2,128 / 2,568 / 2,975 / 2,556 chars.
- Against the first run (MRR 0.933 for 600, 1000 and fixed): the drop for 600 comes from two questions added in Stage 3 (D6 rank 5, R4 rank 2) and D3, which went from rank 1 to rank 2 (cause not isolated: header change or question rewrite). The only question that separates 600 from 1000 / fixed is R4 (rank 2 vs 1), so the gap is one question out of 12.
- [DECISION] kept at 600 / 90 (confirmed by the user on 2026-11-20): smaller context and an inconclusive gap. **1000 is now slightly ahead on MRR; the user may want to revisit.** Changing it would rebuild the index and invalidate the cached answers.
- The site page `03-chunking-vectordb` and the README were updated with the new numbers; the plot's y-axis was widened (the 300 bar was below the old lower limit).

### Answer cache removed (2026-11-23, user request, branch `chunking-remeasure`)
- The user decided the evaluation answer cache is not needed and that run-to-run differences from Gemini ignoring `temperature` do not matter.
- Removed `eval/answer_cache.py`, the 59 files in `eval/cache/`, the `--refresh` flag, `key_parts`, `Container.corpus_hash` and the `from_cache` field; `run_questions` now always calls the chain. README, site page `04-evaluacion`, `documentacion` and `.dockerignore` updated.
- The committed `eval/results.json` and `results.csv` come from the last cached run and still have a `from_cache` field; the next `eval.run_eval` drops it. `PROMPT_VERSION` and `ROUTER_VERSION` stay (still tested) but no longer feed any key.
- Consequence: every `eval.run_eval` run now spends LLM calls (free-tier rate limits apply) and may change the numbers slightly.
- Chunk size: **kept at 600 / 90** (option 2). The cache was one of the two costs of switching to 1000; the other (index rebuild, re-running the evaluation and ablation, updating the site) remains.

### Cleanup / refactor (2026-11-25, branch `refactor-cleanup`)
Audit with `ruff` (F, ERA), `vulture` and `deptry`, checked by hand against the code, the Quarto pages and the Dockerfile.
- Removed dead code left by the answer-cache removal: `PROMPT_VERSION` (`rag/prompts.py`), `ROUTER_VERSION` (`rag/router.py`) and their two tests' assertions.
- Removed the unused `hf_token` setting (nothing read it; `.env` values never reach `huggingface_hub`, which reads the real `HF_TOKEN` environment variable). Dropped it from `.env.example`, the README table and `tests/conftest.py`.
- Removed the unused direct dependencies `langchain` and `langchain-core` (never imported; `langchain-core` stays as a transitive dependency of the provider packages, the orchestration packages `langgraph*` left the lock). `scikit-learn` moved to the `pipeline` group (only `build_docs.py` imports it; `sentence-transformers` still brings it into the image). `uv.lock` regenerated.
- Deleted two empty `.gitkeep` files in folders that already have content (`docs/`, `tests/fixtures/`).
- Folder structure reviewed: no change needed (`core/ rag/ data_pipeline/ eval/ app/ deploy/ docs/ tests/` match the plan; `documentacion/` is the team's defence guide).
- Left on purpose: `export-reqs` poe task (the plan keeps it as a convenience), `from_cache` in the committed `eval/results.json` (dropped on the next `eval.run_eval`).
Verified: `ruff check`, `ruff format --check`, `mypy` clean; 222 unit tests pass, 95% coverage.

### Spanish answers, Quarto navigation and presentation guide (2026-11-25, branch `refactor-cleanup`)
- **Answers in Spanish:** the No-RAG and RAG prompts now say "always answer in Spanish"; the fixed `REFUSAL_PII` and `NO_CONTEXT` messages are in Spanish; `eval/score.py` recognises Spanish abstentions. Tests updated (223 pass). The saved `eval/results.json` is from before this change (English answers); re-run `eval.run_eval` to refresh it (spends LLM calls).
- **Quarto navbar:** a *Contenido* dropdown holds the seven parts. Anterior / Siguiente buttons come from `page-navigation`; Quarto only builds them from a sidebar, so a sidebar with the page order is declared and hidden in `styles.css`.
- **Presentation guide:** `guia_presentacion/` moved to `documentacion/` (separate from the Quarto `docs/`), rewritten for the current state, with new files for advanced RAG and for the app, deployment and site.

### Final QA round (2026-11-26, branch `stage-7-final-qa`)
- Re-ran `eval.run_eval` (42 answers, 0 failed, all in Spanish) and `eval.score`. Numeric match unchanged (D 0.00 / 0.83 / 1.00, R 0.25 / 0.75 / 0.62); citation OK moved with run-to-run noise (D: RAG 0.83 to 0.50, advanced 0.67 to 0.83). Median latency 1.2 / 1.0 / 1.8 s. R3 without RAG now answers a wrong, out-of-date ECB rate (3.25% since 2024-10-23): the baseline is confidently wrong there, not silent.
- Example buttons in the app are in Spanish: `eval/questions.json` has a `question_es` field (used only by `app/ui.py`; evaluation still uses the English `question`, so results stay comparable).
- DeepSeek key tested: smoke test passes, `deepseek-flash` answers the three modes in Spanish through the chain, and the fallback works (gemini with an invalid key and `fallback_order: [gemini, deepseek]` answered through DeepSeek). Note: DeepSeek without RAG answered the Germany churn rate correctly (32.44%), so the dataset appears in its training data; Gemini's baseline does not know it.
- Quarto: removed the CSS that squeezed the "En esta página" table of contents; team names in `docs/07` formatted.
- Still pending: manual scoring and L1/L2 verification (team).
