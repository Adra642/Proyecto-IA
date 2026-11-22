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
