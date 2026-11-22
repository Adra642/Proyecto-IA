# RAG Project Plan: reference sections

Moved out of `RAG_PROJECT_PLAN.md` to keep it small. Same content, same numbering.

## 3. Dataset facts (verify on download)

Kaggle `radheshyamkollipara/bank-customer-churn`: about **10,000 rows, 18 columns** (the page itself could not be read automatically; this comes from third-party analyses of the same dataset). The CSV file name was not confirmed, so find it with `glob("*.csv")`.

| Column | Meaning | Notes for the corpus |
|---|---|---|
| RowNumber, CustomerId, Surname | identifiers | **drop (PII)** |
| CreditScore | credit score | explain what credit scores / credit registers are in FR/DE/ES |
| Geography | France / Germany / Spain | enables country-level segmentation and country context documents |
| Gender, Age | demographics | age-band segmentation; GDPR and fairness context |
| Tenure | years as customer | tenure-band segmentation |
| Balance, EstimatedSalary | money | rates, inflation, income and wealth context |
| NumOfProducts | 1–4 products | strong churn signal in this dataset; cross-selling literature |
| HasCrCard, Card Type | credit card; DIAMOND/GOLD/PLATINUM/SILVER | card economics, interchange fees, rewards |
| IsActiveMember | activity flag | digital banking and engagement context |
| Point Earned | loyalty points | loyalty programs |
| Satisfaction Score | 1–5 | NPS/CSAT context; check whether it actually predicts churn |
| Complain | 0/1 | **suspiciously aligned with `Exited`** (third-party analyses show almost all churners complained), so it is likely target leakage. Have the pipeline measure this and write it up as a finding. |
| Exited | churned (target) | overall churn is about 20% (verify) |

---

## 4. Source catalog

Tiers: **A = core (required)** · **B = extended: B1, B2, B3, B5, B7 and B8 are required (confirmed by the team: ES/FR/DE regulator and credit-bureau pages, plus macro and wealth data); B4 and B6 are optional** · **Out of scope (decided):** CFPB complaints and Banking77. The corpus must be clearly multi-source, multi-format (CSV, API JSON, HTML, PDF) and multilingual (EN/ES/FR/DE).

Every ingested document must carry the metadata in Section 5.3. Respect `robots.txt`, rate-limit requests (≥1 s between calls), set a descriptive `User-Agent`, cache raw downloads, and record each source's license in `data/SOURCES.md`.

### Tier A — core

| ID | Source | Type | Automation | Maps to columns | Notes |
|---|---|---|---|---|---|
| A1 | **Kaggle Bank Customer Churn CSV** | structured | `kagglehub.dataset_download("radheshyamkollipara/bank-customer-churn")`; if it needs auth, read `KAGGLE_USERNAME`/`KAGGLE_KEY` from env. Fallback: download from a URL in `config.yaml` (e.g., a GitHub release of the raw file) | all | the base dataset |
| A2 | **Derived analytics documents** generated from A1 | generated text | `build_docs.py` (pandas) | all | segment cards, data dictionary, findings, correlations (Section 6, Stage 1) |
| A3 | **arXiv papers** on churn prediction / retention | PDF + metadata | arXiv API (`export.arxiv.org/api/query`), then download PDFs and extract text with `pymupdf` | all (modeling, SHAP, class imbalance) | queries: "bank customer churn prediction", "customer attrition retail banking", "explainable churn SHAP bank", "complaints satisfaction churn banking". Cap at about 25 papers; store title, authors, year, arXiv id |
| A4 | **Wikipedia** concept pages | HTML | MediaWiki API (`action=query&prop=extracts&explaintext=1`) or REST `/page/html` | all | pages: Customer attrition, Churn rate, Customer retention, Customer lifetime value, Net promoter score, Customer satisfaction, Credit score, Credit card, Loyalty program, Retail banking, Cross-selling, Deposit insurance, Open banking, SCHUFA, Interchange fee, General Data Protection Regulation. CC BY-SA: cite and keep the URL |
| A5 | **ECB Data Portal API** (current rates) | API (SDMX CSV) | `https://data-api.ecb.europa.eu/service/data/{flow}/{key}?format=csvdata&lastNObservations=N` | Balance, Geography | the **up-to-date** component: key policy rates (flow `FM`), euro-area/country bank deposit and lending rates (flow `MIR`), inflation (flow `ICP`). Look up exact series keys on the portal; write each series as a short text note ("As of {date}, the ECB deposit facility rate is …") |
| A6 | **EU legal texts (EUR-Lex)** | HTML/PDF | `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{id}` | NumOfProducts, Card Type, CreditScore, Gender, Age | Deposit Guarantee Schemes Directive 2014/49/EU (32014L0049) · Payment Accounts Directive 2014/92/EU, **account switching**, which is directly about churn (32014L0092) · PSD2 (32015L2366) · Interchange Fee Regulation (32015R0751) · GDPR (32016R0679) · AI Act (32024R1689), credit scoring as high-risk · Consumer Credit Directive (32023L2225). **Risk:** EUR-Lex sometimes serves a bot-challenge page; test early and fall back to the PDF endpoint or the Publications Office (Cellar) API |

### Tier B: extended (B1, B2, B3, B5, B7, B8 required; B4, B6 optional)

| ID | Source | Type | Automation | Maps to columns | Notes |
|---|---|---|---|---|---|
| B1 | **Banco de España — Portal del Cliente Bancario** (`clientebancario.bde.es`) | HTML, **Spanish** | scrape guide pages (accounts, cards, switching, complaints procedure) | Complain, HasCrCard, Card Type | gives Spanish-language text to prove multilingual retrieval; also look for the annual complaints report ("Memoria del Servicio de Reclamaciones") for complaint statistics |
| B2 | **Eurostat API** | API JSON/CSV | `https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}` | EstimatedSalary, Age, IsActiveMember | country-level (FR/DE/ES) unemployment, inflation (HICP), income/earnings, and **internet banking usage** (find the dataset code on Eurostat; likely under `isoc_`). Written as short country notes |
| B3 | **World Bank API** | API JSON | `https://api.worldbank.org/v2/country/FRA;DEU;ESP/indicator/{id}?format=json` | Geography, Balance | indicators (verify ids): inflation `FP.CPI.TOTL.ZG`, GDP per capita `NY.GDP.PCAP.CD`, domestic credit to private sector `FS.AST.PRVT.GD.ZS`, account ownership (Global Findex) |
| B4 | **ECB press releases / monetary policy decisions** | HTML / RSS | ECB press RSS feed, then fetch articles | Balance | recent decisions on rates, time-stamped, so "what did the ECB decide most recently?" is answerable only with RAG |
| B5 | **ECB Household Finance and Consumption Survey (HFCS)** | PDF/HTML | download statistical tables / bulletins from ecb.europa.eu | Age, Balance | wealth and deposit holdings by age group and country |
| B6 | **EBA publications** (Consumer Trends Report, loan-origination and product-oversight guidelines) | PDF | download from eba.europa.eu | CreditScore, NumOfProducts, Complain | supervisory view on consumer treatment and complaints |
| B7 | **Credit reporting systems** (Banco de España CIRBE, SCHUFA, Banque de France FICP) | HTML | scrape the public explanatory pages | CreditScore, Geography | explains how "credit score" differs per country, so the same score isn't comparable across France, Germany and Spain |
| B8 | **France and Germany regulator/central-bank consumer pages** (Banque de France / ACPR médiation, BaFin Verbraucher, Bundesbank) | HTML, **French and German** | scrape selected pages | Geography, Complain | avoids a Spain-only bias; supports country-filtered retrieval |

### Out of scope (decided with the team)

- **CFPB US complaints database** and **Hugging Face `banking77`**: not included, to keep the corpus focused on European retail banking. Do not add them.

---

### 5.2 Repository layout (inside the team repo)

```
rag_project/
├── README.md
├── PROGRESS.md
├── RAG_PROJECT_PLAN.md
├── pyproject.toml              # metadata, dependencies and groups, ruff/mypy/pytest config
├── uv.lock                     # exact pinned versions (committed)
├── requirements.txt            # GENERATED by `uv export`, only for Hugging Face Spaces; never edit by hand
├── .python-version             # interpreter pin (uv)
├── .gitattributes              # `* text=auto eol=lf`: same line endings on Windows and Linux
├── .pre-commit-config.yaml
├── .env.example                # variable names only
├── config.yaml                 # paths, sources, chunk sizes, models, k values
├── data/
│   ├── SOURCES.md              # every source: URL, license, retrieval date
│   ├── raw/                    # git-ignored (cache)
│   └── processed/              # markdown docs + stats.json (small, committed)
├── data_pipeline/
│   ├── fetchers/               # one module per source: kaggle.py, arxiv.py, wikipedia.py, ecb.py, eurlex.py, bde.py, eurostat.py, worldbank.py ...
│   ├── clean.py                # normalization, de-duplication, PII drop
│   ├── build_docs.py           # pandas -> segment cards, data dictionary, findings, stats.json
│   └── run_all.py              # orchestrates everything, logs, retries, summary report
├── rag/
│   ├── chunking.py
│   ├── index.py                # build/update the Chroma collection
│   ├── retrievers.py           # vector, BM25, hybrid, rerank, filtered
│   ├── chain.py                # RAG chain + No-RAG baseline + router
│   └── prompts.py
├── eval/
│   ├── questions.json          # questions + gold answers + gold source ids
│   ├── generate_gold.py        # fills gold numbers from stats.json
│   ├── run_eval.py             # No-RAG vs RAG (vs advanced) -> results.csv
│   ├── score.py                # retrieval hit@k + exports a sheet for manual scoring
│   └── chunking_experiment.py
├── app/
│   ├── app.py                  # Gradio
│   └── README.md               # Hugging Face Space front matter lives in the Space repo
├── tests/
│   ├── unit/                   # fast, offline, deterministic (fakes/mocks, tiny fixtures)
│   ├── integration/            # marked `integration`: real network/LLM, excluded by default
│   └── fixtures/               # tiny CSV, saved API/HTML responses, sample PDFs text
├── deploy/                     # OPTIONAL: Dockerfile + compose.yaml, only if a container is truly needed
└── docs/                       # Quarto site
```

## 6. Stages

### Stage 0: Scaffold and environment
**Tasks**
- Check tools (`uv --version`, `python --version`). Manage the project with **uv**; never install globally.
- Create `pyproject.toml` (`uv init --bare` or by hand) with `[project]`, `requires-python`, dependency groups `dev` and `docs`, and tool configuration for ruff, mypy and pytest. Pin the interpreter with `uv python pin`.
- Create the virtualenv and install: `uv venv .venv`, add runtime dependencies with `uv add ...`, dev tools with `uv add --group dev pytest pytest-cov ruff mypy pre-commit poethepoet` (plus type stubs as needed), then `uv sync`. Commit `uv.lock`.
- Create the folder layout above and `.gitignore` (`.venv/`, `.env`, `data/raw/`, `chroma_db/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `docs/_site/`, `docs/.quarto/`, `logs/`).
- Add `config.yaml`, a **typed settings module** (pydantic-settings: YAML defaults + env overrides + secrets from `.env`), `.env.example`, `PROGRESS.md`, `.gitattributes`, `.pre-commit-config.yaml`, and the **`poe` tasks** in `pyproject.toml` (`pipeline`, `index`, `eval`, `app`, `test`, `test-integration`, `lint`, `typecheck`, `docs`, `export-reqs`), so the same commands work on Linux and Windows.
- Add a logging setup (console + `logs/pipeline.log`) used by every module (no `print`).
- Write the first version of the **README** (Section 11) with the setup instructions.
- Add smoke tests: settings load, each configured LLM provider answers one call, embedding model returns one vector (the last two marked `integration`). Add offline unit tests that the provider factory builds the right adapter for `gemini`, `groq` and `deepseek`, fails with a clear message when the selected provider's key is missing, and follows `fallback_order`.

**Acceptance:** from a clean clone, `uv sync` creates `.venv/` and installs everything; `uv run pytest` passes; `uv run ruff check .` and `uv run mypy .` are clean; the LLM and embedding smoke tests pass.

### Stage 1: Automated data pipeline
**Tasks**
1. One fetcher per source in Section 4 (all of Tier A and the required Tier B sources B1, B2, B3, B5, B7, B8; B4 and B6 if time allows). Each fetcher: retries with backoff, polite rate limit, raw cache in `data/raw/<source>/`, returns a list of documents with the Section 5.3 metadata, and logs failures **without crashing the whole run**.
2. `clean.py`: strip HTML boilerplate/navigation, normalize unicode and whitespace, convert to Markdown (keep headings), remove duplicates (hash), drop empty or tiny documents, **drop PII columns**.
3. `build_docs.py` (pandas) writes these documents automatically from A1:
   - **Data dictionary**: every column, type, range, meaning.
   - **Overview**: rows, overall churn rate, class balance.
   - **Segment cards**, one small Markdown document per segment, with count, churn rate, mean balance/salary/credit score, and a plain-language sentence. Segments: by Geography; Gender; Age band (<30, 30–39, 40–49, 50–59, 60+); Tenure band; NumOfProducts; Card Type; HasCrCard; IsActiveMember; Satisfaction Score; Complain; and **Geography × NumOfProducts** and **Geography × IsActiveMember**.
   - **Findings** document: top churn drivers (correlations, mutual information or a simple tree importance), the **Complain/Exited leakage check**, and the observation (if confirmed) that satisfaction score alone doesn't track churn.
   - `stats.json`: every number used in the documents, keyed by name (used later to generate evaluation gold answers).
4. `run_all.py`: runs fetch → clean → build_docs → writes `data/processed/*.md` and a **run report** (documents per source, failures, total words). `--sources a1,a3` flag to run subsets; `--refresh` to bypass the cache.
5. Write `data/SOURCES.md` automatically (URL, license, retrieval date per source).

**Acceptance**
- From a clean checkout with only `.env` set: `uv run python -m data_pipeline.run_all` produces `data/processed/` with documents from ≥ 10 distinct sources, ≥ 3 formats and ≥ 3 languages (EN/ES + at least one of FR/DE), no manual files.
- Running it twice gives identical output with no duplicates.
- A grep for any surname from the CSV in `data/processed/` returns nothing.
- The run report lists any failed source; a failed source never stops the others.

### Stage 2: Chunking, embeddings, vector DB
**Chunking strategy (decide per document type, and justify in the memoir):**

| Document type | Strategy | Reason |
|---|---|---|
| Segment cards, data dictionary, ECB/macro notes | **No splitting**: one document = one chunk (or one Markdown section) | already atomic and short; splitting would separate a number from its label |
| Wikipedia, regulator pages, laws, papers | `MarkdownHeaderTextSplitter` (keep header path as metadata) then `RecursiveCharacterTextSplitter` (starting point: 800 characters, 100 overlap; separators paragraph → sentence → word) | respects document structure; overlap protects sentences that cross a boundary |
| Optional | Semantic chunking on the longest papers only, as a comparison | shows awareness of the alternative |

**Tasks**
- `chunking.py` implements the above; `index.py` embeds and stores chunks in a **persistent Chroma collection** with full metadata; make it rebuildable (`--rebuild`) and incremental.
- `chunking_experiment.py`: build indexes at chunk sizes **300 / 600 / 1000** (overlap 10–15%), run the evaluation questions' retrieval only, and report **hit@k (k=1,3,5)** and **MRR** against `gold_source_ids`, plus average chunk count and context length. Save to `eval/chunking_results.csv` and a plot. This is the empirical justification for the chosen size.
- Pick the final size from the data **[DECISION: confirm with the team]**.

**Acceptance:** index builds from `data/processed/` in a few minutes; a query like "churn rate in Germany" returns the Geography/Germany card in the top 3; the chunking experiment table exists.

### Stage 3: Baseline vs. RAG and evaluation
**Question set (12 questions; aim for 10 + 2 behavioral checks).** Include the full question context so the baseline LLM is being asked about a specific thing, e.g. "In the Kaggle *Bank Customer Churn* dataset (10,000 customers, France/Germany/Spain), what is the churn rate of customers in Germany?"

| Cat. | # | Examples (gold values come from `stats.json`; do not type numbers by hand) |
|---|---|---|
| Dataset (D) | 5 | churn rate Germany vs France vs Spain · churn by number of products (especially 4) · how `Complain` relates to `Exited` · which Card Type has the highest churn / most points earned · which age band churns most |
| Literature (L) | 2 | pick two concrete findings from the ingested papers (model/metric used, key driver). Claude drafts, **a human verifies against the PDF** |
| Regulatory / current (R) | 3 | deposit guarantee level under Directive 2014/49/EU · the interchange-fee caps under Regulation 2015/751 (verify wording) · **today's** ECB deposit facility rate (from the A5 fetch, with date) |
| Behavioral (U) | 2 | an answer not in the corpus (e.g., churn rate in Portugal) → the system must say it doesn't know · "what is the surname of customer 15634602?" → refuses (PII removed) |

**Tasks**
- `eval/questions.json`: id, category, question, gold answer text, `gold_source_ids`, `gold_numbers` (filled by `generate_gold.py` from `stats.json`).
- `chain.py`: `answer_no_rag(question)` (plain LLM, same model/temperature) and `answer_rag(question, **opts)`.
  - RAG prompt: *answer only from the context; cite sources as [doc_id]; if the context is insufficient, say so; answer in the user's language.* Return the answer, the retrieved chunks (id, score, metadata) and latency.
- `run_eval.py`: runs every question in each mode, saves `eval/results.csv` and `eval/results.json` (answer, retrieved ids, latency, token counts if available).
- `score.py`: computes **retrieval hit@k** automatically and exports a sheet for the team to score manually: **0** = wrong/hallucinated, **1** = partly right or vague, **2** = correct and specific; plus "citation correct? Y/N" for RAG. Auto-check numeric answers against `gold_numbers` as a second opinion.
- Save a few clean before/after examples for the memoir (the question, the No-RAG answer, the RAG answer with its cited chunks).

**Acceptance:** `results.csv` exists for all questions; RAG scores higher than No-RAG on D, L and R categories; U questions are handled correctly. If the baseline already answers a question well, **replace that question** with a more niche one.

### Stage 4: Web app and deployment
**App spec (Gradio Blocks)**
- Chat tab with a **mode selector**: *No RAG* / *RAG* / *RAG advanced* (enabled in Stage 5). The interface labels, example-question buttons and About text are in **Spanish**; answers follow the language of the question.
- **Compare tab**: one question box, two columns (No RAG | RAG) answered side by side, which is the best live demo of the project's central claim.
- "Sources" accordion under each RAG answer: doc title, source, URL, similarity/rerank score, the retrieved chunk text.
- Example-question buttons (the evaluation questions); response time shown; clear error messages when the LLM is rate-limited (retry once, then show a friendly message).
- Filter controls (added in Stage 5): `source_type`, `geography`, `language`.
- Short "About" panel: what the data is, which sources, and the note "answers are grounded in the retrieved documents only".

**Deployment (Google Cloud Run, container) [DECISION 2026-11-22: replaces Hugging Face Spaces, which now requires a paid plan for Gradio/Docker Spaces]**
- `deploy/Dockerfile` (works with Docker and Podman): `uv sync --frozen --no-dev`, copies the code and the committed `data/processed/`, then runs `python -m rag.index` so the e5 model and the Chroma index are **baked into the image** (Cloud Run disk is ephemeral). The app also syncs the index at start (a no-op when it is current).
- The app reads `PORT` (Cloud Run sets it) and listens on `0.0.0.0`.
- API keys go in **Secret Manager** and are mounted as environment variables (`--set-secrets`), never in the repo or the image.
- Deploy from the Cloud Console (Cloud Run, continuous deployment from the GitHub repo) or Cloud Shell, so nobody needs a local `gcloud`; `deploy/cloudbuild.yaml` sets a 30-minute build timeout. Set **memory 2 GiB or more, 2 CPU, min instances 0, max instances 2** and a **budget alert** on the billing account.
- `requirements.txt` is no longer needed (`export-reqs` stays as a convenience).

**Acceptance:** public URL works from another device and a private browser window; both modes answer; sources are shown; no secrets in the repo; cold start completes in a reasonable time.

### Stage 5: Advanced RAG
Implement in this order and **measure each** with the same questions (ablation table: baseline → +hybrid → +rerank → +filters → +router):

1. **Hybrid search**: BM25 + vector with an ensemble retriever (start with 0.5/0.5 weights; tune). Helps exact terms such as directive numbers, "DIAMOND" or "NumOfProducts".
2. **Reranking**: retrieve top-20 (hybrid), rerank with FlashRank, pass the top-4 to the LLM.
3. **Metadata filters**: Chroma `where` filters on `source_type`, `geography`, `language`, `topic`. Expose in the UI. Optionally add simple rule-based extraction (question mentions "Germany" → `geography in [DE, EU]`).
4. **Adaptive/modular RAG**: a small router prompt (or rules) decides `retrieve` vs `answer directly` (e.g., chit-chat, definitions) and picks the retriever (dataset questions → `source_type=dataset_stats`).

**Acceptance:** `eval/ablation.csv` with hit@k and manual-score means per configuration; the UI exposes the advanced mode; there is at least one clear example where reranking or filtering fixed a failure.

### Stage 6: Quarto memoir and publication
**Site structure (`rag_project/docs/`)**

| Page | Content |
|---|---|
| `index.qmd` | summary, team, links to **app**, **repo**, and Loop |
| `01-motivacion-datos.qmd` | why churn/banking, link to project 1, dataset description and column table, the source catalog table (A/B/C) with licenses, ethical note (PII drop, synthetic data, leakage of `Complain`) |
| `02-pipeline.qmd` | automation diagram, fetcher design (retries, cache, idempotence), cleaning, how segment cards are generated; explained code excerpts |
| `03-chunking-vectordb.qmd` | chunking strategy per document type, **chunk-size experiment** (table + plot), why Chroma, embedding choice |
| `04-evaluacion.qmd` | question table, No-RAG vs RAG results (read live from `eval/results.csv` in a Python cell), clean before/after examples with screenshots, scoring summary |
| `05-rag-avanzado.qmd` | hybrid, reranking, filters, router, ablation table |
| `06-app-despliegue.qmd` | app screenshots, deployment steps, link |
| `07-equipo-referencias.qmd` | who did what, sources and references |

- Use `freeze: auto`; code cells display short, pedagogical excerpts (don't dump whole files).
- **Publish as a static site, via a GitHub Actions workflow** (repo-root `.github/workflows/rag_project-docs.yml`, path-filtered to `rag_project/**`). Support both options and document them in the README; the team picks one:
  - **GitHub Pages** (default): the workflow runs `quarto render rag_project/docs` and deploys `docs/_site` to Pages (or use `quarto publish gh-pages`).
  - **Cloudflare Pages**: connect the repo, root directory `rag_project/docs`, build command `quarto render`, output directory `_site` (Quarto must be available in the build; if not, build in the GitHub Action and deploy the output with Wrangler). Needs the Cloudflare account token as a repo secret.
  Verify all links, then **post the URL in Loop (Teams)**.

**Acceptance:** public site renders without errors; every required section from Section 1 is present; links to app and repo work.

### Stage 7: Freeze, QA, rehearsal
- Feature freeze. Test from a clean clone **following only the README**: `uv sync` → pipeline → index → evaluation → app locally → tests and linters. Fix any README step that fails.
- Check the Space is awake; prepare a backup demo (recorded video and screenshots) in case of rate limits or cold start.
- Rehearse the oral defense (below); every member must be able to answer technical questions on their part.

### Presentation — Fri 27 Nov
Four parts, one per member:
1. Data capture and pipeline (A) · 2. Vector DB, chunking and libraries (B) · 3. UI and deployment (D) · 4. Evaluation and advanced techniques (C, E).

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Kaggle download needs credentials in automation | test in Stage 0; env vars; fallback URL in `config.yaml` |
| A source blocks scraping (EUR-Lex challenge page, ECB, regulator sites) | polite crawling, alternate endpoint (PDF/Cellar/API), skip and log; the run never aborts |
| Free LLM tier rate limits during the demo | caching of eval answers, second provider, backup recorded demo |
| Baseline LLM already knows the answer | make questions specific to computed numbers, the latest ECB data, or the dataset itself; replace weak questions |
| RAG fails on aggregate questions | segment cards and `stats.json` approach (Section 2), not raw rows |
| Cloud Run cold start, memory or cost | small embedding model, committed processed docs, index baked into the image, 2 GiB memory, budget alert, `--min-instances 1` for the live demo |
| Licensing/copyright of scraped PDFs | list licenses in `SOURCES.md`; cite sources in answers; for papers keep only what the project needs and don't republish full PDFs |
| Hallucinated gold answers | gold numbers come from `stats.json`; literature and regulatory golds are verified by a human against the source |

---

## 9. Final checklist (maps to the grading criteria)

- [ ] **Ingestion robustness:** one command rebuilds the corpus; ≥ 10 sources, ≥ 3 formats, ≥ 3 languages; idempotent; failures logged, not fatal; PII dropped
- [ ] **Technical justification:** Chroma choice explained; chunking per document type, with the **chunk-size experiment** results
- [ ] **Comparison clarity:** ≥ 10 questions, No-RAG vs RAG table, manual scores, cited chunks, at least one correct "I don't know" and one PII refusal
- [ ] **UI and deployment:** public Gradio app, compare tab, sources panel, no secrets in repo
- [ ] **Documentation:** Quarto site published (all required sections), URL posted in Loop, repo organized with README and pinned requirements
- [ ] **Advanced RAG (extra):** hybrid + rerank + filters (+ router) with an ablation table
- [ ] **Engineering quality:** `.venv` + uv with committed `uv.lock`; typed, documented code; design patterns where justified; `ruff`, `mypy` and `pytest` all green; README reproduces everything from a clean clone
- [ ] **Oral defense:** each member can explain their part and run a live example

---

## 11. README specification (`rag_project/README.md`)

The README must let someone with a clean machine reproduce the project **using only its instructions**. Required sections:

1. **Overview**: what the project is, the subtopic, and an architecture diagram (Mermaid) of pipeline → index → retrieval → LLM → UI.
2. **Links**: deployed app, Quarto site, GitHub repo.
3. **Prerequisites** (Linux and Windows): Python version (from `.python-version`), git, Quarto (for the docs), and **uv** with both install commands: Linux `curl -LsSf https://astral.sh/uv/install.sh | sh`; Windows PowerShell `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` (verify against the current uv docs). *Optional:* Docker or Podman, only if `deploy/` exists.
4. **Setup**, with **two tabbed/side-by-side command sets, Linux (bash) and Windows (PowerShell)**:
   ```
   # Linux
   git clone <repo> && cd <repo>/rag_project
   uv sync                      # creates .venv and installs locked dependencies
   cp .env.example .env         # then fill in the keys

   # Windows PowerShell
   git clone <repo>; cd <repo>\rag_project
   uv sync
   Copy-Item .env.example .env
   ```
   Activating `.venv` is optional because `uv run` uses it (`source .venv/bin/activate` on Linux, `.venv\Scripts\Activate.ps1` on Windows; if PowerShell blocks scripts, note `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`). Fallback without uv: `python -m venv .venv`, activate it, `pip install -e ".[dev]"`; never outside `.venv`.
5. **Configuration**: table of every `.env` variable (name, purpose, where to obtain it: Kaggle token, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, Hugging Face token) with a note that only the selected LLM provider's key is needed, how to switch provider (`llm.provider` / `LLM_PROVIDER`), and the main `config.yaml` options (sources, chunk size, models, k).
6. **Run the data pipeline**: `uv run python -m data_pipeline.run_all` (+ flags: `--sources`, `--refresh`) and what it produces.
7. **Build the index**: `uv run python -m rag.index --rebuild`.
8. **Run the evaluation**: `uv run python -m eval.run_eval`, scoring, chunk-size experiment, and where outputs are stored.
9. **Run the app locally**: `uv run python app/app.py` and the local URL.
10. **Tests and quality**: `uv run pytest`, `uv run pytest -m integration`, `uv run ruff check .`, `uv run mypy .`, pre-commit install.
11. **Task shortcuts** (poethepoet, identical on Linux and Windows): `uv run poe pipeline | index | eval | app | test | test-integration | lint | typecheck | docs | export-reqs`.
12. **Deployment**: Cloud Run steps, secrets and budget alert; building the image with Docker and Podman.
13. **Quarto docs**: preview (`quarto preview docs`) and publish.
14. **Project structure** (tree with one-line descriptions) and the design patterns used.
15. **Troubleshooting**: Kaggle authentication, LLM rate limits, EUR-Lex blocking, empty index, Space cold start.
16. **Data attribution and licenses**, and the team with roles.

Claude Code must **run every README command from a clean clone** at the end of Stage 7 and fix whatever does not work.
