# RAG Project Plan — Bank Customer Churn (for implementation with Claude Code)

> **Presentation:** Fri 27 Nov 2026. The stages below have **no calendar dates on purpose**: implement them back-to-back as fast as quality allows, starting each one as soon as the previous one passes its acceptance criteria.
> **Goal:** a complete, automated, deployed RAG system that demonstrably beats the same LLM without retrieval, on the banking / customer-churn domain of project 1 (Kaggle "Bank Customer Churn").

---

> Section 3 (dataset facts), 4 (source catalog), 5.2 (repo layout), 6 (stage details), 8 (risks), 9 (checklist) and 11 (README spec) live in `RAG_PLAN_REFERENCE.md`. Read the relevant part on demand (the matching Stage in section 6 before starting a stage).

## 0. How Claude Code should use this file

1. Read the whole file once. Work **stage by stage** (Section 6). Do not start a stage before the previous one meets its **Acceptance criteria**.
2. Keep a running log in `rag_project/PROGRESS.md` (stage, what was done, what is pending, decisions taken, open issues).
3. **Git workflow: one branch per stage and a pull request for the team to review.** Create `stage-<n>-<short-name>` from an up-to-date `main`, make small commits with clear messages (at least one per sub-stage), push, and open a PR with `gh pr create` that lists what was done, how it was verified (commands and results) and any open issue. **Never merge your own PR and never push to `main`**; stop and wait for the team's review before starting the next stage (if the next stage depends on an unmerged PR, branch from that stage's branch and say so in the PR). Never commit secrets (`.env`, API keys, tokens). Keep `.env.example` with variable names only.
4. Ask the human only for things you cannot do yourself: credentials (Kaggle, LLM provider, GitHub), and choices marked **[DECISION]**. The evaluation is scored automatically (no manual scoring).
5. Before writing code that touches an external API, check the **installed library version** and the **current docs** (LangChain import paths and model names change often). Versions are pinned by `uv.lock`.
6. Prefer small, testable modules over notebooks. Every script must run from the `rag_project/` folder with `uv run python -m ...` and must be **idempotent** (re-running does not duplicate or corrupt data).
7. Source URLs and API series keys in Section 4 were not all tested. Treat each as **"verify before relying on it"**; log any that fail in `PROGRESS.md` and substitute an equivalent.
8. **Environment:** use **uv** and the project virtualenv **`.venv/`** (Section 5.1). Never install packages globally or with a bare `pip install` outside `.venv`.
9. **Engineering standards in Section 10 are mandatory** (design patterns where justified, typed and documented code, tests, linting) and the **README must stay accurate** (Section 11): update it at the end of every stage.
10. **Project isolation:** the RAG project is a **self-contained project inside `rag_project/`**, with its own `pyproject.toml`, `uv.lock` and `.venv`. It must **not import from, depend on, or modify** the first project (the churn prediction model) or any file outside `rag_project/`, with one unavoidable exception: GitHub only runs workflows from the repo-root `.github/workflows/`, so any CI or Pages workflow lives there, is named `rag_project-*.yml` and is **path-filtered to `rag_project/**`**. The first project's model is not reused (this project does no churn prediction); it is only referenced in the memoir's narrative and with a link.

**Kick-off prompt to give Claude Code:**
> Read `RAG_PROJECT_PLAN.md`. Implement Stage 0 and Stage 1 exactly as specified, run the acceptance checks, update `PROGRESS.md`, and open a pull request for each stage as described in rule 3, then stop for my review.

---

## 1. Project constraints (from the assignment)

| Phase | Requirement |
|---|---|
| A | Corpus on a **subtopic** of our theme; mix of public sources; **fully automated** download/clean/prepare in Python (no manual file loading) |
| B | RAG framework (LangChain / LlamaIndex / Haystack) + vector DB + **technically justified chunking** (explain chunk size vs. context/retrieval) |
| C | 5–10+ niche questions a generic LLM can't answer; side-by-side **Without RAG vs. With RAG**, documented |
| D | Interactive web app (Streamlit or Gradio), **publicly deployed** |
| E (extra) | Reranking, metadata filters, adaptive RAG, hybrid BM25 + vector |
| Docs | Code in GitHub folder `/rag_project`; **Quarto** website (motivation + provenance, pipeline + chunking, evaluation, links to app and repo, explained code excerpts), published online (GitHub Pages or Cloudflare Pages / Quarto Pub) |
| Grading | Ingestion robustness · justification of vector DB and chunking · clarity of comparison · UI quality and public deployment · Quarto/GitHub quality · individual technical mastery |

---

## 2. Subtopic and key design decision

**Subtopic:** *Customer retention and churn in European retail banking (France, Germany, Spain)*: churn drivers, what the bank's own customer data shows, and the regulatory and market context around it.

**Key design decision — turn the CSV into knowledge, don't embed rows.**
RAG cannot answer aggregate questions ("churn rate in Germany?") from 10,000 individual row-chunks. The pipeline must therefore **compute statistics with pandas and write them as text documents** (segment cards, data dictionary, findings). Those documents are the specialized knowledge the base LLM does not have. Every number in them is computed by code, so the evaluation ground truth is also generated, not typed by hand.

**Privacy rule:** `Surname`, `CustomerId` and `RowNumber` are personal identifiers (even if synthetic). They are **dropped before any document is created** and never indexed. One evaluation question will check that the system declines to reveal them.

**Project isolation:** this project is independent of the first project (the churn prediction model). It has its own environment and folder, shares no code with it, and trains no model. The dataset is the only link between the two.

---

## 5. Architecture and conventions

### 5.1 Stack **(confirmed: use exactly these)**

| Component | Default | Why |
|---|---|---|
| Orchestration | **LangChain** (pin version; check import paths, since several retrievers moved to `langchain_classic` in 1.x) | has BM25, ensemble, FlashRank and Chroma integrations |
| Vector DB | **ChromaDB** (persistent, local) | supports metadata filtering, easy to ship inside a container, no server. Be ready to explain the trade-offs against FAISS (no native metadata filters) and Qdrant/Milvus/Pinecone (extra infrastructure) |
| Embeddings | `intfloat/multilingual-e5-small` via `sentence-transformers` (remember `"query: "` / `"passage: "` prefixes) | multilingual (EN/ES/FR/DE), small enough for free CPU hosting |
| LLM | **Selectable provider: Google Gemini (default), Groq or DeepSeek.** Choose with `llm.provider: gemini \| groq \| deepseek` in `config.yaml` or the `LLM_PROVIDER` env var; an optional `llm.fallback_order` (e.g. `[gemini, groq, deepseek]`) is tried when the active provider is rate-limited or fails. Packages: `langchain-google-genai`, `langchain-groq`, and for DeepSeek `langchain-deepseek` (or `ChatOpenAI` with DeepSeek's OpenAI-compatible `base_url`). Keys: `GOOGLE_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`; only the selected provider's key is required. Same model and `temperature=0` for both "No RAG" and "RAG", so the comparison is fair | verify current model names and limits. Gemini and Groq have free tiers; **DeepSeek's API is pay-as-you-go (no free tier, low cost)**. Because the evaluation and the live demo make many calls, add client-side throttling, retry with backoff, and an on-disk cache of evaluation answers (cache key includes provider and model) |
| Reranker | **FlashRank** (local, no API) | no extra cost or key |
| Lexical search | `rank_bm25` via LangChain `BM25Retriever` | hybrid search |
| UI | **Gradio** (Blocks) | simple Python UI that runs in one container |
| Docs | **Quarto**, published as a static site on **GitHub Pages or Cloudflare Pages** (team choice, see Stage 6) | required |
| App hosting | **Google Cloud Run** (container) | scales to zero and has a monthly free quota (needs a billing account: set a budget alert). Cloudflare Pages is static-only, so it is only suitable for the Quarto site |
| Package / env management | **uv** (`pyproject.toml` + committed `uv.lock`), virtualenv in **`.venv/`** | lockfile for reproducible installs, very fast, native dependency groups (`dev`, `docs`), and `uv sync --frozen` gives the container the exact same install. `pip` is only a fallback and **only inside `.venv`** |
| Containers | **Docker or Podman, required for the app** (Cloud Run runs containers). `deploy/Dockerfile` bakes the e5 model and the Chroma index into the image; `deploy/cloudbuild.yaml` is the Cloud Build config. Keep it free of Docker-only features so `podman build` also works | a compose file is only needed if a separate server (Chroma/Qdrant) ever appears |
| Quality tooling | `ruff` (lint + format), `mypy`, `pytest` + `pytest-cov`, `pre-commit`, **`poethepoet`** (cross-platform task runner, replaces `make`) | see Section 10 |
| Language | corpus is EN + ES + FR + DE; the app answers in the language of the question; the **Quarto site and the app interface labels are in Spanish (confirmed)** | |

### 5.3 Document and chunk metadata (mandatory)

```json
{
  "doc_id": "wiki_customer_attrition",
  "title": "Customer attrition",
  "source": "wikipedia | arxiv | ecb_api | eurlex | bde | eurostat | worldbank | kaggle_derived | ...",
  "source_type": "dataset_stats | paper | encyclopedia | regulation | macro_data | regulator_guide | press",
  "url": "https://...",
  "language": "en | es | fr | de",
  "geography": "FR | DE | ES | EU | global | US",
  "topic": "churn | credit_score | cards | deposits_rates | complaints | privacy_ai | switching | ...",
  "columns": ["Geography", "Exited"],
  "date": "YYYY-MM-DD",
  "retrieved_at": "YYYY-MM-DD",
  "license": "CC BY-SA | public domain | ..."
}
```

`columns` is a list linking a document to dataset columns (it makes metadata filters and the Quarto write-up easier).

---

## 7. Execution order

Run **Stage 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7** in order, each as soon as the previous one passes its acceptance criteria. Teammates can draft the Quarto pages (Stage 6) in parallel while Stages 4 and 5 are being built, since the structure is known.

Cut order if time runs short (drop from the bottom of this list first): optional sources B4/B6 → semantic chunking → adaptive router → metadata-filter UI → individual Tier B sources (keep at least B1, B2 and B7).
**Never cut:** automated pipeline, chunk-size experiment, evaluation table, deployed app, Quarto site, tests, README.

---

## 10. Engineering standards (mandatory)

### 10.1 Design principles and patterns
Use a pattern **only where there are at least two interchangeable implementations or a real need** (no over-engineering). Expected uses:

| Pattern | Where |
|---|---|
| **Strategy** | `Chunker` (passthrough, markdown+recursive, fixed-size, semantic), `Retriever` (vector, BM25, hybrid), `Reranker` (none, FlashRank): chosen from `config.yaml` |
| **Factory / Registry** | `@register_fetcher("arxiv")` registry and `build_chunker(cfg)`, `build_retriever(cfg)`: adding a source or strategy means adding one class, not editing the orchestrator (open/closed) |
| **Template Method** | `BaseFetcher.fetch()` implements cache, retry, rate limit and logging once; subclasses only implement `_download()` and `_parse()` |
| **Adapter + Factory** | `LLMClient` and `Embedder` protocols with provider adapters (Gemini, Groq, DeepSeek, local HF) built by `build_llm(settings)`, so switching providers touches only configuration; a small `FallbackLLM` decorator tries the next provider in `fallback_order` on rate-limit or connection errors |
| **Decorator** | retry/backoff, rate limiting and disk-cache wrappers, reusable across fetchers |
| **Pipeline / chain of steps** | cleaning (`strip_html → normalize → drop_pii → dedupe → min_length`) as an ordered list of small pure functions |
| **Repository** | `VectorStoreRepository` wraps Chroma behind `add / query / delete / rebuild`, so the rest of the code never imports Chroma directly |
| **Dependency injection + composition root** | components are passed in through constructors; a single `factory.py` wires everything from settings; **no module-level globals or singletons** holding clients |

Also: single responsibility per module, small functions, **pure functions for statistics** (easy to test), domain models as typed `dataclass`/pydantic classes (`Document`, `Chunk`, `RetrievedChunk`, `RagAnswer`), custom exceptions (`FetchError`, `IndexingError`, ...), no magic numbers or hard-coded paths (everything from settings), and no `print` (use `logging`).

### 10.2 Code quality
- **Type hints** on all public functions; `mypy` clean on the packages.
- **Docstrings** (Google style) on every module, class and public function: purpose, args, returns, raises. Add short comments only for non-obvious decisions (the *why*).
- `ruff check` and `ruff format` clean; configured in `pyproject.toml`; enforced by `pre-commit`.
- Secrets only from environment/`.env`; never logged.
- Errors are handled at boundaries (network, files, LLM); a failing source or provider degrades gracefully with a clear log message.

### 10.3 Testing (pytest)
- **Unit tests** (`tests/unit`, offline, fast, deterministic): fakes instead of real services (a `FakeLLM`, a deterministic `FakeEmbedder`, saved HTTP responses in `tests/fixtures`). Cover at least:
  - cleaning: PII columns removed, duplicates dropped, HTML stripped
  - `build_docs`: statistics checked against a **tiny fixture CSV with hand-calculated values**
  - chunkers: sizes, overlap, header metadata preserved, atomic documents not split
  - fetcher base class: cache hit, retry on failure, a failing source does not stop the run
  - retrievers: hybrid fusion and rerank ordering with fake data, metadata filters applied
  - chain: context is injected into the prompt; when no context is retrieved the answer says it does not know; PII request is refused
  - eval scoring: hit@k and MRR computed correctly
  - settings: invalid config fails with a clear message
- **Integration tests** (`tests/integration`, `@pytest.mark.integration`): real downloads, embeddings and LLM calls. Excluded by default (`-m "not integration"`); run explicitly with `uv run pytest -m integration`.
- **App smoke test:** the Gradio app builds and its handler function returns an answer using fakes.
- Target **≥ 80% coverage** on `data_pipeline/` and `rag/` (`pytest-cov`). Tests must pass offline.
- Optional: a GitHub Actions workflow running `ruff`, `mypy` and the unit tests on every push.

### 10.4 Cross-platform rules (team uses Linux and Windows)
- Use `pathlib.Path` for every path; never hard-code `/` or `\`, and never build shell commands as strings.
- Always pass `encoding="utf-8"` when reading or writing text (Spanish, French and German accents break on Windows defaults otherwise); keep file names ASCII and short (Windows path-length limits).
- No shell-specific scripts (`.sh`/`.ps1`/`make`): all tasks go through `uv run poe ...` or Python entry points. No symlinks.
- Protect entry points with `if __name__ == "__main__":` (Windows multiprocessing), and avoid fork-only features.
- `.gitattributes` with `* text=auto eol=lf` so line endings don't cause diffs or broken scripts.
- Dependencies must install on both platforms from `uv.lock` (CPU-only builds; no CUDA requirement). If a dependency is Linux-only, find an alternative.
- Optional CI: run `ruff`, `mypy` and the unit tests on a matrix of `ubuntu-latest` and `windows-latest`.

---
