# 05. Ingeniería: patrones de diseño, configuración, tests y calidad

La rúbrica incluye "dominio técnico individual" y el plan exige patrones de diseño **solo donde haya al menos dos implementaciones intercambiables o una necesidad real** (nada de sobreingeniería). Esta guía indica dónde se usa cada patrón y cómo justificarlo.

## 1. Patrones de diseño usados y dónde

| Patrón | Dónde está | Por qué se usa (la justificación) |
|---|---|---|
| **Strategy** | `Chunker` (`PassthroughChunker`, `MarkdownRecursiveChunker`, `FixedSizeChunker`) en `rag/chunking.py`; `Retriever` (`VectorRetriever`, y en la Etapa 5 híbrido y reranking) en `rag/retrievers.py` | Hay varias formas válidas de trocear y de recuperar, y se comparan experimentalmente; se elige desde `config.yaml` |
| **Factory / Registry** | `@register_fetcher("a4")` y `FETCHER_REGISTRY`; `build_chunker(cfg)`; `build_llm(settings)`; `build_embedder(settings)` | Añadir una fuente = una clase nueva + un bloque de configuración, sin editar el orquestador (abierto/cerrado) |
| **Template Method** | `BaseFetcher.fetch()` en `data_pipeline/fetchers/base.py` | Caché, reintentos, aislamiento de errores y logging se escriben **una sola vez**; las subclases solo implementan `_items`, `_download` y `_parse` |
| **Adapter** | `GeminiAdapter`, `GroqAdapter`, `DeepSeekAdapter` en `rag/llm.py`; `E5Embedder` en `rag/embeddings.py` | Cambiar de proveedor toca solo la configuración; el resto del código usa el protocolo `LLMClient` |
| **Decorator** | `ThrottledLLM`, `RetryingLLM`, `FallbackLLM`; `RateLimiter` y `call_with_retry` en `resilience.py` | Añaden comportamiento (límite de ritmo, reintentos, alternativa) sin modificar la clase envuelta, y se apilan |
| **Pipeline (cadena de pasos)** | `DEFAULT_STEPS` en `clean.py` (`html_step`, `normalize_step`, `dedupe_step`, `min_length_step`) | Pasos pequeños, puros y reordenables; cada uno se prueba por separado |
| **Repository** | `VectorStoreRepository` en `rag/vectorstore.py` | El resto del código no depende de Chroma; si se cambiase de base vectorial, solo cambia esta clase |
| **Inyección de dependencias + raíz de composición** | `factory.py` (`Container`) | Los componentes reciben sus colaboradores por el constructor; **un solo sitio** los conecta desde la configuración; **no hay variables globales ni singletons** con clientes |

Qué **no** se hizo (a propósito): no hay jerarquías de herencia profundas ni interfaces para cosas con una sola implementación. Por ejemplo, `Retriever` ya es un protocolo pero hoy solo tiene `VectorRetriever`; se justifica porque en la Etapa 5 tendrá varias.

### Protocolos en vez de clases base abstractas
`LLMClient`, `Embedder`, `Chunker` y `Retriever` son `typing.Protocol`: cualquier objeto con los métodos adecuados vale (por eso los tests usan *fakes* simples sin heredar de nada).

## 2. Configuración tipada (`core/settings.py`)

- Una clase `Settings` de **pydantic-settings** con subsecciones tipadas (`PathsSettings`, `HttpSettings`, `LLMSettings`, `EmbeddingSettings`, `ChunkingSettings`, `IndexSettings`, `RetrievalSettings`).
- **Prioridad** (de mayor a menor): variables de entorno, archivo `.env`, `config.yaml`. Se implementa en `settings_customise_sources`.
- Atajo: la variable `LLM_PROVIDER` sustituye a `llm.provider`. Valores anidados con doble guion bajo: `LLM__MODELS__GROQ=...`.
- **Secretos** (`GOOGLE_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, claves de Kaggle y Hugging Face) solo salen del entorno o `.env`, son `SecretStr` y nunca se registran. `.env.example` contiene solo los **nombres**.
- **Validación con mensaje claro**: una configuración inválida lanza `ConfigError` indicando el campo (por ejemplo, intervalo mínimo de red menor que 1 s, solapamiento mayor o igual que el tamaño, proveedor desconocido).
- `load_settings(path)` carga y valida; `Settings.resolve(path)` convierte rutas relativas en absolutas respecto a la raíz del proyecto (sin rutas "a mano").

## 3. Modelos de dominio (`core/models.py`)

Clases `pydantic` con tipos estrictos:
- `DocumentMetadata` (los campos de la sección 5.3 del plan; `language`, `geography` y `source_type` son valores permitidos fijos, así un dato mal escrito falla al instante).
- `Document` (texto + metadatos), `Chunk` (con `embedding_text`), `RetrievedChunk` (con puntuación), `RagAnswer` (respuesta con evidencias y latencia; serializable, por eso puede ir a la caché).

## 4. Excepciones propias (`core/exceptions.py`)

Jerarquía bajo `RagProjectError`: `ConfigError`, `FetchError`, `IndexingError`, `ProviderError` (con `MissingCredentialError` y `RateLimitError`). Los errores se gestionan **en las fronteras** (red, ficheros, LLM): una fuente o proveedor que falla se degrada con un mensaje claro, no tira todo abajo.

## 5. Logging

`core/logging_setup.py`: `setup_logging()` crea manejadores de consola y de archivo rotativo (`logs/pipeline.log`); `get_logger(__name__)` lo usan todos los módulos. **No hay `print` en el código.**

## 6. Tests

| Tipo | Dónde | Características |
|---|---|---|
| **Unitarios** | `tests/unit/` | Sin red ni claves; rápidos y deterministas; **184 tests**; cobertura de aproximadamente 96 % en `core`, `rag` y `data_pipeline` (objetivo del plan: 80 %) |
| **Integración** | `tests/integration/` | Marcados `@pytest.mark.integration`; excluidos por defecto; usan red, modelo de embeddings y LLM reales (por ejemplo: cada fuente devuelve documentos, el índice acierta consultas en ES y DE, Gemini responde) |

Cómo se prueba sin red ni claves:
- `httpx.MockTransport` simula respuestas HTTP (reintento tras un 500, `robots.txt` que prohíbe, caché).
- `FakeLLM` y `FakeEmbedder` (una bolsa de palabras con hash, determinista) en `tests/fakes.py`.
- **Respuestas reales guardadas** como *fixtures* (arXiv, Wikipedia, BCE, Eurostat, Banco Mundial) para probar los analizadores.
- Un **CSV diminuto de 12 filas con valores calculados a mano** para verificar las estadísticas de `build_docs`.
- Los PDFs de prueba se generan en el propio test con `pymupdf`.

Qué se cubre (resumen): limpieza (PII fuera, duplicados, HTML), estadísticas, chunkers (tamaños, solapamiento, rutas de encabezados, atómicos sin partir), clase base de fetchers (caché, reintento, un fallo no detiene el resto), repositorio e indexador (incremental, rebuild, filtros), cadena (contexto en el prompt, sin contexto = "no sé", PII rechazada, citas), puntuación (hit@k, MRR, números), configuración (errores claros) y decoradores del LLM.

## 7. Calidad de código

- **Tipos**: `mypy` en modo estricto, limpio (con excepciones documentadas solo para librerías sin tipos como `pymupdf`).
- **Estilo y errores**: `ruff` (lint y formato, reglas E, F, I, B, UP, D, SIM) con docstrings estilo Google en módulos, clases y funciones públicas.
- **pre-commit** con los hooks de ruff (mypy se ejecuta con `poe typecheck`, necesita la carpeta del proyecto).
- **Tareas multiplataforma** con `poethepoet`: `uv run poe pipeline | test | test-integration | lint | typecheck | export-reqs` (más `index`, `eval`, `app` y `docs` definidos).

## 8. Reproducibilidad y multiplataforma

- **uv**: `uv sync` crea `.venv/` desde `uv.lock` (versiones exactas). **Python 3.12** fijado en `.python-version`.
- PyTorch **solo CPU** (índice de PyTorch declarado en `pyproject.toml`) para que instale igual en Linux y Windows y en Hugging Face Spaces.
- `pathlib.Path` para todo, `encoding="utf-8"` siempre (acentos en ES/FR/DE), escritura de archivos en **bytes** (sin traducción de saltos de línea en Windows), `.gitattributes` con `eol=lf`, sin scripts de shell ni enlaces simbólicos.
- Todos los scripts se ejecutan con `uv run python -m ...` desde `rag_project/` y son **idempotentes**.

## 9. Flujo de trabajo con git

- **Una rama y un pull request por etapa** (`stage-0-scaffold`, `stage-1-pipeline`, `stage-2-chunking-index`, `stage-3-evaluation`), commits pequeños con mensaje claro, y PR con qué se hizo, cómo se verificó (comandos y resultados) y problemas abiertos.
- Nunca se sube a `main` directamente y nadie fusiona su propio PR salvo instrucción expresa del usuario.
- Nunca se versionan secretos (`.env` está en `.gitignore`).
- Lección aprendida: al apilar PRs, si la rama base se fusiona primero, el PR siguiente debe **reorientarse a `main`** o se fusiona en la rama equivocada (pasó con la Etapa 1 y se resolvió con un PR de traspaso).
- Archivos de seguimiento: `PROGRESS.md` (diario) y `README.md` (se actualizan al final de cada etapa).

## 10. Preguntas probables

- *¿Qué patrones usasteis y por qué?* Tabla de la sección 1; recordar que se usan solo con motivo (por ejemplo, Strategy en el chunking porque comparamos varias estrategias).
- *¿Cómo cambiaríais de Gemini a otro proveedor?* Cambiando `llm.provider` o `LLM_PROVIDER`; el resto del código no cambia (Adapter + Factory).
- *¿Cómo probáis sin gastar llamadas ni red?* Fakes, `MockTransport` y respuestas reales guardadas.
- *¿Qué es la raíz de composición?* `factory.py`: el único lugar que conecta componentes; evita globales.
- *¿Cómo evitáis filtrar claves?* `SecretStr`, `.env` fuera de git, `.env.example` solo con nombres.
