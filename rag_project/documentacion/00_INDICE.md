# Guía de preparación para la presentación: proyecto RAG de churn bancario

> Estado de esta guía: **20 de noviembre de 2026**, con las Etapas 0, 1 y 2 terminadas y fusionadas en `main`, y la Etapa 3 (evaluación) terminada en código en la rama `stage-3-evaluation` (pendiente de revisión y de la puntuación manual). Las Etapas 4 a 7 (app web, RAG avanzado, sitio Quarto, ensayo final) **todavía no existen**; en esta guía aparecen marcadas como *pendiente* para que nadie las presente como hechas.
> Presentación: viernes 27 de noviembre de 2026.

## Para qué sirve esta carpeta

Es un manual para que **cada miembro del equipo pueda explicar y defender** cualquier parte del proyecto, no solo la suya. La nota incluye "dominio técnico individual", así que la guía explica el *porqué* de cada decisión, no solo el *qué*.

Cada archivo es independiente, pero conviene leerlos en orden la primera vez.

| Archivo | Contenido | Quién debería dominarlo (según el reparto del plan) |
|---|---|---|
| [01_vision_general_y_arquitectura.md](01_vision_general_y_arquitectura.md) | Qué es el proyecto, la idea central, diagrama de arquitectura, mapa de carpetas, stack y flujo completo de una pregunta | Todos |
| [02_pipeline_de_datos.md](02_pipeline_de_datos.md) | Cómo se descargan, limpian y convierten en conocimiento las 13 fuentes; clases y métodos clave; privacidad | Parte 1: captura de datos y pipeline |
| [03_chunking_embeddings_vectordb.md](03_chunking_embeddings_vectordb.md) | Estrategia de chunking por tipo de documento, experimento de tamaño, embeddings e5, ChromaDB e indexador incremental | Parte 2: base vectorial, chunking y librerías |
| [04_rag_llm_y_evaluacion.md](04_rag_llm_y_evaluacion.md) | Capa de LLM, prompts, cadena RAG vs No-RAG, evaluación, resultados y hallazgos honestos | Parte 4: evaluación y técnicas avanzadas (y Parte 3 para la futura app) |
| [05_ingenieria_patrones_y_calidad.md](05_ingenieria_patrones_y_calidad.md) | Patrones de diseño usados y dónde, configuración, tests, calidad, flujo de git | Todos (sirve para "dominio técnico") |
| [06_decisiones_y_lecciones.md](06_decisiones_y_lecciones.md) | Tabla de todas las decisiones con alternativas y motivo, problemas encontrados y cómo se resolvieron, limitaciones | Todos |
| [07_guion_y_preguntas_defensa.md](07_guion_y_preguntas_defensa.md) | Reparto en 4 partes, demo en vivo, y más de 40 preguntas probables con respuesta | Todos |
| [08_glosario.md](08_glosario.md) | Términos técnicos explicados en lenguaje sencillo | Todos |

## Estado del proyecto (qué existe y qué no)

| Etapa | Qué es | Estado |
|---|---|---|
| 0 | Andamiaje: `uv`, configuración tipada, adaptadores de LLM, embeddings, tests, calidad | Hecha y fusionada |
| 1 | Pipeline de datos automático: 13 fuentes, limpieza, documentos derivados del CSV | Hecha y fusionada |
| 2 | Chunking, embeddings, ChromaDB, experimento de tamaño de chunk (se decidió 600 / 90 de solapamiento) | Hecha y fusionada |
| 3 | Comparación sin RAG vs con RAG y evaluación | **Hecha en código y verificada** (14 preguntas, ejecución final en `eval/results.csv`); falta la **puntuación manual del equipo** y la verificación humana de L1/L2 |
| 4 | App web Gradio desplegada en Hugging Face Spaces | Pendiente |
| 5 | RAG avanzado: híbrido BM25+vector, reranking, filtros, router | Pendiente |
| 6 | Memoria en Quarto publicada | Pendiente |
| 7 | Congelación, pruebas desde un clon limpio, ensayo | Pendiente |

## Cifras clave para memorizar

- Dataset: Kaggle *Bank Customer Churn*, **10.000 clientes, 18 columnas**, Francia / Alemania / España. Churn global **20,4 %**.
- Corpus: **13 fuentes, 178 documentos, unas 399.000 palabras**, 4 idiomas (EN, ES, FR, DE) y 5 formatos (CSV, JSON, HTML, PDF y documentos generados).
- Índice: **6.309 trozos (chunks)** con tamaño 600 caracteres y 90 de solapamiento; construcción en frío unos 5 minutos en CPU; resincronización sin cambios, aproximadamente 1 segundo.
- Embeddings: `intfloat/multilingual-e5-small`, vectores de **384** dimensiones.
- LLM por defecto: `gemini-3.5-flash-lite` (el mismo modelo para el modo sin RAG y con RAG).
- Tests: **184 tests unitarios** offline, cobertura de aproximadamente **96 %** en `core`, `rag` y `data_pipeline`; `ruff` y `mypy` limpios.
- Hallazgos de datos: Alemania 32,4 % de churn frente a Francia 16,2 % y España 16,7 %; clientes con 4 productos 100 %; el 99,8 % de los clientes que se fueron habían presentado una queja (**fuga de objetivo**, *target leakage*); la puntuación de satisfacción no predice el churn.

## Comandos de la demo (desde la carpeta `rag_project/`)

```bash
uv sync                                                  # crea .venv e instala dependencias bloqueadas
uv run python -m data_pipeline.run_all                   # construye el corpus (usa la caché de descargas)
uv run python -m rag.index --query "churn rate in Germany"   # indexa (incremental) y prueba una consulta
uv run python -m eval.run_eval                           # sin RAG vs con RAG (respuestas en caché)
uv run python -m eval.score                              # puntuaciones automáticas y hoja de puntuación manual
uv run pytest                                            # 184 tests unitarios
uv run ruff check . && uv run mypy                       # calidad
```

Más detalle de la demo y de qué decir en `07_guion_y_preguntas_defensa.md`.

## Cómo mantener esta guía

Si cambia algo importante (tamaño de chunk, modelo, resultados de la evaluación, aparición de la app), actualizar el archivo correspondiente **y la tabla de estado de arriba**. Los resultados de evaluación de `04_rag_llm_y_evaluacion.md` deben revisarse tras cada ejecución de `eval.run_eval`.
