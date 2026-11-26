# Guía de presentación: proyecto RAG sobre churn en banca europea

> Estado: **26 de noviembre de 2026**. El sistema está completo y publicado: pipeline de datos, índice vectorial, cadena RAG con modo avanzado, evaluación, app Gradio desplegada en Google Cloud Run y sitio Quarto en GitHub Pages. Las referencias L1 y L2 están verificadas contra el PDF. Solo queda el ensayo de la defensa, que hace el equipo.
> Presentación: **viernes 27 de noviembre de 2026**.

## Enlaces

| Qué | Dónde |
|---|---|
| App pública (Google Cloud Run) | <https://rag-churn-6lkxbo5a5a-ew.a.run.app/> |
| Sitio Quarto (GitHub Pages) | <https://adra642.github.io/Proyecto-IA/> |
| Repositorio | <https://github.com/Adra642/Proyecto-IA> (carpeta `rag_project/`) |

## Para qué sirve esta carpeta

Es un manual para que **cada miembro del equipo pueda explicar y defender cualquier parte del proyecto**, no solo la suya (la nota incluye "dominio técnico individual"). Describe **cómo es el sistema hoy**; las decisiones pasadas aparecen solo cuando ayudan a justificar algo.

Los archivos son independientes, pero conviene leerlos en orden la primera vez.

| Archivo | Contenido | Quién debería dominarlo |
|---|---|---|
| [01_vision_general_y_arquitectura.md](01_vision_general_y_arquitectura.md) | Qué es el proyecto, idea central, arquitectura, mapa de carpetas, stack y flujo de una pregunta | Todos |
| [02_pipeline_de_datos.md](02_pipeline_de_datos.md) | Cómo se descargan, limpian y convierten en conocimiento las 13 fuentes; privacidad | Parte 1: datos y pipeline |
| [03_chunking_embeddings_vectordb.md](03_chunking_embeddings_vectordb.md) | Chunking por tipo de documento, experimento de tamaño, embeddings e5, ChromaDB, indexación incremental | Parte 2: base vectorial y chunking |
| [04_rag_llm_y_evaluacion.md](04_rag_llm_y_evaluacion.md) | Capa de LLM, prompts, cadena RAG, evaluación y resultados | Parte 4: evaluación |
| [05_rag_avanzado.md](05_rag_avanzado.md) | Búsqueda híbrida, reranking, filtros, router y tabla de ablación | Parte 4: técnicas avanzadas |
| [06_app_despliegue_y_sitio.md](06_app_despliegue_y_sitio.md) | App Gradio, despliegue en Cloud Run, sitio Quarto y su publicación | Parte 3: interfaz y despliegue |
| [07_ingenieria_patrones_y_calidad.md](07_ingenieria_patrones_y_calidad.md) | Patrones de diseño, configuración, tests, calidad, git | Todos |
| [08_decisiones_y_lecciones.md](08_decisiones_y_lecciones.md) | Decisiones con su motivo y lecciones | Todos |
| [09_guion_y_preguntas_defensa.md](09_guion_y_preguntas_defensa.md) | Reparto en 4 partes, demo en vivo y preguntas probables con respuesta | Todos |
| [10_glosario.md](10_glosario.md) | Términos técnicos en lenguaje sencillo | Todos |
| [11_verificacion_L1_L2.md](11_verificacion_L1_L2.md) | Por qué y cómo se verificaron las referencias L1 y L2 | Parte 4 |

## Estado del proyecto

| Apartado del enunciado | Qué hay | Estado |
|---|---|---|
| A. Corpus automatizado | 13 fuentes, 177 documentos, un solo comando, idempotente | Hecho |
| B. RAG, base vectorial y chunking | LangChain + ChromaDB + e5; chunking por tipo de documento; experimento de tamaño | Hecho |
| C. Sin RAG frente a con RAG | 15 preguntas, tres modos, puntuación automática | Hecho (puntuación automática; sin puntuación manual) |
| D. App web pública | Gradio en Cloud Run, interfaz en español, respuestas siempre en español | Hecho y desplegada |
| E. Extras | Híbrido BM25 + vector, filtros por metadatos, router, reranking mixto (medido por trozo y activado) | Hecho |
| Documentación | Sitio Quarto en español, publicado en GitHub Pages, con menú desplegable y navegación Anterior/Siguiente | Hecho |
| Calidad | 230 tests unitarios, cobertura 95 %, ruff y mypy estrictos limpios | Hecho |

### Pendiente
Nada de código. El equipo ensaya la defensa por su cuenta (guion y preguntas en [09](09_guion_y_preguntas_defensa.md)). Opcional: capturas de una respuesta, de *Fuentes* y de *Comparar* para la página de la app.

## Cifras clave para memorizar

- **Dataset:** Kaggle *Bank Customer Churn*, 10.000 clientes, Francia / Alemania / España. Churn global **20,4 %**; Alemania **32,4 %**, Francia 16,2 %, España 16,7 %.
- **Corpus:** 13 fuentes, **177 documentos**, unas 325.000 palabras, 4 idiomas (inglés, español, francés y alemán), 5 formatos.
- **Índice:** **4.663 trozos** de 600 caracteres con 90 de solapamiento; 384 dimensiones; construcción en frío unos 5 minutos en CPU, 1 segundo sin cambios.
- **Modelos:** embeddings `intfloat/multilingual-e5-small`; LLM `gemini-3.5-flash-lite` (el mismo en todos los modos).
- **Resultados (cifras correctas):** datos D **0,12 → 0,75 → 1,00** y regulatorio/actual R **0,00 → 0,83 → 0,83** (sin RAG → RAG → RAG avanzado).
- **Recuperación (13 preguntas):** vectorial MRR 0,887; **híbrida 1,000** (hit@1 de 0,846 a 1,000).
- **Calidad:** 220 tests, 96 % de cobertura.

## Comandos de la demo (desde `rag_project/`)

```bash
uv sync                                                      # crea .venv con las dependencias bloqueadas
uv run python -m data_pipeline.run_all                       # corpus (usa la caché de descargas)
uv run python -m rag.index --query "churn rate in Germany"   # indexa (incremental) y prueba una consulta
uv run python -m app.app                                     # app en http://localhost:8080 (necesita GOOGLE_API_KEY)
uv run python -m eval.run_eval                               # sin RAG, RAG y RAG avanzado (gasta llamadas al LLM)
uv run python -m eval.score                                  # puntuación automática
uv run pytest                                                # 230 tests sin red
uv run quarto render docs                                    # sitio Quarto en docs/_site
```

Más detalle de la demo y de qué decir en [09_guion_y_preguntas_defensa.md](09_guion_y_preguntas_defensa.md).

## Cómo mantener esta guía

Si cambia algo importante (tamaño de chunk, modelo, resultados, despliegue), actualizar el archivo correspondiente **y la tabla de estado de arriba**. Esta carpeta (`documentacion/`) está separada de `docs/` (el sitio Quarto) y no se publica.
