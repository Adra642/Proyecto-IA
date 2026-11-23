# 06. Decisiones, lecciones aprendidas y limitaciones

Una tabla para consultar rápido "¿por qué hicimos X?" y poder defenderlo. Cada decisión tiene alternativas y un motivo.

## 1. Decisiones de diseño

| # | Decisión | Alternativas consideradas | Motivo | Dónde |
|---|---|---|---|---|
| 1 | **Convertir el CSV en documentos de estadísticas** y no indexar filas | Indexar 10.000 filas | Las preguntas son agregadas; ninguna fila contiene la respuesta | `build_docs.py` |
| 2 | **`stats.json` con cada cifra** y gold generado desde él | Escribir las respuestas a mano | Evita errores humanos; el gold es reproducible | `build_docs.py`, `eval/generate_gold.py` |
| 3 | **Eliminar `RowNumber`, `CustomerId`, `Surname` antes de generar nada** | Dejarlos | Privacidad por diseño (aunque el dato sea sintético) | `clean.drop_pii`, `pii_check.py` |
| 4 | Proyecto **autocontenido** dentro de `rag_project/` | Reutilizar código del proyecto 1 | Independencia pedida; el único vínculo es el dataset | Estructura general |
| 5 | **Python 3.12 + uv** | Python 3.14 del sistema; pip | 3.14 es demasiado nuevo para torch/chromadb; uv da instalaciones bloqueadas y rápidas | `pyproject.toml`, `.python-version` |
| 6 | **LangChain** para el LLM y los divisores de texto | Llamar a cada API a mano | Integraciones listas; BM25/rerank en la etapa 5 | `rag/llm.py`, `rag/chunking.py` |
| 7 | **ChromaDB** persistente | FAISS, Qdrant, Milvus, Pinecone | Filtros por metadatos nativos, sin servidor, fácil de desplegar gratis | `rag/vectorstore.py` |
| 8 | **`multilingual-e5-small`** | Modelos solo en inglés; modelos grandes | Corpus en 4 idiomas y ejecución en CPU gratuita | `rag/embeddings.py` |
| 9 | **Chunking según el tipo de documento** | Un único tamaño para todo | Los documentos atómicos no deben partirse; los largos sí, respetando su estructura | `rag/chunking.py` |
| 10 | **600 caracteres / 90 de solapamiento** | 300, 1000, tamaño fijo sin encabezados | Mejor calidad con el menor contexto, apoyado en el experimento | `config.yaml`, `eval/chunking_results.csv` |
| 11 | **Título del documento + ruta de encabezados** en el texto que se convierte en vector, **sin el H1** | Solo el texto del trozo; incluir el H1 | El contexto ayuda a recuperar apartados de leyes; el H1 largo repetido ahogaba lo distintivo | `core/models.py`, `rag/chunking.py` |
| 12 | **Indexación incremental con huella** (incluye `CHUNKER_VERSION`) | Reconstruir siempre | Resincronizar sin cambios tarda un segundo; un cambio de lógica fuerza la reindexación | `rag/index.py` |
| 13 | **Gemini `gemini-3.5-flash-lite`** por defecto, con selector de proveedor | Otros modelos de Gemini; Groq; DeepSeek | Verificado en vivo: rápido y correcto; capa gratuita | `config.yaml` |
| 14 | **Mismo modelo en los dos modos** | Modelos distintos | La comparación solo es justa si cambia únicamente la recuperación | `RagChain` |
| 15 | **Prompt sin RAG que permite abstenerse** | Forzar siempre una respuesta | Línea base honesta; evita medir "alucinación forzada" | `rag/prompts.py` |
| 16 | **Guarda de privacidad determinista** en el modo RAG | Confiar solo en el prompt | No llama al LLM, no puede alucinar un apellido | `rag/chain.py` |
| 17 | **Verificar las citas** contra los trozos recuperados | Fiarse de lo que escribe el modelo | Una cita inventada no cuenta | `extract_citations` |
| 18 | **Sin caché de respuestas**, una ejecución por pregunta (caché retirada el 2026-11-23) | Caché en disco; varias ejecuciones | Gemini 3.5 ignora `temperature` y las pequeñas diferencias entre ejecuciones no importan | `eval/run_eval.py` |
| 19 | **Sustituir preguntas que la línea base ya sabía** (R1, R2 originales) | Mantenerlas | Pide el plan; si el modelo ya acierta, no demuestra nada | `eval/questions.json` |
| 20 | **Conservar una pregunta donde el RAG falla (R4)** como diagnóstico | Ocultarla | Honestidad y motivación para el RAG avanzado | `eval/questions.json` |
| 21 | **arXiv: resumen y extractos**, no el texto completo | Texto completo | Derechos de autor: no republicar PDFs enteros | `fetchers/arxiv.py` |
| 22 | **Escaneo de privacidad con alcance**: comparación exacta en documentos derivados; colisiones externas solo se registran | Búsqueda literal de cualquier apellido en todo el corpus | Muchos apellidos son palabras comunes o autores de papers | `pii_check.py` |
| 23 | **Cadena de alternativas para EUR-Lex** (HTML, Cellar XHTML, PDF) | Abandonar la fuente | El reto anti-bots bloquea el HTML directo | `fetchers/eurlex.py` |
| 24 | **Limpieza de obsoletos solo en carpetas con documentos nuevos** | Borrar todo lo que no esté en la ejecución | Una fuente caída no debe borrar su corpus previo | `writer.py` |
| 25 | **`core/` compartido** y `data_pipeline`/`rag` sin importarse | Importaciones cruzadas | Bajo acoplamiento | Estructura |
| 26 | **PRs por etapa, sin fusionar el propio** | Commits directos a `main` | Revisión del equipo | `CLAUDE.md` |

## 2. Problemas reales y qué se aprendió

1. **Quien "sabe" la respuesta**: la línea base respondió bien R1 y R2 (conocimiento público). Lección: las preguntas del RAG deben ser **específicas o recientes**.
2. **Todo el contenido borrado por una clase CSS** (`layout-no-sidebars`): una regla de limpieza demasiado codiciosa. Lección: las reglas de borrado necesitan excepciones para contenedores y **tests de regresión**.
3. **Falsos positivos de privacidad** con apellidos que son palabras ("band", "see"). Lección: definir bien el alcance de una comprobación para que sea útil y no ruidosa.
4. **Etiquetas "4.0"** por la conversión a decimales de pandas. Lección: revisar los **nombres de documentos y claves** generados, no solo los números.
5. **El portátil se suspendió** durante un experimento largo (55 min de reloj frente a 304 s medidos). Lección: guardar resultados **tras cada variante** (ya implementado) y medir el tiempo con relojes monótonos.
6. **PR fusionado en la rama equivocada**: la Etapa 1 acabó en `stage-0-scaffold` y no en `main`. Lección: reorientar los PR apilados antes de fusionar.
7. **Modelos retirados**: `gemini-2.5-flash` devuelve 404. Lección: verificar los nombres de modelo **llamándolos de verdad**, no solo leyendo documentación.
8. **Recuperación aparentemente perfecta (hit@k = 1) pero respuesta incorrecta**: el documento correcto estaba pero no el trozo con la cifra. Lección: medir también si el trozo recuperado **contiene la respuesta**.
9. **Cloudflare Pages falla con `quarto: not found`**: es el proyecto 1 conectado al repositorio (necesita raíz y comando de construcción correctos y Quarto instalado). No afecta al RAG; la documentación de este proyecto se publicará mediante un workflow de GitHub Actions en la Etapa 6.

## 3. Limitaciones conocidas (reconocerlas es parte de una buena defensa)

- Conjunto de evaluación pequeño (14 preguntas, 3 de ellas diagnósticos) y una ejecución por pregunta.
- El experimento de tamaño de chunk discrimina poco (ver `03_chunking_embeddings_vectordb.md`).
- Gemini 3.5 ignora la temperatura: respuestas no estrictamente reproducibles (se acepta).
- Gold de L1 y L2 pendiente de verificación humana contra el PDF; puntuación manual pendiente.
- Groq y DeepSeek no probados con llamadas reales (sin claves).
- Fuente HFCS (B5) sin tablas de riqueza por edad (el BCE no las expone en una URL estable); la fuente B4 solo ve lo que hay en el feed RSS; no se encontró la página del informe *Consumer Trends* de la EBA.
- Licencias: la del dataset de Kaggle y las condiciones de reutilización de las páginas de reguladores **no están verificadas** (constan así en `SOURCES.md`).
- No se implementó el chunking semántico (opcional).
- `chroma_db/` no se versiona: la Etapa 4 decide entre un índice prefabricado o construirlo al arrancar.

## 4. Lo que falta (para no presentarlo como hecho)

| Etapa | Pendiente |
|---|---|
| 3 | Puntuación manual del equipo, verificación de L1/L2, re-medir tras el ajuste de chunking, memoria de ejemplos antes/después |
| 4 | App Gradio (pestaña de chat con selector de modo y pestaña de comparación lado a lado, panel de fuentes) y despliegue público en Hugging Face Spaces |
| 5 | RAG avanzado: híbrido BM25 + vector, reranking con FlashRank, filtros por metadatos, router; tabla de ablación |
| 6 | Sitio Quarto (en español) publicado, con el enlace a la app y al repositorio |
| 7 | Congelación, prueba desde un clon limpio siguiendo solo el README, demo de respaldo (vídeo y capturas), ensayo |
