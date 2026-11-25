# 03. Chunking, embeddings y base vectorial (Etapa 2)

Esta es la parte que más se pregunta en la defensa ("justificación técnica del chunking y de la base vectorial"). Hay que poder explicar **qué se decidió, por qué y con qué evidencia**.

## 1. Por qué hay que trocear (chunking)

Un LLM y un buscador vectorial trabajan mal con documentos enormes: el vector de un documento de 500 KB (la ley de IA de la UE) es una "media borrosa" de miles de temas, y el *prompt* tiene un tamaño limitado. Se divide cada documento en **trozos (chunks)** que se indexan por separado.

El tamaño del trozo es un **compromiso**:

| Trozo pequeño (p. ej. 300 caracteres) | Trozo grande (p. ej. 1000 caracteres) |
|---|---|
| Vector más preciso y específico | Vector más difuso (mezcla temas) |
| Puede cortar una idea o separar una cifra de su etiqueta | Conserva más contexto en cada trozo |
| Muchísimos vectores (más coste y búsqueda más lenta) | Menos vectores |
| Menos texto por respuesta | Más texto (y más ruido) enviado al LLM, más tokens |

El **solapamiento (overlap)** repite el final de un trozo al principio del siguiente, para que una frase que cruza la frontera aparezca entera en alguno de los dos.

## 2. Estrategia por tipo de documento (`rag/chunking.py`)

No se trata igual a todo. Esta es la decisión central:

| Tipo de documento | Estrategia | Razón |
|---|---|---|
| Fichas de segmento, diccionario de datos, notas macro del BCE/Eurostat/Banco Mundial (`dataset_stats`, `macro_data`) | **No se trocean**: un documento = un chunk | Ya son atómicos y cortos; partirlos separaría una cifra de su etiqueta |
| Wikipedia, leyes, páginas de reguladores, papers, prensa | **Primero por encabezados Markdown, luego por tamaño** | Respeta la estructura (artículos, secciones) y el solapamiento protege las frases en las fronteras |

### Clases (patrón *Strategy*)

- `Chunker` (Protocolo): una sola operación, `chunk(doc) -> list[Chunk]`.
- `PassthroughChunker`: devuelve el documento entero como un chunk.
- `MarkdownRecursiveChunker(chunk_size, chunk_overlap)`: usa `MarkdownHeaderTextSplitter` (divide por `#`, `##`, `###`, `####` conservando la **ruta de encabezados** como metadato) y después `RecursiveCharacterTextSplitter` (corta por párrafo, línea, frase, palabra, en ese orden de preferencia).
- `FixedSizeChunker`: solo por tamaño, ignora la estructura. Existe para **comparar** en el experimento.
- `RoutingChunker`: decide por `source_type` si usa el atómico o el divisor.
- `build_chunker(cfg)`: fábrica que lo construye todo desde `config.yaml`.

### Detalles que importan

- **`Chunk.embedding_text`** (en `core/models.py`): lo que realmente se convierte en vector es *título del documento + ruta de encabezados + texto*. Así un trozo suelto de una ley sabe que es "Artículo 3, tasas de intercambio de débito" aunque su texto no lo repita.
- **Ruta de encabezados sin el H1** (`PATH_LEVELS = h2..h4`) y **`CHUNKER_VERSION = 2`**: el H1 es el título del documento, que ya está en los metadatos; en las leyes es una línea oficial de unos 150 caracteres que, repetida en cada trozo de 600, ahogaba la parte distintiva y dejaba todas las puntuaciones muy parecidas (0,90-0,93). Se quitó. La versión del chunker entra en la *huella* del indexador para que un cambio de lógica fuerce la reindexación.
- Identificador de chunk estable: `doc_id#0004`.

## 3. El experimento de tamaño de chunk (justificación empírica)

Archivo: `eval/chunking_experiment.py`. Para cada tamaño se trocea **todo el corpus**, se calculan los embeddings, se crea un índice temporal y se ejecutan **solo las preguntas de recuperación** (sin LLM) puntuando contra los documentos correctos (`gold_source_ids`).

Métricas (a nivel de chunk):
- **hit@k**: ¿alguno de los k primeros trozos pertenece a un documento correcto?
- **MRR** (*mean reciprocal rank*): media de 1/posición del primer trozo correcto. Vale 1 si siempre es el primero.

Resultados (10 preguntas con fuentes de referencia):

| Variante | Chunks | Caracteres medios | hit@1 | hit@3 | hit@5 | MRR | Contexto top-5 (caracteres) |
|---|---|---|---|---|---|---|---|
| 300 / solape 45 | 13.466 | 197 | 0,80 | 1,00 | 1,00 | 0,883 | 2.154 |
| **600 / solape 90** | 6.309 | 415 | 0,90 | 1,00 | 1,00 | 0,933 | 2.795 |
| 1000 / solape 150 | 3.873 | 681 | 0,90 | 1,00 | 1,00 | 0,933 | 3.341 |
| Fijo 600 (sin encabezados) | 6.000 | 438 | 0,90 | 1,00 | 1,00 | 0,933 | 2.735 |

**Decisión del equipo: 600 caracteres con 90 de solapamiento (15 %).** Argumentos:
1. 300 es peor (una pregunta más falla en el primer puesto) y **duplica** los trozos.
2. 600 y 1000 empatan en calidad, pero 600 envía al LLM un **16 % menos de contexto** (menos ruido y menos tokens).
3. Es el valor por defecto razonable para texto estructurado y encaja con el límite del modelo de embeddings.

**Limitaciones que hay que reconocer (ser honestos puntúa):**
- Con 10 preguntas, una pregunta pesa 0,1; la diferencia entre 300 y 600 es una sola pregunta.
- Los documentos atómicos (fichas, notas macro) **no cambian con el tamaño**, así que el experimento discrimina poco.
- El chunking por encabezados **empata** con el fijo en estas métricas; su ventaja real es que el trozo conserva la ruta de encabezados para citar. Las métricas son a nivel de documento/chunk, no miden si el trozo contiene la respuesta (ver `04_rag_llm_y_evaluacion.md`, caso R1/R4).
- El experimento se ejecutó antes del ajuste "sin H1" (versión 2 del chunker); el ajuste no cambia los límites de los trozos, solo el texto que se convierte en vector.
- No se implementó el chunking semántico (era opcional).

Anécdota útil: la variante de 600 tardó 55 minutos de reloj pero el temporizador interno midió solo 304 s: el portátil se suspendió. Las otras variantes tardaron unos 4 minutos.

## 4. Embeddings: `intfloat/multilingual-e5-small`

Un **embedding** convierte un texto en un vector de números de modo que textos con significado parecido quedan cerca. Aquí son vectores de **384 dimensiones**.

Por qué este modelo:
- **Multilingüe**: el corpus está en inglés, español, francés y alemán, y una pregunta en español debe encontrar un texto en español o inglés. Se comprobó: "reclamación al Banco de España" y "SCHUFA Score Berechnung" recuperan los documentos correctos.
- **Pequeño**: corre en CPU, lo que permite alojarlo en el nivel gratuito de Hugging Face Spaces.

Particularidad de la familia e5: exige **prefijos**. Los pasajes se codifican como `"passage: ..."` y las consultas como `"query: ..."`. Lo implementa `E5Embedder` (`rag/embeddings.py`) con `embed_passages()` y `embed_query()`, y los vectores se normalizan (así el producto escalar equivale al coseno).

## 5. ChromaDB y por qué

Se usa **ChromaDB persistente** (carpeta `chroma_db/`, no se versiona).

| Opción | Pros | Contras para este proyecto |
|---|---|---|
| **ChromaDB** (elegida) | Embebida (sin servidor), persistente, **filtros por metadatos** nativos, trivial de desplegar en un Space | Menos escalable que sistemas distribuidos (irrelevante con 6.309 vectores) |
| FAISS | Muy rápida | Sin filtros nativos por metadatos; habría que construirlos |
| Qdrant / Milvus / Pinecone | Escalan a millones, funciones avanzadas | Requieren servidor o servicio externo: infraestructura extra sin necesidad |

### `VectorStoreRepository` (`rag/vectorstore.py`): patrón *Repository*
Envuelve Chroma; **el resto del código nunca importa `chromadb`**.
- `add(chunks, doc_hash)`: calcula embeddings por lotes y hace *upsert* por `chunk_id`.
- `query(texto, k, where)`: devuelve `RetrievedChunk` ordenados, con `score = 1 - distancia coseno`. `where` filtra por metadatos (por ejemplo `{"geography": {"$in": ["DE", "EU"]}}`).
- `delete_document(doc_id)`, `doc_hashes()` (huella guardada por documento), `rebuild()` (borra la colección), `count()`.
- Los metadatos se **aplanan** a tipos escalares (Chroma no admite listas): `columns` se guarda como `"Balance,Exited"`.
- La colección usa distancia **coseno** (`hnsw:space = cosine`).

## 6. Indexación incremental (`rag/index.py`)

`Indexer.sync(docs)` mantiene el índice igual que el corpus sin reprocesar todo:
- Calcula la **huella** (`fingerprint`) de cada documento: SHA-256 de *texto + metadatos + configuración de chunking + modelo de embeddings + `CHUNKER_VERSION`*.
- Si la huella guardada coincide: se **omite**. Si cambió: se borra y se reindexa. Si el documento ya no existe: se elimina.
- `Indexer.rebuild(docs)`: borra y empieza de cero.

Tiempos medidos: construcción completa unos **292 segundos** en CPU (6.309 trozos); resincronización de un corpus sin cambios, **aproximadamente 1 segundo**.

CLI: `uv run python -m rag.index [--rebuild] [--query "texto"] [-k 5]`.

Comprobación de aceptación de la etapa: la consulta "churn rate in Germany" devuelve la ficha de Alemania en el **top 3** (en el puesto 2, tras el resumen por geografía); las consultas en español y alemán recuperan el documento correcto; el filtro por país funciona (`tests/integration/test_index_acceptance.py`).

## 7. Preguntas probables

- *¿Por qué 600 y no 1000 si empatan?* Mismo resultado con menos contexto (menos ruido y menos coste de tokens); lo decidió el equipo con los datos del experimento.
- *¿Por qué no troceáis las fichas?* Una ficha es una unidad completa; cortarla separaría la cifra de su etiqueta.
- *¿Qué hace el solapamiento?* Evita perder una frase que cae justo en la frontera entre dos trozos.
- *¿Por qué ChromaDB y no FAISS o Qdrant?* Filtros por metadatos nativos, sin servidor, persistente y fácil de desplegar gratis. Con unos miles de vectores no se necesita más.
- *¿Por qué e5-small?* Multilingüe y lo bastante pequeño para CPU gratuita. Hay que recordar los prefijos `query:`/`passage:`.
- *¿Cómo evitáis reindexar todo cada vez?* Huella por documento; solo se procesa lo que cambió.
- *¿Qué mide hit@k y MRR?* Si el trozo correcto aparece entre los k primeros y en qué posición media.
- *¿Qué limitaciones tiene vuestro experimento?* Pocas preguntas, y los documentos atómicos no dependen del tamaño; lo decimos explícitamente.
