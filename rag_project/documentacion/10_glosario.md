# 08. Glosario

Términos del proyecto explicados en lenguaje sencillo, ordenados por tema.

## RAG y recuperación

| Término | Explicación |
|---|---|
| **RAG** (*Retrieval-Augmented Generation*) | Técnica en la que, antes de que el LLM responda, se **recuperan** documentos relevantes y se le entregan como contexto para que responda basándose en ellos |
| **LLM** | Modelo de lenguaje grande (aquí Gemini). Genera texto; no conoce nuestros datos ni lo posterior a su fecha de corte |
| **Sin RAG / línea base** | El mismo LLM respondiendo sin contexto recuperado; sirve de comparación |
| **Alucinación** | Cuando el modelo inventa una respuesta que suena plausible pero no es cierta |
| **Contexto** | Los fragmentos recuperados que se añaden al *prompt* |
| **Prompt** | El texto completo que se envía al LLM: reglas, contexto y pregunta |
| **Cita** | Referencia `[doc_id]` a la fuente de una afirmación; aquí se verifica que el documento fue realmente recuperado |
| **Retriever** | Componente que, dada una pregunta, devuelve los trozos más relevantes |
| **Grounding / anclaje** | Que la respuesta se base únicamente en el contexto recuperado |

## Chunking y vectores

| Término | Explicación |
|---|---|
| **Chunk (trozo)** | Fragmento de un documento que se indexa por separado |
| **Chunking** | Proceso de dividir documentos en trozos |
| **Solapamiento (overlap)** | Caracteres que se repiten entre trozos consecutivos para no cortar ideas en la frontera |
| **Chunk atómico** | Documento corto que se mantiene entero (ficha de segmento, nota macro) |
| **Ruta de encabezados** | Camino de títulos de un trozo (por ejemplo "CHAPTER II > Article 3") guardado como metadato |
| **Embedding** | Vector de números que representa el significado de un texto; textos parecidos quedan cerca |
| **Similitud coseno** | Medida de cercanía entre dos vectores (1 = misma dirección); en este proyecto, puntuación = 1 − distancia coseno |
| **Base vectorial** | Base de datos que guarda vectores y encuentra los más cercanos a una consulta (aquí ChromaDB) |
| **Metadatos / filtros** | Datos asociados a cada trozo (país, idioma, tipo, fuente) que permiten filtrar la búsqueda |
| **e5 y prefijos** | Familia de modelos de embeddings que exige anteponer `query: ` a las consultas y `passage: ` a los documentos |
| **Huella (fingerprint)** | Hash que identifica el contenido de un documento y su configuración; permite saber si hay que reindexar |
| **Indexación incremental** | Actualizar el índice solo con lo que cambió |

## Métricas

| Término | Explicación |
|---|---|
| **hit@k** | Proporción de preguntas para las que algún trozo correcto aparece entre los k primeros resultados |
| **MRR** | Media del inverso de la posición del primer resultado correcto; 1 si siempre es el primero |
| **Gold (respuesta de referencia)** | Respuesta correcta conocida con la que se compara; aquí se genera desde `stats.json` o el texto de las fuentes |
| **Puntuación manual 0/1/2** | 0 incorrecta o inventada, 1 parcial o vaga, 2 correcta y específica |
| **Segunda opinión automática** | Comprobaciones por código (cifras, citas, comportamiento) que acompañan, sin sustituir, a la puntuación humana |

## Datos y privacidad

| Término | Explicación |
|---|---|
| **Churn (abandono)** | Que un cliente deje el banco; columna `Exited` |
| **Tasa de churn** | Porcentaje de clientes que se van |
| **Segmento / ficha de segmento** | Grupo de clientes (por ejemplo "Alemania") y su documento resumen con recuento, tasa y medias |
| **Fuga de objetivo (target leakage)** | Una variable que ya contiene (o copia) el resultado que se quiere predecir; `Complain` casi coincide con `Exited` |
| **PII** | Información de identificación personal: aquí `RowNumber`, `CustomerId` y `Surname` |
| **Información mutua** | Medida de cuánto informa una variable sobre otra; se usa para ordenar las variables que más influyen en el churn |
| **stats.json** | Archivo con todas las cifras calculadas, por nombre |

## Ingeniería

| Término | Explicación |
|---|---|
| **Idempotente** | Que ejecutarlo varias veces da el mismo resultado sin duplicar ni corromper |
| **Fetcher** | Módulo que descarga y convierte una fuente en documentos |
| **robots.txt** | Archivo de un sitio web que indica qué partes pueden rastrear los robots |
| **Throttling** | Limitar la frecuencia de llamadas |
| **Backoff exponencial** | Esperar cada vez más (10 s, 20 s, 40 s) tras un fallo antes de reintentar |
| **Fallback** | Alternativa que se usa cuando falla la opción principal |
| **Inyección de dependencias** | Pasar a una clase sus colaboradores en lugar de que los cree ella |
| **Raíz de composición** | El único lugar (`factory.py`) donde se conectan todos los componentes |
| **Protocolo (typing.Protocol)** | Contrato de métodos que cualquier objeto compatible cumple, sin herencia |
| **Fake** | Sustituto simple de un servicio real para probar sin red |
| **Fixture** | Dato de entrada guardado para tests (por ejemplo una respuesta real de una API) |
| **Cobertura** | Porcentaje de líneas de código ejecutadas por los tests |
| **uv** | Gestor de entornos y dependencias de Python con archivo de bloqueo (`uv.lock`) |
| **poe (poethepoet)** | Ejecutor de tareas multiplataforma (`uv run poe test`) |

## Técnicas de la Etapa 5 (pendientes)

| Término | Explicación |
|---|---|
| **BM25** | Búsqueda léxica clásica: puntúa por palabras coincidentes; acierta con términos exactos (números de artículo, nombres) |
| **Búsqueda híbrida** | Combinar la búsqueda léxica (BM25) y la vectorial |
| **Reranking** | Reordenar los mejores resultados con un modelo más preciso (aquí FlashRank, local) |
| **Router / RAG adaptativo** | Decidir si hace falta recuperar y con qué estrategia, según la pregunta |
| **Ablación** | Medir el efecto de cada técnica añadiéndolas una a una |
