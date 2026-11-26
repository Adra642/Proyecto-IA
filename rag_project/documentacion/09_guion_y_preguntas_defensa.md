# 09. Guion de la presentación, demo en vivo y preguntas de defensa

Presentación: **viernes 27 de noviembre de 2026**. Según el plan, son **cuatro partes, una por miembro**. Los tiempos son una propuesta (ajustar al tiempo real que den).

## 1. Reparto y esquema de cada parte

### Parte 1: captura de datos y pipeline (fase A) · ~6 min
Idea fuerza: *"Un comando reconstruye un corpus multifuente, multiformato y multilingüe, y convierte el CSV en conocimiento."*
1. El problema: un LLM no conoce nuestras cifras ni datos recientes. (Ejemplo D1: sin RAG "no lo sé", con RAG 32,4 %.)
2. Las 13 fuentes (tabla de `02`), 5 formatos, 4 idiomas, 174 documentos.
3. Robustez: caché, reintentos, `robots.txt`, límite de ritmo, fallo aislado, idempotencia (`diff -r` vacío).
4. **La decisión clave**: no indexar filas; pandas calcula estadísticas y las escribe como fichas; `stats.json`.
5. Privacidad: PII eliminada antes de crear documentos; verificación automática.
6. Problemas reales: EUR-Lex bloqueaba, arXiv prohíbe su host de PDFs, la clase CSS que borraba una página.

Demo: `uv run python -m data_pipeline.run_all` (rápido, con caché); abrir `data/SOURCES.md` y una ficha como `data/processed/kaggle_derived/kaggle_segment_geography_germany.md`.

### Parte 2: base vectorial, chunking y librerías (fase B) · ~6 min
Idea fuerza: *"Tres decisiones justificadas con datos: chunking por tipo de documento, tamaño elegido por experimento, y ChromaDB."*
1. Por qué trocear y el compromiso del tamaño (tabla de `03`).
2. Estrategia por tipo: atómicos sin partir; el resto por encabezados y luego por tamaño con solapamiento.
3. El **experimento** (300 / 600 / 1000 / fijo) y la decisión 600/90, con sus limitaciones.
4. Embeddings e5 multilingüe (prefijos `query:` y `passage:`), vectores de 384 dimensiones, CPU.
5. ChromaDB vs FAISS/Qdrant; patrón Repository; indexación incremental con huella (5 min en frío, 1 s sin cambios).
6. Hallazgo: título largo repetido en todos los trozos → se quitó el H1.

Demo: `uv run python -m rag.index --query "churn rate in Germany"` y una consulta en español o alemán ("reclamación al Banco de España", "SCHUFA Score Berechnung"). Mostrar `eval/chunking_results.png`.

### Parte 3: interfaz y despliegue (fase D) · ~5 min
Idea fuerza: *"Una app pública, en español, que deja ver de dónde sale cada respuesta."*
1. La app en <https://rag-churn-6lkxbo5a5a-ew.a.run.app/>: Chat con selector de modo (*Sin RAG*, *RAG*, *RAG avanzado*), filtros, acordeón **Fuentes** y estado (tiempo y ruta).
2. Pestaña **Comparar**: la misma pregunta en los tres modos a la vez.
3. Robustez: pregunta vacía o demasiado larga, límite de la API con reintento y mensaje amable, rechazo de datos personales.
4. Despliegue: contenedor con el modelo e5 y el índice dentro de la imagen, Cloud Run (2 GiB, 2 CPU), clave en Secret Manager, alerta de presupuesto; por qué Cloud Run (escala a cero, cuota gratuita).
5. El sitio Quarto: menú **Contenido**, botones Anterior/Siguiente y tablas generadas desde los resultados guardados.

Demo: abrir la app, preguntar «¿Cuál es la tasa de churn en Alemania?» en los tres modos (o en **Comparar**) y abrir *Fuentes*. Antes de la demo: poner 1 instancia mínima en Cloud Run y probar la URL en una ventana privada.

### Parte 4: evaluación y técnicas avanzadas (fases C y E) · ~7 min
Idea fuerza: *"Mismo modelo, mismas preguntas: lo único que cambia es la recuperación. Y mostramos también dónde falla."*
1. Diseño: 14 preguntas (D, L, R, U), gold generado y puntuación automática (cifras, citas, abstención, fugas de datos personales).
2. Resultados (cifras correctas, sin RAG → RAG → RAG avanzado): D **0,00 → 0,83 → 1,00**; R **0,25 → 0,75 → 0,62** (citas R 0,50 → 1,00 en avanzado); L cita el paper correcto; R3 da el dato vigente del BCE; U no discrimina.
3. Técnicas avanzadas y ablación: híbrida BM25 + vectorial (MRR 0,794 → 0,903; hit@4 0,917 → 1,00), filtros por metadatos y router por reglas; reranking implementado, medido y desactivado porque no mejora.
4. Honestidad: D6, R1 y R4 como diagnósticos (el documento correcto se recuperó pero no siempre el trozo); la evaluación es solo automática (L1/L2 verificadas por una persona); 12 preguntas son pocas.

Demo: `uv run python -m eval.ablation` (sin LLM) y abrir `eval/ablation.csv`; `uv run python -m eval.score --examples` para ejemplos antes/después.

## 2. Plan B de la demo (por si falla algo)

- **Límite de la API gratuita:** cada pregunta gasta llamadas. Tener 2 o 3 preguntas ya respondidas y capturas.
- **Arranque en frío de Cloud Run:** la primera petición tras un rato sin tráfico tarda; subir a 1 instancia mínima antes y lanzar una pregunta de prueba.
- **Sin red:** `uv run pytest` (223 tests sin red) y `uv run python -m rag.index --query "..."` funcionan sin LLM; el sitio se puede abrir desde `docs/_site/`.

## 3. Cómo repartir las preguntas del tribunal

Cada miembro debe poder responder las de su parte **y las generales**. Antes de la presentación, practicad que cada uno explique en 1 minuto *una decisión de otra parte* (el "dominio técnico individual" se evalúa por persona).

## 4. Preguntas probables y respuestas cortas

### Generales
1. **¿Qué aporta el proyecto?** Un RAG completo y automatizado sobre churn bancario europeo que demuestra con datos que el mismo LLM responde mejor con recuperación, y que documenta dónde falla.
2. **¿Cuál es la idea más importante?** Convertir el CSV en documentos de estadísticas calculadas por código, porque RAG no puede responder preguntas agregadas desde filas sueltas.
3. **¿Qué relación tiene con el proyecto 1?** Solo el dataset. No se reutiliza código ni el modelo; aquí no se predice churn.
4. **¿Qué está terminado y qué no?** Todo el sistema está hecho y publicado (pipeline, índice, RAG y RAG avanzado, evaluación, app en Cloud Run y sitio en GitHub Pages). Solo falta el ensayo del equipo. La evaluación es solo automática y las referencias L1/L2 están verificadas.
5. **¿Qué haríais con más tiempo?** Más preguntas de evaluación y varias ejecuciones por pregunta, trozos vecinos para R1, chunking semántico, probar otros proveedores de LLM.

### Datos y pipeline
6. **¿Cómo garantizáis la automatización?** Un comando; `kagglehub` descarga el CSV; ninguna carga manual.
7. **¿Qué pasa si una fuente falla?** Se registra en el informe y las demás continúan; el corpus previo de esa fuente no se borra.
8. **¿Qué es la idempotencia y cómo la demostráis?** Ejecutar dos veces da el mismo resultado; lo comprobamos con `diff -r` sobre `data/processed/`.
9. **¿Por qué respetáis `robots.txt` y esperáis entre llamadas?** Por ética y para evitar bloqueos; además arXiv pide 15 s.
10. **¿Cómo tratáis EUR-Lex si bloquea?** Detectamos el reto anti-bots y usamos el XHTML de la Oficina de Publicaciones, y como último recurso el PDF.
11. **¿Por qué no usáis el texto completo de los papers?** Derechos de autor: guardamos metadatos, resumen y extractos con enlace.
12. **¿Qué datos personales hay y qué hacéis con ellos?** `RowNumber`, `CustomerId` y `Surname`; se eliminan antes de generar documentos y se verifica con un escaneo automático.
13. **¿Por qué la comprobación de apellidos distingue mayúsculas?** Porque algunos apellidos son palabras comunes ("band", "see") que usan nuestras plantillas; un apellido real aparecería tal como está en el CSV.
14. **¿Qué es la fuga de objetivo en `Complain`?** El 99,8 % de quienes se fueron habían presentado una queja; la variable "copia" el objetivo y no debe usarse para predecir.
15. **¿La satisfacción predice el churn?** No: la tasa de churn es casi plana (19,6 % a 21,8 %).

### Chunking y base vectorial
16. **¿Por qué 600 caracteres?** Misma cobertura (hit@5 = 1,00) que 1000 con menos contexto y la mitad de trozos que 300; 1000 sale algo mejor en MRR por una sola pregunta de 12, y lo reconocemos.
17. **¿Qué limitaciones tiene el experimento?** Pocas preguntas (una pregunta pesa 0,083) y los documentos atómicos no dependen del tamaño.
18. **¿Por qué no se trocean las fichas?** Una ficha ya es una unidad; cortarla separaría una cifra de su etiqueta.
19. **¿Qué es el solapamiento y para qué sirve?** Repite el final de un trozo al inicio del siguiente para no perder frases en la frontera.
20. **¿Qué es la ruta de encabezados y para qué la usáis?** El camino "Capítulo > Artículo" de cada trozo; sirve para dar contexto al vector y para citar.
21. **¿Por qué quitasteis el H1 del contexto del vector?** En las leyes es un título oficial de unos 150 caracteres repetido en cada trozo, que ahogaba lo distintivo.
22. **¿Por qué ChromaDB?** Filtros por metadatos nativos, sin servidor, persistente y fácil de desplegar gratis; FAISS no tiene filtros y Qdrant o Milvus requieren servidor.
23. **¿Por qué e5-small y qué son los prefijos?** Multilingüe y pequeño para CPU; el modelo espera "query: " para consultas y "passage: " para documentos.
24. **¿Cómo evitáis reindexar todo?** Huella por documento (texto, metadatos, configuración, modelo y versión del chunker); solo se procesa lo que cambió.
25. **¿Qué métrica de similitud usáis?** Coseno; la puntuación es 1 menos la distancia.
26. **¿Cuántos vectores hay y cuánto tarda indexar?** 6.309; unos 5 minutos en frío, 1 segundo si no hay cambios.
26b. **¿Y en producción?** El índice va dentro de la imagen del contenedor, así que el arranque solo carga el modelo y el índice.

### RAG, LLM y evaluación
27. **¿Cómo es justa la comparación?** Mismo modelo, mismas preguntas; solo cambia el contexto recuperado.
28. **¿El modelo sin RAG no es demasiado "tímido"?** Su prompt le permite abstenerse; es una línea base honesta y lo reconocemos como limitación.
29. **¿Cómo evitáis que el modelo invente fuentes?** El prompt exige citar `[doc_id]` y el código solo cuenta citas a documentos realmente recuperados.
30. **¿Qué pasa con un tema que no está en el corpus?** Debe decir que no lo sabe (pregunta U1, Portugal); con contexto vacío, respuesta fija sin llamar al LLM.
31. **¿Cómo se protege la privacidad en el sistema?** Los datos no están indexados y además una guarda determinista rechaza las peticiones de apellidos o clientes concretos antes de llamar al LLM.
32. **¿Cómo generáis las respuestas correctas?** Desde `stats.json` y con regex sobre el texto ingerido; solo L1 y L2 se redactaron a mano y una persona las verificó contra el PDF.
33. **¿Por qué falla el RAG básico en D6 y R1?** Recupera el documento correcto pero no el trozo con la cifra, o la frase común del dataset diluye «Alemania». La búsqueda híbrida del modo avanzado arregla D6; R1 necesitaría trozos vecinos.
34. **¿Por qué `hit@k` puede engañar?** Mide si el documento correcto aparece, no si el trozo contiene la respuesta.
35. **¿Cómo gestionáis los límites de la API?** Throttling de 1 s, reintentos con espera exponencial, proveedor alternativo opcional.
36. **¿Por qué Gemini 3.5 flash lite?** Se probaron los modelos disponibles: este responde en torno a 1 s con respuestas correctas; el 2.5 está retirado.
36b. **¿En qué idioma responde?** Siempre en español, también si la pregunta está en inglés, francés o alemán; las instrucciones del prompt están en inglés y los tres modos las reciben igual.
37. **¿Qué pasa con la temperatura?** Los modelos Gemini 3.5 la ignoran; por eso es una ejecución por pregunta y se aceptan pequeñas diferencias entre ejecuciones.
38. **¿Cuántas preguntas y por qué sustituisteis algunas?** 14 (12 de recuperación más 2 diagnósticos); R1 y R2 originales las sabía el modelo sin RAG y el plan pide sustituirlas por preguntas más específicas.
39. **¿Quién puntúa las respuestas?** Se puntúan automáticamente: cifras de referencia generadas desde `stats.json`, verificación de citas contra los trozos recuperados, abstención y fugas de datos personales. No hay puntuación manual; es una limitación que reconocemos (14 preguntas, una ejecución).
39b. **¿Qué aporta el RAG avanzado?** La búsqueda híbrida sube el MRR de 0,794 a 0,903 y resuelve D6; los filtros no añaden nada medible y el reranking no ayuda, por eso está desactivado.
39c. **¿Por qué Cloud Run?** Ejecuta el contenedor con el modelo y el índice incluidos, escala a cero y tiene cuota gratuita.

### Ingeniería
40. **¿Qué patrones de diseño usáis?** Strategy, Factory/Registry, Template Method, Adapter, Decorator, Pipeline, Repository e inyección de dependencias (ver `07`).
41. **¿Cómo se prueba sin red?** Fakes, `MockTransport` y respuestas reales guardadas; 223 tests unitarios y cobertura de aproximadamente 95 %.
42. **¿Cómo se reproduce desde cero?** `uv sync`, copiar `.env.example` a `.env`, ejecutar el pipeline, el índice y la evaluación (pasos en el `README.md`).
43. **¿Cómo cambiáis de proveedor de LLM?** `llm.provider` en `config.yaml` o la variable `LLM_PROVIDER`.
44. **¿Cómo evitáis filtrar claves?** Variables de entorno y `.env` fuera de git, `SecretStr`, y `.env.example` solo con nombres.
45. **¿Cómo trabajáis con git?** Una rama y un pull request por etapa, revisado por el equipo; nadie fusiona su propio PR ni sube a `main`.

## 5. Tarjetas de cifras (una por persona)

- **Datos**: 10.000 clientes · 20,4 % de churn · Alemania 32,4 % · 4 productos 100 % · 99,8 % de quienes se fueron se habían quejado.
- **Corpus**: 13 fuentes · 174 documentos · 282.000 palabras · 4 idiomas.
- **Índice**: 6.309 trozos · 600/90 · 384 dimensiones · 5 min en frío · 1 s sin cambios.
- **Experimento de tamaño**: MRR 0,781 (300), 0,794 (600), 0,836 (1000 y fijo) · contexto top-5 2.128 / 2.568 / 2.975 caracteres.
- **Evaluación**: 14 preguntas · D: 0,00 → 0,83 → 1,00 de cifras correctas · R: 0,25 → 0,75 → 0,62.
- **Recuperación**: vectorial MRR 0,794 → híbrida 0,903 · hit@4 0,917 → 1,000 · reranking sin mejora.
- **Despliegue**: Cloud Run, 2 GiB / 2 CPU, imagen de 2,76 GB · sitio en GitHub Pages.
- **Calidad**: 223 tests · 95 % de cobertura · ruff y mypy estrictos limpios.
