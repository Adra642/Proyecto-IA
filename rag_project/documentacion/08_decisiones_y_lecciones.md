# 08. Decisiones, lecciones y limitaciones

Consulta rápida para "¿por qué es así hoy?". Está ordenada por área y describe la situación actual; la historia completa está en `PROGRESS.md`.

## 1. Decisiones vigentes

### Datos
| Decisión | Alternativa descartada | Motivo |
|---|---|---|
| Convertir el CSV en **documentos de estadísticas** calculadas con pandas | Indexar 10.000 filas | Las preguntas son agregadas; ninguna fila contiene la respuesta |
| `stats.json` con cada cifra, y el **gold generado** desde él | Escribir las respuestas a mano | Reproducible y sin errores humanos |
| **Eliminar `RowNumber`, `CustomerId`, `Surname`** antes de crear documentos, con escaneo automático con alcance | Dejarlos | Privacidad por diseño; el escaneo compara exacto en documentos derivados y solo registra colisiones en texto externo (muchos apellidos son palabras comunes) |
| arXiv: **resumen y extractos** con enlace | Texto completo | Derechos de autor |
| Cadena de alternativas para EUR-Lex (HTML, Cellar XHTML, PDF) | Abandonar la fuente | El HTML directo lo bloquea un reto anti-bots |
| La limpieza de obsoletos solo toca carpetas con documentos nuevos | Borrar todo lo que no esté en la ejecución | Una fuente caída no debe borrar su corpus previo |
| Proyecto **autocontenido** (`core/` compartido; `data_pipeline` y `rag` no se importan) | Reutilizar código del proyecto 1 | Independencia pedida y bajo acoplamiento |

### Índice y recuperación
| Decisión | Alternativa descartada | Motivo |
|---|---|---|
| **ChromaDB** persistente | FAISS, Qdrant, Milvus, Pinecone | Filtros por metadatos nativos, sin servidor, cabe en la imagen del contenedor |
| `multilingual-e5-small` con prefijos `query:`/`passage:` | Modelos solo en inglés o grandes | Corpus en 4 idiomas y ejecución en CPU |
| **Chunking según tipo de documento**, 600 / 90 | Un tamaño único; 300 o 1000 | Los atómicos no se parten; 600 equilibra calidad y contexto (ver `03`) |
| Texto del vector = título + ruta de encabezados (sin H1) + texto | Solo el texto del trozo | El contexto ayuda a recuperar apartados de leyes; el H1 largo repetido ahogaba lo distintivo |
| **Indexación incremental** con huella (incluye `CHUNKER_VERSION`) | Reconstruir siempre | 1 s sin cambios; un cambio de lógica fuerza la reindexación |
| **Híbrida BM25 + vectorial** en el modo avanzado | Solo vectorial | Medido: MRR de 0,794 a 0,903 y arregla D6 |
| **Reranking implementado pero desactivado** | Activarlo | Medido: no mejora y los grandes empeoran |
| **Router por reglas** | Router con LLM | Gratis, determinista y comprobable |

### Generación y evaluación
| Decisión | Alternativa descartada | Motivo |
|---|---|---|
| **Mismo modelo y prompt** en los tres modos | Modelos distintos | Solo cambia la recuperación |
| `gemini-3.5-flash-lite`, selector de proveedor con alternativas | Otros modelos | Verificado en vivo: ~1 s y correcto |
| **Respuestas siempre en español** (prompts de los tres modos y mensajes fijos) | Responder en el idioma de la pregunta | Coherencia con la interfaz y la audiencia; la comparación sigue siendo justa porque los tres modos lo reciben |
| Prompt sin RAG que **permite abstenerse** | Forzar una respuesta | Línea base honesta |
| **Guarda de privacidad determinista** | Confiar solo en el prompt | No llama al LLM, no puede inventar un apellido |
| **Verificar las citas** contra los trozos recuperados | Fiarse del modelo | Una cita inventada no cuenta |
| **Sin caché de respuestas**, una ejecución por pregunta | Caché en disco | Gemini 3.5 ignora `temperature`; las diferencias pequeñas se aceptan |
| Sustituir las preguntas que la línea base ya sabía; conservar los **diagnósticos** (D6, R1, R4) | Mantenerlas u ocultar los fallos | Lo exige el plan y la honestidad |

### Entrega
| Decisión | Alternativa descartada | Motivo |
|---|---|---|
| **Cloud Run** para la app | Servidor siempre encendido | Ejecuta el contenedor, escala a cero y tiene cuota gratuita |
| Índice y modelo **dentro de la imagen** | Construirlos al arrancar | Disco efímero; evita 5 minutos por arranque en frío |
| **GitHub Pages** para el sitio, con workflow | Cloudflare Pages | Decisión del equipo; Cloudflare Pages es solo estático |
| Menú desplegable y botones Anterior/Siguiente en Quarto | Barra con ocho enlaces | La barra estaba sobrecargada; la navegación secuencial mejora la lectura |
| Python 3.12 + **uv** con `uv.lock` | Python 3.14; pip | 3.14 es demasiado nuevo para torch y chromadb; instalaciones bloqueadas |

## 2. Lecciones aprendidas

1. **El modelo "ya sabía" la respuesta**: R1 y R2 originales eran conocimiento público; las preguntas de un RAG deben ser específicas o recientes.
2. **`hit@k` a nivel de documento engaña**: el documento correcto estaba, pero no el trozo con la cifra. Hay que medir a nivel de trozo y leer los fallos.
3. **Una frase común diluye la consulta**: «dataset Kaggle Bank Customer Churn» aparece en todas las fichas y escondía «Alemania»; la búsqueda léxica lo resolvió.
4. **Los modelos de reranking genéricos no siempre ayudan**: los cross-encoders entrenados con web hunden las fichas de estadísticas.
5. **Las reglas de limpieza necesitan tests de regresión**: una clase CSS (`layout-no-sidebars`) vació una página entera.
6. **Los comprobadores también tienen falsos positivos** (apellidos que son palabras comunes; repetir el id de la pregunta); definir bien el alcance.
7. **Revisar nombres y claves generados, no solo números**: pandas convirtió "4" en "4.0".
8. **Verificar los modelos llamándolos**: `gemini-2.5-flash` estaba retirado aunque la documentación aún lo citaba.
9. **Guardar resultados tras cada variante** de un experimento largo (el portátil se suspendió una vez).
10. **Mantener la documentación generada desde los resultados** evita que el sitio contradiga los datos.

## 3. Limitaciones conocidas (reconocerlas puntúa)

- Evaluación pequeña: 14 preguntas (12 de recuperación) y una ejecución por pregunta; Gemini 3.5 ignora `temperature`.
- El experimento de tamaño de chunk discrimina poco; 1000 sale algo mejor en MRR (una pregunta de 12).
- **Sin puntuación manual** (decisión del equipo): la evaluación es solo automática. Las referencias L1/L2 están verificadas por una persona (26 de noviembre de 2026).
- R1 necesitaría trozos vecinos; R4 varía entre ejecuciones.
- El router solo conoce las palabras clave previstas (EN/ES/FR/DE).
- Groq no probado con llamadas reales; DeepSeek sí (humo, cadena y alternativa), pero la evaluación se hizo solo con Gemini.
- Fuente HFCS sin tablas de riqueza por edad; B4 solo ve lo que hay en el feed RSS; falta la página *Consumer Trends* de la EBA.
- Licencias: la del dataset de Kaggle y las condiciones de reutilización de páginas de reguladores no están verificadas (constan así en `SOURCES.md`).
- Sin chunking semántico ni router con LLM (opcionales).
- Una captura de la app en el sitio; faltan respuestas, *Fuentes* y *Comparar*.
