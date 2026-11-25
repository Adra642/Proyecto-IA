# 06. La app, el despliegue y el sitio Quarto

## 1. La app Gradio (`app/`)

Pública en <https://rag-churn-6lkxbo5a5a-ew.a.run.app/>. La **interfaz y las respuestas están en español**, aunque la pregunta se haga en otro idioma.

| Archivo | Responsabilidad |
|---|---|
| `app/app.py` | Punto de entrada: carga la configuración, sincroniza el índice con `data/processed/` (no hace nada si ya está al día) y lanza Gradio en el puerto `PORT` (8080 por defecto) |
| `app/handlers.py` | `AppService`: ejecuta la pregunta en el modo elegido y convierte los fallos en mensajes amables en español; formatea fuentes y estado |
| `app/ui.py` | Interfaz Gradio Blocks, construida a partir de un `AppService` inyectado (por eso se prueba con *fakes*) |

### Qué ve el usuario
- **Chat:** selector de modo (*Sin RAG*, *RAG*, *RAG avanzado*), cuadro de pregunta, botones de ejemplo (las preguntas de la evaluación), acordeón **Fuentes** (título, enlace, fuente, similitud y texto recuperado de cada trozo) y línea de estado (modo, tiempo de respuesta, ruta del router).
- **Filtros** (solo en RAG avanzado): tipo de fuente, geografía e idioma; sustituyen al router.
- **Comparar:** la misma pregunta en los tres modos, en columnas, con las fuentes de RAG y RAG avanzado.
- **Acerca de:** qué es, de dónde salen los datos y la advertencia de privacidad.

### Robustez
- Pregunta vacía o de más de 1.000 caracteres: mensaje claro, sin llamar al modelo.
- Límite de la API: un reintento tras 3 s y, si sigue fallando, mensaje amable; falta de clave y fallo del proveedor tienen mensajes propios.
- La guarda de privacidad y el contexto vacío responden sin llamar al LLM.

## 2. Despliegue en Google Cloud Run

Hugging Face Spaces era el plan original, pero su documentación exige hoy un plan de pago para Gradio y Docker, así que se despliega un **contenedor en Cloud Run** (escala a cero y tiene cuota gratuita mensual; exige cuenta de facturación y una alerta de presupuesto).

`deploy/Dockerfile`, por capas ordenadas de las que menos cambian a las que más:
1. **Dependencias de ejecución** (`uv sync --frozen --no-default-groups --no-cache`): sin herramientas de desarrollo ni el grupo `pipeline`.
2. **Modelo e5 e índice de Chroma incluidos en la imagen** (`python -m rag.index`): el disco de Cloud Run es efímero y construirlos al arrancar repetiría 5 minutos en cada arranque en frío.
3. **La app** (`app/` y las preguntas de ejemplo): lo que más cambia, sin invalidar el índice.

Cifras: imagen de **2,76 GB** (antes 4,67 GB: la caché de uv ocupaba 1,74 GB y el grupo `pipeline` se separó). Reconstruir tras un cambio solo de `app/` tarda segundos gracias a la caché de capas.

Configuración en producción: región `europe-west1`, **2 GiB de memoria y 2 CPU**, 0 instancias mínimas (1 para la demo en vivo), máximo 2, acceso sin autenticar; la clave `GOOGLE_API_KEY` llega desde **Secret Manager** como variable de entorno. `deploy/cloudbuild.yaml` construye con `--cache-from` y un tiempo máximo de 30 minutos. Comando usado, desde Cloud Shell:

```bash
gcloud builds submit rag_project --config rag_project/deploy/cloudbuild.yaml \
  --substitutions=SHORT_SHA=v2,_CONTEXT=. --timeout=30m
```

**Antes de la demo:** poner 1 instancia mínima para evitar el arranque en frío y probar la URL desde otro dispositivo y en ventana privada.

## 3. El sitio Quarto (`docs/`)

Publicado en <https://adra642.github.io/Proyecto-IA/>, en español.

| Página | Contenido |
|---|---|
| Inicio | Resumen del proyecto, enlaces y tabla de resultados |
| 01 Datos | Motivación, fuentes y procedencia, hallazgos del dataset |
| 02 Pipeline | Arquitectura de la ingesta, patrones y fragmentos de código explicados |
| 03 Chunking y vector DB | Estrategia, experimento de tamaño, justificación de ChromaDB |
| 04 Evaluación | Preguntas, tablas sin RAG frente a con RAG, ejemplos antes/después |
| 05 RAG avanzado | Híbrido, filtros, reranking y ablación |
| 06 App y despliegue | La app, el contenedor y el despliegue |
| 07 Equipo y referencias | Equipo (por rellenar), tecnologías y enlaces |

Cómo funciona:
- Las tablas y gráficos son **celdas de Python que leen los resultados guardados** (`eval/*.csv`, `eval/results.json`, `data/`), y los fragmentos de código se **cortan de los archivos fuente** con `ast` (`docs/_helpers.py`): el sitio muestra lo que hay en el repositorio, no copias.
- **Navegación:** la barra superior tiene *Inicio* y un menú desplegable **Contenido** con las siete partes; al final de cada página hay botones **Anterior** y **Siguiente**. Quarto solo genera esos botones a partir de una barra lateral; hay una declarada en `_quarto.yml` solo para fijar el orden, oculta con CSS (`styles.css`).
- `documentacion/` (esta guía, en la raíz del proyecto) está separada del sitio y **no se publica**.
- **Publicación:** `.github/workflows/rag_project-docs.yml` (en la raíz del repositorio, filtrado a `rag_project/**`) renderiza con Quarto 1.10.19 y despliega en GitHub Pages en cada push a `main`; en los pull requests solo comprueba que renderiza. No necesita claves.
- En local: `uv sync --group docs` y `uv run quarto preview docs` (o `uv run poe docs`).

## 4. Preguntas probables

- *¿Por qué Cloud Run y no Hugging Face Spaces?* Spaces exige un plan de pago para Gradio y Docker; Cloud Run escala a cero y tiene cuota gratuita. Cloudflare Pages es solo estático (sirve para el sitio, no para la app).
- *¿Por qué el índice va dentro de la imagen?* El disco es efímero; así el arranque en frío solo carga el modelo y el índice.
- *¿Qué pasa si se agota la cuota del modelo?* Un reintento y un mensaje amable; con `llm.fallback_order` podría pasar a otro proveedor.
- *¿Dónde está la clave?* En Secret Manager, inyectada como variable de entorno; nunca en el repositorio ni en la imagen.
- *¿Cuánto cuesta?* Escala a cero y la cuota gratuita cubre la demo; hay una alerta de presupuesto.
- *¿Cómo se actualiza el sitio?* Un push a `main` con cambios en `rag_project/` dispara el workflow.
- *¿Por qué las tablas del sitio no se escriben a mano?* Para que el sitio no pueda contradecir los resultados guardados.
