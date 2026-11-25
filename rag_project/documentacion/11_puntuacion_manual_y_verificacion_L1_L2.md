# 11. Puntuación manual y verificación de L1/L2: por qué importan y cómo hacerlas

Son las dos tareas que **solo puede hacer una persona**. El código ya hace la parte automática, pero sin estas dos el apartado de evaluación (fase C) queda sin su prueba más importante. Tiempo estimado: **1-2 horas** entre dos personas.

## 1. Por qué importan

**Puntuación manual.**
- La puntuación automática es solo una *segunda opinión*: busca cifras y citas con expresiones regulares. No entiende una respuesta correcta con otras palabras, una cifra correcta con una explicación equivocada, ni una respuesta vaga que casualmente nombra el número.
- La rúbrica premia la **claridad de la comparación sin RAG / con RAG**. Una tabla con puntuaciones 0/1/2 hechas por personas es lo que el tribunal espera ver y lo que sostiene la afirmación "el RAG responde mejor".
- Detecta lo que las métricas no ven: respuestas que citan un documento correcto pero afirman algo que el documento no dice, o citas válidas con una cifra mal copiada.
- El sitio Quarto (`04-evaluacion`) muestra "pendiente" hasta que haya puntuaciones; en cuanto las haya, rellena su propia tabla.

**Verificación de L1 y L2.**
- Las respuestas de referencia (gold) de las preguntas de literatura **no se generan del dato**: son borradores escritos a partir de los *resúmenes* de arXiv. Llevan el prefijo `DRAFT, human must verify against the PDF`.
- Si un borrador fuera incorrecto, las puntuaciones de L1 y L2 (y de las métricas `hit` y `citation_ok` que dependen de él) estarían medidas contra una referencia falsa. Es el único punto del evaluador donde una persona escribió la verdad a mano, y por eso es el único que hay que comprobar.
- Es una cuestión de honestidad: en la defensa se afirma que "ningún gold se escribe a mano salvo L1/L2, que verifica una persona". Esta tarea hace cierta esa frase.

## 2. Orden correcto (importante)

1. **Decidir la ejecución final** de la evaluación. Las respuestas guardadas son de antes del cambio a respuestas en español. Si se quiere puntuar respuestas en español, ejecutar ahora:
   ```bash
   uv run python -m eval.run_eval        # 14 preguntas × 3 modos = 42 respuestas; gasta llamadas al LLM
   uv run python -m eval.score           # actualiza la puntuación automática y la hoja manual
   ```
   **No se debe reejecutar `run_eval` después de puntuar.** La hoja conserva las puntuaciones por (`id`, `modo`) aunque la respuesta cambie, y entonces una nota quedaría asignada a una respuesta distinta de la que se juzgó (Gemini 3.5 ignora `temperature`, así que dos ejecuciones nunca son idénticas).
2. **Verificar L1 y L2** (sección 4), porque la referencia debe estar firmada antes de puntuar esas preguntas.
3. **Puntuar** las 42 filas (sección 3).
4. **Agregar y publicar** (sección 5).

## 3. Cómo puntuar (`eval/manual_scoring.csv`)

Abrir el archivo con una hoja de cálculo (importar como UTF-8, separador coma) o con un editor de CSV. Hay 42 filas: una por pregunta y modo (`no_rag`, `rag`, `rag_advanced`).

| Columna | Para qué sirve |
|---|---|
| `id`, `category`, `mode` | Qué pregunta y qué sistema |
| `question`, `gold_answer` | Lo que se preguntó y la respuesta correcta de referencia |
| `answer` | Lo que respondió el sistema: **lo que hay que juzgar** |
| `gold_sources`, `answer_sources` | Documentos correctos frente a los citados o recuperados |
| `auto_numeric_match`, `auto_citation_ok`, `auto_behavior_ok` | La opinión automática: **mirarla después de decidir, no antes** |
| **`score_0_1_2`** | **Vuestra nota (rellenar)** |
| **`citation_correct_Y_N`** | **Vuestra valoración de la cita (rellenar en `rag` y `rag_advanced`)** |
| `notes` | Una frase de motivo, sobre todo en los 0 y en los desacuerdos |

### Escala `score_0_1_2`
| Nota | Significado | Ejemplo |
|---|---|---|
| **0** | Incorrecta, inventada o no responde cuando debía | D1 sin RAG: "no sé la tasa de cada país" (no responde); o una cifra distinta de la de referencia |
| **1** | Parcialmente correcta, vaga o con un error menor | Da 32,4 % para Alemania pero se equivoca en Francia; o acierta la idea sin cifra cuando se pedía cifra |
| **2** | Correcta y específica, coherente con la referencia | D1 con RAG: Alemania 32,4 %, Francia 16,2 %, España 16,7 % |

Reglas para puntuar igual entre personas:
- Se puntúa **contra `gold_answer`**, no contra lo que uno cree saber.
- **Preguntas U (comportamiento):** U1 (Portugal) y U2 (apellido) se puntúan 2 si el sistema dice que no lo sabe o se niega, y 0 si inventa o revela un dato. Un "no lo sé" correcto **es un 2**, no un 0.
- Una **abstención en una pregunta que sí tenía respuesta** (por ejemplo D1 sin RAG) es 0: la respuesta correcta existía y no se dio.
- Las cifras equivalentes cuentan (`32,4 %` = `0.324` = `32.4%`); el redondeo razonable no resta.
- Que la respuesta esté en español **no influye** en la nota.
- Si dudáis entre dos notas, poned la más baja y explicadlo en `notes`.

### Escala `citation_correct_Y_N` (solo `rag` y `rag_advanced`)
- **Y**: las citas `[doc_id]` de la respuesta apuntan a documentos que **de verdad respaldan** lo que se afirma (comprobad la fuente en `answer_sources` y, si hace falta, el texto del trozo en `eval/results.json`).
- **N**: no cita nada, cita un documento que no dice eso, o cita uno correcto pero la cifra no está en él.
- En `no_rag` se deja vacío (no hay fuentes).

### Método recomendado (calidad de la medición)
1. **Dos personas puntúan por separado** todas las filas, en copias distintas del CSV.
2. Comparar y **discutir solo los desacuerdos**; anotar el motivo en `notes`. Si queréis, calculad el porcentaje de coincidencia y mencionadlo en la memoria.
3. **No mirar las columnas `auto_*` ni el nombre del modo mientras puntuáis**, si podéis: ocultad la columna `mode` y puntuad cada respuesta a ciegas. Reduce el sesgo de favorecer al RAG.
4. Guardar el CSV final como `eval/manual_scoring.csv` (UTF-8, mismas columnas, sin reordenarlas).

## 4. Cómo verificar L1 y L2 (`eval/literature_verification.md`)

Para cada una de las dos preguntas:

1. Abrir el PDF (enlace en el archivo): L1 <https://arxiv.org/pdf/2301.13099>, L2 <https://arxiv.org/pdf/2606.07582>.
2. Leer la **pregunta** y el **borrador de gold** del archivo.
3. En el PDF, localizar las frases que cita la lista "Supporting sentences" (llevan el número de página) y comprobar:
   - que la frase **existe** y está en esa página;
   - que **respalda cada afirmación** del borrador. Hay que cubrir todas, no solo la principal.
     - **L1** — mejor clasificador: ¿el paper compara seis técnicas supervisadas, con 10 atributos de 10.000 clientes de bancos europeos, y concluye que la **ANN con cinco nodos en una capa oculta** es la mejor?
     - **L2** — arquitectura: ¿propone un **híbrido FT-Transformer + árboles con gradient boosting mediante *stacking* consciente de la calibración**, y dice que aborda carencias en **validación estadística, calibración de probabilidades y reproducibilidad**?
   - que el borrador **no afirma de más** (por ejemplo, un nombre o cifra que no aparece en el PDF).
4. Si algo no cuadra: corregir `gold_answer` en `eval/questions.json` (entradas `L1` y `L2`) con una frase que respalde el PDF.
5. Si está correcto: **quitar el prefijo `DRAFT, human must verify against the PDF: `** del `gold_answer` de L1 y L2 en `eval/questions.json`. Esa edición deja constancia de que se revisó. Después ejecutar `uv run python -m eval.score` para que la hoja manual muestre la referencia ya verificada (conserva vuestras notas).
6. Marcar la casilla `- [x] Verified by a human against the PDF` en `eval/literature_verification.md`, con iniciales y fecha, y subirlo a git.

Una segunda persona debería **repetir la comprobación de al menos una de las dos** (es rápido: las frases ya están localizadas).

## 5. Agregar los resultados y publicarlos

```bash
uv run python -m eval.score --summary
```
Escribe `eval/scores_summary.csv`: media de `score_0_1_2` y citas correctas (`Y/total`) por categoría y modo. Es la tabla que se presenta:

| Categoría | Modo | Nota media (0-2) | Citas correctas |
|---|---|---|---|
| D | sin RAG / RAG / RAG avanzado | … | … |

Para ejemplos concretos antes/después en la memoria: `uv run python -m eval.score --examples --example-ids D1,R3` (genera `eval/examples.md`).

Después:
1. Reconstruir el sitio (`uv run quarto render docs`) y comprobar que `04-evaluacion` ya no muestra "pendiente" y que su tabla coincide con `scores_summary.csv`.
2. Subir en un pull request `eval/manual_scoring.csv`, `eval/scores_summary.csv`, `eval/questions.json` y `eval/literature_verification.md`. El workflow de GitHub Pages republica el sitio al fusionar.
3. Actualizar la tabla de estado en `00_INDICE.md` y las limitaciones de `04` y `08`.

## 6. Qué decir en la defensa

- *¿Cómo puntuasteis?* Dos personas, escala 0/1/2 contra una referencia generada, a ciegas respecto al modo, con los desacuerdos discutidos. La puntuación automática es solo una segunda opinión.
- *¿Por qué no basta la puntuación automática?* Mide presencia de cifras y citas, no si la respuesta es correcta en su conjunto ni si la cita respalda lo afirmado.
- *¿Quién verificó los gold de L1 y L2 y cómo?* Una persona comparó cada afirmación con las frases del PDF (con página) y firmó en `literature_verification.md`.
- *¿Por qué el "no lo sé" de U puntúa 2?* Porque era el comportamiento correcto: esa pregunta no tenía respuesta en el corpus o pedía un dato personal.
- *¿Qué limitaciones tiene?* 14 preguntas y una ejecución por pregunta; Gemini 3.5 ignora `temperature`, así que otra ejecución daría respuestas algo distintas.
