# 11. Verificación de L1 y L2: por qué importa y cómo se hizo

**Estado: hecha el 26 de noviembre de 2026.** Se verificaron las dos referencias contra el PDF, se quitó el prefijo `DRAFT` de `eval/questions.json` y se actualizaron las páginas de las frases de L2 (las dos últimas están en la p. 20, la conclusión). Este archivo explica el motivo y el método, por si alguien pregunta o hay que repetirlo.

Es la única comprobación humana de la evaluación. La evaluación es **solo automática** (decisión del equipo: no hay puntuación manual), así que la fiabilidad de las dos preguntas de literatura depende de que sus respuestas de referencia estén comprobadas. Tiempo estimado: **15-30 minutos**.

## 1. Por qué importa

- Las referencias (gold) de D, R y U **se generan del dato** (`stats.json` y regex sobre el texto ingerido). Las de **L1 y L2 no**: son borradores escritos a partir de los *resúmenes* de arXiv y llevan el prefijo `DRAFT, human must verify against the PDF` en `eval/questions.json`.
- Si un borrador fuera incorrecto, L1 y L2 se medirían contra una referencia falsa, y con ellas `hit@k` y `citation_ok` de esas preguntas.
- En la defensa se dice que "ningún gold se escribe a mano salvo L1/L2, que verifica una persona". Esta verificación hace cierta esa frase.
- Como no hay puntuación manual, **este es el único control humano** de la evaluación.

## 2. Cómo hacerla (`eval/literature_verification.md`)

Para cada pregunta:

1. Abrir el PDF: L1 <https://arxiv.org/pdf/2301.13099>, L2 <https://arxiv.org/pdf/2606.07582>.
2. Leer la **pregunta** y el **borrador de gold** en `eval/literature_verification.md`.
3. Localizar en el PDF las frases de la lista "Supporting sentences" (con número de página) y comprobar:
   - que cada frase **existe** en esa página;
   - que **respalda cada afirmación** del borrador (todas, no solo la principal):
     - **L1:** el paper compara seis técnicas supervisadas con 10 atributos de 10.000 clientes de bancos europeos y concluye que la **ANN con cinco nodos en una capa oculta** es la mejor.
     - **L2:** propone un **híbrido FT-Transformer + árboles con gradient boosting mediante *stacking* consciente de la calibración** y dice abordar carencias en **validación estadística, calibración de probabilidades y reproducibilidad**;
   - que el borrador **no afirma de más** (ningún nombre o cifra ausente del PDF).
4. Si algo no cuadra, corregir el `gold_answer` de L1 o L2 en `eval/questions.json` con una frase que respalde el PDF.
5. Si es correcto, **quitar el prefijo `DRAFT, human must verify against the PDF: `** del `gold_answer` de L1 y L2 en `eval/questions.json`. Esa edición deja constancia de la revisión.
6. Marcar `- [x] Verified by a human against the PDF` en `eval/literature_verification.md` con iniciales y fecha.
7. Ejecutar `uv run python -m eval.score` para refrescar `eval/auto_summary.csv` con la referencia ya verificada, y abrir un pull request con `eval/questions.json`, `eval/literature_verification.md` y los CSV de `eval/`.

Idealmente una segunda persona repite la comprobación de al menos una de las dos (las frases ya están localizadas, es rápido).

## 3. Qué decir en la defensa

- *¿Quién verificó los gold de L1 y L2 y cómo?* Una persona comparó cada afirmación con las frases del PDF (con página) y firmó en `literature_verification.md`.
- *¿Por qué no hay puntuación manual?* Decisión del equipo por tiempo; la puntuación automática compara cifras generadas desde el dato, comprueba citas contra los trozos recuperados y detecta abstenciones y fugas de datos personales. Limitación: no juzga la calidad global de una respuesta.
- *¿Qué limitaciones tiene?* 14 preguntas y una ejecución por pregunta; Gemini 3.5 ignora `temperature`, así que otra ejecución daría respuestas algo distintas.
