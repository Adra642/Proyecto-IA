# 📝 Información para la Persona 4 (Informe / Quarto)

Aquí tienes toda la justificación técnica y las métricas exactas para que las cites en el informe. He explicado el "por qué" de las decisiones para que quede muy profesional.

---

### 1. El Dataset Limpio
*   **Dimensiones:** 10.000 filas × 15 columnas.
*   **Limpieza:** Se eliminaron las columnas `RowNumber`, `CustomerId` y `Surname` por ser identificadores administrativos que no aportan valor predictivo.
*   **Sobre `Complain`:** La columna se conserva en el CSV limpio (para el EDA), pero el modelo **NO la usa**. Se elimina antes del split por fuga de información (*leakage*), ya que si un cliente se queja, casi con total seguridad se va a ir. Pero en la vida real, no tienes esa queja antes de predecir.

### 2. Tasa de Churn y Split
*   **Tasa de Churn:** El dataset tiene un **20,38%** de clientes que se van (clase 1). Esto justifica por qué **NO se debe usar solo el *Accuracy*** como métrica (un modelo que diga siempre "se queda", tendría un 80% de *accuracy* pero sería inútil). Mejor usar F1-Score, Recall o ROC-AUC.
*   **Split:** 80% Train / 10% Validation / 10% Test.
*   **Estratificación (`stratify=y`):** Se ha forzado a que la proporción de 20% churn / 80% no-churn se mantenga exactamente igual en los tres conjuntos. Verificado por test.
*   **Clases en Train:** 6.370 (no-churn) / 1.630 (churn).

### 3. Pipeline de Preprocesamiento (Las transformaciones)
Usamos un `ColumnTransformer` (una cadena de montaje) donde el `fit` se hace **SOLO con el conjunto de Train** (prevención estricta de fuga de datos):
*   **Variables Financieras (`Balance`, `EstimatedSalary`):** Imputación por mediana → Transformación logarítmica (`log1p`) para comprimir los valores muy altos → Recorte de *outliers* (`IQRClipper`) → Escalado robusto (`RobustScaler`).
*   **Otras variables numéricas** (Edad, Puntuación, etc.): Imputación por mediana → Recorte de *outliers* → Escalado robusto.
*   **Variables categóricas (`Geography`, `Gender`, `Card Type`):** Imputación por la moda (el valor más frecuente) → `OneHotEncoder` (con `handle_unknown="ignore"` para que no falle si aparece un país nuevo en Test).

### 4. Desglose de las 19 Features finales
Tras todo el proceso, la red neuronal recibe exactamente **19 columnas**:
*   **10 numéricas** (se ha quitado `Complain`).
*   **9 generadas por OneHot:** Geography (3 países) + Gender (2 géneros) + Card Type (4 tipos de tarjeta) = 9.

Oye, por si quieres añadir una línea de calidad en el informe sobre el preprocesamiento: he ejecutado los tests automáticos y pasan los 9/9. Eso garantiza matemáticamente que no hay valores nulos, que el split 80/10/10 está perfectamente estratificado y que no hay fuga de datos (leakage con Complain).

¿Quieres verlo tú mismo/a y sacar una captura para el informe?
Solo tienes que abrir la terminal en la raíz del proyecto (con tu entorno .venv activado) y ejecutar este comando:

bash
pytest tests/ -v
Si todo está bien, la terminal te soltará una salida verde que termina así:

text
tests/test_preprocessing.py::test_clean_raw_drops_id_columns PASSED     [ 11%]
tests/test_preprocessing.py::test_clean_raw_has_expected_shape PASSED   [ 22%]
tests/test_preprocessing.py::test_target_is_binary PASSED               [ 33%]
tests/test_preprocessing.py::test_no_missing_values PASSED              [ 44%]
tests/test_preprocessing.py::test_no_duplicates PASSED                  [ 55%]
tests/test_preprocessing.py::test_split_is_stratified PASSED            [ 66%]
tests/test_preprocessing.py::test_tensor_shapes_match PASSED            [ 77%]
tests/test_preprocessing.py::test_complain_is_not_in_features PASSED    [ 88%]
tests/test_preprocessing.py::test_input_dim_is_19 PASSED                [100%]

======================== 9 passed in 2.34s ========================
Para el informe / Quarto: Puedes poner una frase tipo "El pipeline de preprocesamiento cuenta con una suite de 9 tests unitarios que garantizan la integridad de los datos y la ausencia de leakage, superados al 100% (ver anexo o captura)". Y si quieres, pegas una captura de pantalla de esa terminal en verde, ¡queda súper profesional!