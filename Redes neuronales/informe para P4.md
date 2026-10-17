ara la Persona 4 (Informe / Quarto)
Aquí tienes toda la justificación técnica y las métricas exactas para que las cites en el informe. He explicado el "por qué" de las decisiones para que quede muy profesional.

1. El Dataset Limpio
Dimensiones: 10.000 filas × 15 columnas.

Limpieza: Se eliminaron las columnas RowNumber, CustomerId y Surname por ser identificadores administrativos que no aportan valor predictivo.

Sobre Complain: La columna se conserva en el CSV limpio (para el EDA), pero el modelo NO la usa. Se elimina antes del split por fuga de información (leakage), ya que si un cliente se queja, casi con total seguridad se va a ir. Pero en la vida real, no tienes esa queja antes de predecir.

2. Tasa de Churn y Split
Tasa de Churn: El dataset tiene un 20,38% de clientes que se van (clase 1). Esto justifica por qué NO se debe usar solo el Accuracy como métrica (un modelo que diga siempre "se queda", tendría un 80% de accuracy pero sería inútil). Mejor usar F1-Score, Recall o ROC-AUC.

Split: 80% Train / 10% Validation / 10% Test.

Estratificación (stratify=y): Se ha forzado a que la proporción de 20% churn / 80% no-churn se mantenga exactamente igual en los tres conjuntos. Verificado por test.

Clases en Train: 6.370 (no-churn) / 1.630 (churn).

3. Pipeline de Preprocesamiento (Las transformaciones)
Usamos un ColumnTransformer (una cadena de montaje) donde el fit se hace SOLO con el conjunto de Train (prevención estricta de fuga de datos):

Variables Financieras (Balance, EstimatedSalary): Imputación por mediana → Transformación logarítmica (log1p) para comprimir los valores muy altos → Recorte de outliers (IQRClipper) → Escalado robusto (RobustScaler).

Otras variables numéricas (Edad, Puntuación, etc.): Imputación por mediana → Recorte de outliers → Escalado robusto.

Variables categóricas (Geography, Gender, Card Type): Imputación por la moda (el valor más frecuente) → OneHotEncoder (con handle_unknown="ignore" para que no falle si aparece un país nuevo en Test).

4. Desglose de las 19 Features finales
Tras todo el proceso, la red neuronal recibe exactamente 19 columnas:

10 numéricas (se ha quitado Complain).

9 generadas por OneHot: Geography (3 países) + Gender (2 géneros) + Card Type (4 tipos de tarjeta) = 