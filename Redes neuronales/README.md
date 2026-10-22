# Bank Churn Neural Network

Proyecto de clasificación binaria para predecir si un cliente bancario
abandonará el banco (`Exited`). El flujo activo usa PyTorch Lightning para el
entrenamiento, scikit-learn para el preprocesamiento y Quarto para la
documentación reproducible y los notebooks.

## Estructura

```text
.
├── data/
│   ├── raw/Customer-Churn-Records.csv
│   └── processed/df_limpio.csv
├── notebooks/                  # Informes y análisis Quarto
├── artifacts/                  # Estado y salidas regenerables
│   └── optuna/optuna.db       # Estudio persistente de Optuna
├── src/
│   ├── preprocessing.py        # Limpieza, split y LightningDataModule
│   ├── model.py                # FocalLoss y ChurnLightningModule
│   ├── callbacks.py            # Selección del threshold por F1
│   ├── lightning_config.py     # Compatibilidad de warnings externos
│   ├── train_lightning.py      # Entrenamiento final
│   └── optuna_search.py        # Búsqueda de hiperparámetros
└── tests/
		└── test_preprocessing.py
```

Los checkpoints, logs de Lightning, cachés de Quarto y entornos virtuales
son artefactos locales y no forman parte del código fuente.

## Instalación

Se recomienda Python 3.12 o 3.13 para maximizar la compatibilidad con el
stack científico. Python 3.14 puede funcionar con versiones recientes, pero
puede mostrar warnings internos de dependencias.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Quarto CLI es una dependencia de desarrollo importante. Debe instalarse desde
[quarto.org](https://quarto.org/docs/get-started/), y se comprueba con:

```bash
quarto check
```

## Datos y preprocesamiento

`load_clean()` carga el CSV, elimina identificadores administrativos y valida
el target binario. `clean_raw()` implementa esa limpieza sin efectos sobre el
DataFrame original.

`build_preprocessor()` crea un `ColumnTransformer`:

- Variables financieras: imputación, `log1p`, recorte IQR y `RobustScaler`.
- Otras numéricas: imputación, recorte IQR y `RobustScaler`.
- Categóricas: imputación por moda y One-Hot Encoding con categorías nuevas
	ignoradas.

`ChurnDataModule` hace un split estratificado 80/10/10, ajusta el
preprocesador únicamente con train y entrega `DataLoader` para train,
validación y test. El modelo recibe 19 features y `Complain` se elimina antes
del entrenamiento para evitar leakage.

## Modelo y métricas

`ChurnLightningModule` contiene una capa de entrada, bloques residuales
`Linear -> BatchNorm1d -> Mish -> Dropout -> Linear`, y una salida logit.

La pérdida puede ser:

- `focal`: recomendada como punto de partida para el desbalance.
- `weighted_bce`: `BCEWithLogitsLoss` con `pos_weight`.

Se registran `BinaryAccuracy`, `BinaryPrecision`, `BinaryRecall`, `BinaryF1Score`
y `BinaryAUROC`. `F1ThresholdCallback` calcula la curva precision-recall en
validación, selecciona el threshold que maximiza F1 y lo usa en test.

## Entrenamiento final

Entrenamiento con los hiperparámetros por defecto:

```bash
python -m src.train_lightning --epochs 100 --batch-size 64
```

El valor predeterminado es `--num-workers 0`, adecuado para este dataset y
especialmente seguro para búsquedas de hiperparámetros. En un entrenamiento
final puedes probar `--num-workers 2` si el rendimiento lo requiere.

El mejor checkpoint se guarda en `artifacts/logs/` y el threshold se recalcula
con la validación del checkpoint seleccionado antes del test.

## Optuna

La búsqueda explora número de bloques, dimensión oculta, learning rate, weight
decay, dropout, tipo de pérdida, gamma y alpha focales.

```bash
python -m src.optuna_search --trials 30
```

Optuna ejecuta los trials con un único proceso de carga, evalúa cada
configuración con varias semillas, usa pruning y guarda el estudio en SQLite:

```text
artifacts/optuna/optuna.db
```

La base permite reanudar la búsqueda sin perder los trials anteriores. Para
reentrenar con el mejor trial:

```bash
python -m src.train_lightning \
	--use-optuna-best \
	--optuna-db artifacts/optuna/optuna.db \
	--optuna-study churn_nn_qmd \
	--epochs 100
```

Para elegir el modelo final, prioriza `value` (F1 medio de validación con
threshold optimizado), revisa precision y recall, y confirma el resultado con
varias semillas antes de informar el rendimiento en test.

## Tests

```bash
pytest tests/ -v
```

Los tests comprueban limpieza, target binario, ausencia de nulos y duplicados,
split estratificado, eliminación de leakage y dimensión final de entrada.

## Documentación y notebooks con Quarto

Quarto es el canal oficial para documentación futura. Los informes y análisis
interactivos deben vivir en `notebooks/` como `.qmd`; evita duplicar contenido
en archivos Markdown sueltos.

Renderizar el proyecto completo:

```bash
quarto render
```

Generar el informe HTML completo de resultados desde la raíz del proyecto:

```bash
quarto render notebooks/proyecto_completo.qmd
```

El HTML se genera en `dist/notebooks/proyecto_completo.html`. El informe
ejecuta el preprocesamiento, entrena un modelo base y puede ejecutar Optuna de
forma opcional:

```bash
OPTUNA_ENABLED=1 OPTUNA_TRIALS=30 quarto render notebooks/proyecto_completo.qmd
```

Para renderizar solo el modelo base:

```bash
OPTUNA_ENABLED=0 quarto render notebooks/proyecto_completo.qmd
```

Previsualizar mientras se edita:

```bash
quarto preview
```

Crear un nuevo notebook Quarto:

```bash
quarto create-project notebooks/nuevo_analisis --type default
```

Los documentos pueden ejecutar código Python y reutilizar directamente
`load_clean()`, `ChurnDataModule` y los módulos de entrenamiento. El resultado
HTML se genera en `dist/`, que permanece fuera del control de versiones.
