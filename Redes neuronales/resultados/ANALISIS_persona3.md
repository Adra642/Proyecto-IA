# Análisis de resultados — Persona 3

## Resumen

Se han entrenado 4 arquitecturas de red neuronal feed-forward para el problema
de clasificación binaria de churn bancario. El objetivo era evaluar cómo afecta
la profundidad de la red y los hiperparámetros al rendimiento.

## Modelos entrenados

| Modelo | Arquitectura | Parámetros | Batch | Learning Rate | Épocas |
|---|---|---|---|---|---|
| A | [32] | 705 | 32 | 0.001 | 50 |
| B | [64, 32] | 3.457 | 32 | 0.001 | 50 |
| C | [128, 64, 32] | 13.057 | 32 | 0.001 | 50 |
| D | [128, 64, 32] | 13.057 | 64 | 0.0005 | 50 |

## Resultados en test

| Modelo | Test Loss | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **A** | **0.3277** | **0.874** | **0.771** | 0.544 | **0.638** |
| B | 0.3727 | 0.854 | 0.671 | **0.559** | 0.610 |
| C | 0.7944 | 0.830 | 0.583 | 0.583 | 0.583 |
| D | 0.3832 | 0.860 | 0.713 | 0.525 | 0.605 |

## Análisis

### Modelo A — el ganador
- Es el modelo más simple (1 capa oculta de 32 neuronas).
- Obtiene el **mejor resultado en todas las métricas principales**: accuracy, precision, F1 y test loss.
- Las curvas de pérdida muestran que train y validación bajan juntas, señal de buena generalización.

### Modelo B — más capacidad, peor resultado
- Pese a tener 2 capas ocultas (64 → 32) y 5 veces más parámetros, **empeora** respecto a A.
- Al final del entrenamiento, la pérdida de validación empieza a subir mientras la de train sigue bajando: primeros síntomas de **overfitting**.

### Modelo C — overfitting severo
- Es el modelo con más capacidad (3 capas, 13.057 parámetros).
- La pérdida de entrenamiento baja hasta 0.11, pero la de validación **sube hasta 0.84**.
- El accuracy en test cae a 0.830. La red **memoriza** el train en lugar de aprender patrones generales.

### Modelo D — mejora ligera respecto a C
- Mismo modelo que C, pero con batch más grande (64) y learning rate más bajo (0.0005).
- El learning rate más bajo frena el overfitting, pero **sigue sin superar al modelo A**.
- Confirma que el problema no es de hiperparámetros, sino de **capacidad excesiva** para este dataset.

## Conclusión

**El mejor modelo es el A.** Es el más simple, el más rápido de entrenar y el que
mejor generaliza. Los modelos más profundos no solo no mejoran, sino que
empeoran por overfitting.

Esto ilustra una lección importante en Machine Learning: **aumentar la complejidad
del modelo no garantiza mejores resultados**. Con 10.000 ejemplos y 20 features,
una red pequeña es suficiente. Para aprovechar una red grande, harían falta más
datos o regularización (dropout, weight decay), que no se han aplicado en esta
primera fase.

## Sobre las métricas

El accuracy es alto en todos los modelos (~0.83-0.87), pero el F1 de la clase
"Churn" es bajo (~0.58-0.64). Esto es esperable dado el **desbalanceo** del
dataset (80% No Churn / 20% Churn). El modelo tiende a predecir "No Churn"
porque es la clase mayoritaria.

De cara a un caso real de negocio, sería más importante **maximizar el recall
de Churn** (detectar al máximo número de clientes que se van) incluso a costa
de bajar la precision. Eso se podría conseguir con:
- `pos_weight` en la función de pérdida.
- Oversampling de la clase minoritaria.
- Umbral de decisión distinto a 0.5.

## Gráficos adjuntos

- `figures/loss_modelo_A.png`, `B`, `C`, `D` → curvas de pérdida train/val.
- `figures/confusion_matrix_A.png`, `B`, `C`, `D` → matrices de confusión.
- `figures/comparacion_modelos.png` → comparación visual de métricas.
- `tables/comparacion_modelos.csv` → tabla completa.