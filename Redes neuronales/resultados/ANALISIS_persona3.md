# Análisis de resultados — Persona 3

## Resumen

Se han entrenado 4 arquitecturas de red neuronal feed-forward para el problema
de clasificación binaria de churn bancario. El objetivo era evaluar cómo afecta
la profundidad de la red y los hiperparámetros al rendimiento.

**Nota**: estos resultados se han generado con la versión actualizada del
preprocesamiento (sin la columna `Complain`, que producía leakage). El
`input_dim` final es 19.

## Modelos entrenados

| Modelo | Arquitectura | Parámetros | Batch | Learning Rate | Épocas |
|---|---|---|---|---|---|
| A | [32] | 673 | 32 | 0.001 | 50 |
| B | [64, 32] | 3.393 | 32 | 0.001 | 50 |
| C | [128, 64, 32] | 12.929 | 32 | 0.001 | 50 |
| D | [128, 64, 32] | 12.929 | 64 | 0.0005 | 50 |

## Resultados en test

| Modelo | Test Loss | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **A** | **0.3374** | **0.874** | **0.783** | 0.529 | 0.632 |
| B | 0.3594 | 0.870 | 0.789 | 0.495 | 0.608 |
| C | 0.6181 | 0.838 | 0.613 | 0.559 | 0.585 |
| **D** | 0.3783 | 0.858 | 0.670 | **0.598** | **0.632** |

## Análisis

### Modelo A — el más simple, el mejor en accuracy y precision
- 1 capa oculta de 32 neuronas, 673 parámetros.
- Obtiene el **mejor accuracy (0.874)** y la **mejor precision (0.783)**.
- Las curvas de pérdida muestran que train y validación bajan juntas: buena generalización.
- Su punto débil es el recall (0.529): se le escapan casi la mitad de los churns reales.

### Modelo B — más capacidad, sin mejora
- 2 capas ocultas (64 → 32), 3.393 parámetros.
- Accuracy y F1 similares a A, pero con peor recall (0.495).
- Empieza a mostrar overfitting al final: val loss sube mientras train loss sigue bajando.

### Modelo C — overfitting severo
- 3 capas ocultas (128 → 64 → 32), 12.929 parámetros.
- La pérdida de entrenamiento baja hasta **0.159**, pero la de validación **sube hasta 0.673**.
- El accuracy en test cae a 0.838. La red memoriza el train en lugar de aprender patrones generales.
- Es el peor modelo en todas las métricas excepto recall.

### Modelo D — el mejor recall, empatado en F1 con A
- Misma arquitectura que C, pero con batch 64 y learning rate 0.0005.
- El learning rate más bajo frena el overfitting: val loss sube solo hasta 0.390.
- Obtiene el **mejor recall (0.598)** de los cuatro modelos.
- Su F1 (0.632) está empatado con el de A (0.632).

## Conclusión

Hay **dos modelos empatados en F1**: A (0.6316) y D (0.6321). La elección
depende del objetivo de negocio:

- **Si el objetivo es minimizar falsas alarmas** (no molestar a clientes que
  se van a quedar): elegir **A**. Mejor precision (0.783), mejor accuracy (0.874).
- **Si el objetivo es detectar al máximo de clientes que se van** (aunque haya
  falsas alarmas): elegir **D**. Mejor recall (0.598).

En un caso real de churn bancario, **D es probablemente la mejor opción**,
porque el coste de perder un cliente es mucho mayor que el coste de ofrecerle
una promoción innecesaria a alguien que se iba a quedar.

## Sobre las métricas y el desbalanceo

El accuracy es alto en todos los modelos (~0.84-0.87), pero el F1 de la clase
"Churn" es moderado (~0.58-0.63). Esto es esperable dado el **desbalanceo** del
dataset (80% No Churn / 20% Churn). El modelo tiende a predecir "No Churn"
porque es la clase mayoritaria.

Para mejorar el recall de Churn en el futuro se podría:
- Usar `pos_weight` en `BCEWithLogitsLoss` para penalizar más los falsos negativos.
- Aplicar oversampling de la clase minoritaria.
- Bajar el umbral de decisión de 0.5 a 0.3 o 0.4.

## Gráficos adjuntos

- `figures/loss_modelo_A.png`, `B`, `C`, `D` → curvas de pérdida train/val.
- `figures/confusion_matrix_A.png`, `B`, `C`, `D` → matrices de confusión.
- `figures/comparacion_modelos.png` → comparación visual de métricas.
- `tables/comparacion_modelos.csv` → tabla completa.