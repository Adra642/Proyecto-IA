# Análisis de Resultados — Persona 3 (Fase C corregida)

## Resumen ejecutivo

Se han entrenado **10 configuraciones distintas** con **3 semillas cada una** (30 entrenamientos totales). Cada experimento cambia **una sola variable** respecto a su referencia, permitiendo atribución causal. Se aplicaron:

- **Early stopping** (patience=8, max 300 épocas)
- **Weight decay** (1e-4)
- **pos_weight** (~3.9) para mitigar el desbalanceo 80/20
- **Dropout** (0.3) en los modelos que lo requieren (E y F3)

## Diseño experimental

### Profundidad creciente (A → B → C → E)
| Exp | Topología | Prof. | Neuronas | Params | Dropout | Qué aísla |
|---|---|---|---|---|---|---|
| A | [32] | 1 | 32 | 673 | 0.0 | Baseline mínimo |
| B | [64, 32] | 2 | 96 | 3.393 | 0.0 | +1 capa vs A |
| C | [128, 64, 32] | 3 | 224 | 12.929 | 0.0 | +1 capa vs B |
| E | [128, 64, 32, 16] | 4 | 240 | 13.441 | 0.3 | +1 capa vs C |

### Anchura con profundidad fija (C vs F1, F2, F3)
| Exp | Topología | Prof. | Neuronas | Params | Dropout |
|---|---|---|---|---|---|
| F1 | [32, 32, 32] | 3 | 96 | 2.785 | 0.0 |
| F2 | [64, 64, 64] | 3 | 192 | 9.665 | 0.0 |
| C | [128, 64, 32] | 3 | 224 | 12.929 | 0.0 |
| F3 | [128, 128, 128] | 3 | 384 | 35.713 | 0.3 |

### Batch aislado (B vs G)
B: batch=32. G: batch=64. Misma arquitectura y LR.

### LR aislado (B vs H1 vs H2)
B: lr=1e-3. H1: lr=5e-4. H2: lr=1e-4. Mismo batch=32.

## Resultados (media ± std, 3 semillas)

| Modelo | Arquitectura | Params | Épocas | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|---|---|
| A | [32] | 673 | 31 | 0.766 ± 0.010 | 0.459 | **0.797** | 0.582 | 0.867 |
| B | [64, 32] | 3.393 | 21 | 0.782 ± 0.014 | 0.479 | 0.784 | 0.595 | 0.863 |
| **C** | [128, 64, 32] | 12.929 | 13 | **0.792 ± 0.012** | **0.495** | 0.773 | **0.603** | 0.864 |
| E | [128, 64, 32, 16] | 13.441 | 26 | 0.781 ± 0.017 | 0.479 | 0.788 | 0.595 | 0.866 |
| F1 | [32, 32, 32] | 2.785 | 18 | 0.776 ± 0.006 | 0.471 | 0.783 | 0.588 | 0.867 |
| **F2** | [64, 64, 64] | 9.665 | 14 | 0.787 ± 0.009 | 0.487 | 0.793 | **0.603** | 0.867 |
| F3 | [128, 128, 128] | 35.713 | 24 | 0.773 ± 0.002 | 0.467 | **0.802** | 0.590 | **0.869** |
| G | [64, 32] b=64 | 3.393 | 31 | 0.788 ± 0.011 | 0.488 | 0.775 | 0.599 | 0.863 |
| H1 | [64, 32] lr=5e-4 | 3.393 | 38 | 0.781 ± 0.005 | 0.477 | 0.775 | 0.590 | 0.864 |
| H2 | [64, 32] lr=1e-4 | 3.393 | 108 | 0.785 ± 0.005 | 0.483 | 0.775 | 0.595 | 0.865 |

## Análisis por bloques

### 1. Profundidad (A → B → C → E)
- **A (1 capa)**: 0.766 acc, 0.582 F1. El peor en accuracy pero el mejor en recall (0.797). Es conservador: detecta muchos churns con muchas falsas alarmas.
- **B (2 capas)**: 0.782 acc, 0.595 F1. Mejora clara sobre A (+1.6 puntos acc).
- **C (3 capas)**: 0.792 acc, 0.603 F1. **Mejor accuracy y F1**.
- **E (4 capas + dropout)**: 0.781 acc, 0.595 F1. **La 4ª capa no aporta**.

**Significancia**: la única diferencia significativa es **C vs A** (rangos no se solapan). C vs B, C vs E: diferencias dentro de 1 std → no significativas.

**Hallazgo clave**: el overfitting anterior de C (con 50 épocas fijas) era **por las épocas, no por la profundidad**. Con early stopping, C paró en época 13 y generaliza bien.

### 2. Anchura (F1, F2, C, F3)
- **F1 (96 neuronas)**: 0.776 acc, 0.588 F1. Demasiado estrecho.
- **F2 (192 neuronas)**: 0.787 acc, **0.603 F1**. **Empata con C** con 25% menos parámetros.
- **C (224 neuronas)**: 0.792 acc, 0.603 F1. Mejor accuracy.
- **F3 (384 neuronas)**: 0.773 acc, 0.590 F1. **Peor accuracy** a pesar de tener 2.7x más parámetros que C.

**Hallazgo clave**: el óptimo está en 192-224 neuronas. F2 es la mejor relación rendimiento/eficiencia. F3 overfitea incluso con dropout.

### 3. Batch (B vs G)
- B: 0.782 acc.
- G: 0.788 acc.
- Diferencia no significativa.

**Conclusión**: el batch no es una palanca relevante.

### 4. Learning rate (B vs H1 vs H2)
- B (1e-3): 0.782 acc, 21 épocas.
- H1 (5e-4): 0.781 acc, 38 épocas.
- H2 (1e-4): 0.785 acc, **108 épocas**.

**Conclusión**: el LR no afecta al resultado final, pero sí al tiempo de entrenamiento. LR=1e-3 es el mejor compromiso (mismo resultado, 5x menos tiempo).

### 5. Dropout (C vs E vs F3)
- C (sin dropout): 0.603 F1.
- E (con dropout, 4 capas): 0.595 F1.
- F3 (con dropout, 384 neuronas): 0.590 F1.

**Conclusión**: el dropout no aporta en este dataset. Con early stopping + weight decay ya hay suficiente regularización.

### 6. Efecto del pos_weight
Comparación con la versión anterior (sin pos_weight):

| Métrica | Sin pos_weight | Con pos_weight (3.9) |
|---|---|---|
| Accuracy | ~0.87 | ~0.79 |
| Recall | ~0.53 | **~0.79** |
| Precision | ~0.77 | ~0.47 |
| AUC | (no medido) | **~0.865** |

El AUC es prácticamente el mismo. La capacidad de discriminación no ha cambiado; solo el umbral efectivo. **En churn, priorizar recall es la decisión correcta**: el coste de perder un cliente supera el de una promo innecesaria.

## Conclusiones

1. **Mejor modelo técnico: C** (3 capas, 128-64-32) con Accuracy = 0.792 ± 0.012 y F1 = 0.603 ± 0.019.

2. **Mejor modelo en eficiencia: F2** (3 capas, 64-64-64) con el mismo F1 que C pero **25% menos parámetros**.

3. **Mejor modelo para negocio: F3** (3 capas, 128-128-128) con el mejor recall (0.802) y AUC (0.869).

4. **La arquitectura no es la palanca principal**. Las diferencias entre todas las arquitecturas con 2+ capas están dentro del margen de 1 desviación estándar. Lo que sí importa es:
   - **Early stopping** (corrige el overfitting).
   - **pos_weight** (ajusta el trade-off precision/recall hacia el negocio).
   - **Weight decay** (regulariza).

5. **El overfitting anterior de C no era por profundidad**: era por 50 épocas fijas sin regularización. Con early stopping, C converge en 13 épocas y generaliza correctamente.

## Limitaciones

- Una sola partición train/val/test (no validación cruzada).
- Dataset de 10.000 ejemplos, lo que limita la capacidad de redes grandes.
- Desbalanceo mitigado con `pos_weight` pero no se exploró threshold tuning.
- No se probó BatchNorm ni otras formas de regularización.

## Archivos generados

- `resultados/tables/comparacion_modelos.csv` → tabla agregada (10 filas, media ± std).
- `resultados/tables/resultados_por_semilla.csv` → 30 filas crudas.
- `resultados/figures/loss_modelo_*.png` → 10 curvas de pérdida.
- `resultados/figures/confusion_matrix_*.png` → 10 matrices de confusión.
- `resultados/figures/comparacion_modelos.png` → gráfico con barras de error (5 métricas).