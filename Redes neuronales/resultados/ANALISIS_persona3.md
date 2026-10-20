# Análisis de Resultados — Persona 3 (Fase C corregida)

---

## 1. Resumen ejecutivo

Se han entrenado **10 configuraciones distintas** de red neuronal con **3 semillas cada una** (30 entrenamientos totales). Cada experimento cambia **una sola variable** respecto a su referencia, lo que permite atribución causal de las diferencias observadas. Se aplicaron técnicas de regularización modernas: **early stopping** (patience=8), **weight decay** (1e-4), **pos_weight** (~3.9) para mitigar el desbalanceo 80/20, y **dropout** (0.3) en los modelos que lo requieren por su exceso de capacidad.

### Hallazgos principales

1. **C y F2 comparten el mejor F1 puntual (0.603)**, aunque C tiene mejor accuracy puntual (0.792) y F2 es más eficiente (25% menos parámetros). Las diferencias con B, E, G y H2 están dentro del margen de ±1 desviación estándar. **En la práctica, todas las configuraciones con 2+ capas sin dropout (B, C, F1, F2, G, H1, H2) y E son equivalentes según el criterio ±1 std. F3 es claramente peor que C según este criterio, pero confunde anchura y dropout, así que no se puede atribuir solo a la arquitectura.**

2. **El overfitting anterior de C no era por profundidad**. Con 50 épocas fijas y sin regularización, C memorizaba el train. Con early stopping, C paró en **época 13** y generaliza correctamente. **La conclusión anterior "3 capas → overfitting severo" era falsa**.

3. **El pos_weight ha cambiado el comportamiento del modelo sin cambiar su capacidad de discriminación**. El AUC se mantiene en ~0.865 en todos los modelos, pero el recall sube de ~0.53 a ~0.79 a costa de bajar la precision de ~0.77 a ~0.47. **No es un empeoramiento, es un cambio de objetivo**: priorizar detectar churns sobre no molestar a los que se quedan.

4. **La arquitectura no es la palanca principal en este problema**. **Las diferencias de accuracy entre las configuraciones con 2+ capas son pequeñas (~0.02 puntos en el rango observado), aunque F3 queda por debajo de C fuera del margen ±1 std.** **Lo que sí importa** es el early stopping (corrige overfitting) y el pos_weight (ajusta el trade-off precision/recall).

5. **Los modelos con dropout (E, F3) rinden peor que C**, pero **no hay una comparación limpia que aísle el dropout** (E cambia profundidad y dropout; F3 cambia anchura y dropout). La conclusión sobre el dropout es **tentativa**, no concluyente.

> **Nota metodológica**: a lo largo del documento se usa el criterio de **solapamiento de media ±1 desviación estándar** como criterio exploratorio para comparar modelos. No se realizaron tests estadísticos formales (t-test, Mann-Whitney) por el bajo número de semillas (3). Las afirmaciones de "diferencia clara" o "diferencia no concluyente" se basan en este criterio heurístico, no en significancia estadística formal.

---

## 2. Contexto y motivación: qué se corrigió respecto a la versión anterior

La versión anterior del trabajo (código inicial) presentaba **6 debilidades metodológicas** señaladas por el profesor:

| Debilidad anterior | Corrección aplicada |
|---|---|
| 4 experimentos, pero solo 3 arquitecturas (D era variante de C) | **7 arquitecturas distintas** (A, B, C, E, F1, F2, F3) + **3 variantes de B** (G, H1, H2) = 10 experimentos |
| D cambiaba batch y LR a la vez → no se podía aislar el efecto | **Cada experimento cambia una sola variable** (ablación real) |
| Una sola semilla (RANDOM_STATE=42) | **3 semillas** ([0, 1, 2]) con media ± std |
| Sin early stopping (50 épocas fijas) | **Early stopping con patience=8**, max 300 épocas |
| Dropout no se usaba aunque ChurnNet lo soporta | **Dropout 0.3** en E y F3, pasado correctamente desde train.py |
| Desbalanceo detectado pero no mitigado | **pos_weight = 3.9** en BCEWithLogitsLoss |

Además, se añadieron dos mejoras mencionadas por el profesor en la reunión:

- **Tabla numérica de pérdidas finales** (Train_Loss_final, Val_Loss_final, Test_Loss).
- **AUC-ROC** como métrica adicional (más robusta que accuracy para comparar modelos).

---

## 3. Diseño experimental: justificación de cada decisión

### 3.1 Rejilla de 10 experimentos

| Exp | Topología | Prof. | Neuronas | Batch | LR | Dropout | Qué aísla |
|---|---|---|---|---|---|---|---|
| **A** | [32] | 1 | 32 | 32 | 1e-3 | 0.0 | Baseline mínimo |
| **B** | [64, 32] | 2 | 96 | 32 | 1e-3 | 0.0 | +1 capa vs A |
| **C** | [128, 64, 32] | 3 | 224 | 32 | 1e-3 | 0.0 | +1 capa vs B |
| **E** | [128, 64, 32, 16] | 4 | 240 | 32 | 1e-3 | 0.3 | +1 capa vs C (con dropout) |
| **F1** | [32, 32, 32] | 3 | 96 | 32 | 1e-3 | 0.0 | Anchura estrecha vs C |
| **F2** | [64, 64, 64] | 3 | 192 | 32 | 1e-3 | 0.0 | Anchura media vs C |
| **F3** | [128, 128, 128] | 3 | 384 | 32 | 1e-3 | 0.3 | Anchura grande vs C (con dropout) |
| **G** | [64, 32] | 2 | 96 | **64** | 1e-3 | 0.0 | Solo batch vs B |
| **H1** | [64, 32] | 2 | 96 | 32 | **5e-4** | 0.0 | Solo LR vs B |
| **H2** | [64, 32] | 2 | 96 | 32 | **1e-4** | 0.0 | Solo LR más bajo vs B |

### 3.2 Cobertura de los requisitos del enunciado

| Requisito | Cómo se cumple |
|---|---|
| **Número de capas ocultas** | 4 profundidades: A (1), B (2), C/F1/F2/F3 (3), E (4) |
| **Número de neuronas por capa** | 3 anchuras con misma profundidad y sin dropout: F1 (96), F2 (192), C (224) |
| **Batch size** | B (32) vs G (64), solo cambia el batch |
| **Learning rate** | B (1e-3) vs H1 (5e-4) vs H2 (1e-4), solo cambia el LR |
| **Ablación** | Cada experimento cambia una variable respecto a su referencia |

### 3.3 Limitaciones del diseño experimental

**Importante**: no todas las comparaciones son perfectamente limpias.

- **Anchura**: F1, F2 y C son comparables directamente (misma profundidad 3, sin dropout). **F3 no es directamente comparable** porque además lleva `dropout=0.3`. Su peor rendimiento puede deberse a la anchura excesiva, al dropout, o a ambos. Para aislar la anchura en el rango alto haría falta un F3 sin dropout (no incluido en la rejilla).
- **Dropout**: no hay una comparación limpia. E cambia profundidad y dropout (C→E). F3 cambia anchura y dropout (C→F3). Para aislar el dropout haría falta `C+dropout` vs `C`, que no existe en la rejilla. La conclusión sobre el dropout es **tentativa**.

---

## 4. Resultados completos (media ± std, 3 semillas)

| Modelo | Arquitectura | Params | Épocas | Train Loss | Val Loss | Test Loss | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | [32] | 673 | 31 | 0.719 ± 0.001 | 0.746 ± 0.002 | 0.732 ± 0.004 | 0.766 ± 0.010 | 0.459 ± 0.013 | 0.797 ± 0.012 | 0.582 ± 0.008 | 0.867 ± 0.002 |
| B | [64, 32] | 3.393 | 21 | 0.696 ± 0.001 | 0.741 ± 0.006 | 0.733 ± 0.006 | 0.782 ± 0.014 | 0.479 ± 0.019 | 0.784 ± 0.010 | 0.595 ± 0.012 | 0.863 ± 0.003 |
| **C** | [128, 64, 32] | 12.929 | 13 | 0.733 ± 0.006 | 0.743 ± 0.005 | 0.735 ± 0.011 | **0.792 ± 0.012** | **0.495 ± 0.019** | 0.773 ± 0.016 | **0.603 ± 0.019** | 0.864 ± 0.004 |
| E | [128, 64, 32, 16] | 13.441 | 26 | 0.740 ± 0.016 | 0.742 ± 0.004 | 0.727 ± 0.008 | 0.781 ± 0.017 | 0.479 ± 0.023 | 0.788 ± 0.015 | 0.595 ± 0.015 | 0.866 ± 0.004 |
| F1 | [32, 32, 32] | 2.785 | 18 | 0.719 ± 0.001 | 0.745 ± 0.010 | 0.727 ± 0.011 | 0.776 ± 0.006 | 0.471 ± 0.008 | 0.783 ± 0.012 | 0.588 ± 0.004 | 0.867 ± 0.003 |
| **F2** | [64, 64, 64] | 9.665 | 14 | 0.690 ± 0.024 | 0.749 ± 0.006 | 0.732 ± 0.017 | 0.787 ± 0.009 | 0.487 ± 0.014 | 0.793 ± 0.012 | **0.603 ± 0.007** | 0.867 ± 0.006 |
| F3 | [128, 128, 128] | 35.713 | 24 | 0.728 ± 0.014 | 0.747 ± 0.004 | 0.724 ± 0.008 | 0.773 ± 0.002 | 0.467 ± 0.002 | **0.802 ± 0.006** | 0.590 ± 0.001 | **0.869 ± 0.003** |
| G | [64, 32] b=64 | 3.393 | 31 | 0.697 ± 0.013 | 0.742 ± 0.002 | 0.726 ± 0.003 | 0.788 ± 0.011 | 0.488 ± 0.016 | 0.775 ± 0.015 | 0.599 ± 0.015 | 0.863 ± 0.001 |
| H1 | [64, 32] lr=5e-4 | 3.393 | 38 | 0.696 ± 0.016 | 0.743 ± 0.009 | 0.734 ± 0.004 | 0.781 ± 0.005 | 0.477 ± 0.007 | 0.775 ± 0.018 | 0.590 ± 0.008 | 0.864 ± 0.002 |
| H2 | [64, 32] lr=1e-4 | 3.393 | 108 | 0.700 ± 0.011 | 0.746 ± 0.012 | 0.732 ± 0.002 | 0.785 ± 0.005 | 0.483 ± 0.008 | 0.775 ± 0.010 | 0.595 ± 0.008 | 0.865 ± 0.002 |

> **Nota**: los valores de std en esta tabla están redondeados a 3 decimales. Los valores sin redondear (usados en la sección 8) son, por ejemplo, F3: std(accuracy) = 0.0015, std(F1) = 0.0006. Los rankings de estabilidad son idénticos con ambos redondeos.

---

## 5. Análisis detallado por bloques

### 5.1 Efecto de la profundidad (A → B → C → E)

**Pregunta**: ¿Aumentar el número de capas ocultas mejora el rendimiento?

**Respuesta**: **La profundidad mejora de forma clara al comparar A (1 capa) con C (3 capas). Las transiciones individuales A→B, B→C y C→E no son concluyentes según el criterio ±1 std. No hay evidencia de que una 4ª capa aporte mejora; E no supera a C.**

| Transición | Cambio en Accuracy | Cambio en F1 | Interpretación |
|---|---|---|---|
| A → B (+1 capa) | +0.015 | +0.013 | **Mejora puntual**, pero los intervalos ±1 std se solapan → **no concluyente** |
| B → C (+1 capa) | +0.011 | +0.008 | **No concluyente** (los intervalos ±1 std se solapan) |
| C → E (+1 capa) | -0.011 | -0.008 | **Empeora ligeramente**, pero no concluyente |

**Comparación según criterio ±1 std**:

- **A vs B**: A = [0.756, 0.776], B = [0.768, 0.796]. **Se solapan** en [0.768, 0.776] → **diferencia no concluyente**.
- **A vs C**: A = [0.756, 0.776], C = [0.780, 0.804]. **No se solapan** → **diferencia clara** según el criterio ±1 std. C es mejor que A.
- **B vs C**: B = [0.768, 0.796], C = [0.780, 0.804]. **Se solapan** en [0.780, 0.796] → **diferencia no concluyente**.
- **C vs E**: C = [0.780, 0.804], E = [0.764, 0.798]. **Se solapan** en [0.780, 0.798] → **diferencia no concluyente**.

**Conclusión**: **A es claramente peor que C** según el criterio ±1 std. **Las transiciones individuales A→B, B→C y C→E no son concluyentes**. **No hay evidencia de que una 4ª capa aporte mejora**; E no supera a C. **El overfitting anterior de C no era por profundidad, era por 50 épocas fijas sin regularización**.

---

### 5.2 Efecto de la anchura (F1 → F2 → C)

**Pregunta**: Con la misma profundidad (3 capas), ¿cómo afecta el número de neuronas por capa?

**Respuesta**: **F2 y C empatan en F1 y son equivalentes. F1 muestra peor accuracy puntual, pero la diferencia con C es marginal y no concluyente con 3 semillas. Los datos sugieren que 192-224 neuronas son suficientes; no se puede afirmar con rigor que 96 sea insuficiente.**

| Modelo | Neuronas | Dropout | Accuracy | F1 | Params | Comparación limpia |
|---|---|---|---|---|---|---|
| F1 | 96 | 0.0 | 0.776 ± 0.006 | 0.588 ± 0.004 | 2.785 | ✅ Sí (sin dropout) |
| F2 | 192 | 0.0 | 0.787 ± 0.009 | **0.603 ± 0.007** | 9.665 | ✅ Sí (sin dropout) |
| C | 224 | 0.0 | **0.792 ± 0.012** | **0.603 ± 0.019** | 12.929 | ✅ Sí (sin dropout) |
| F3 | 384 | **0.3** | 0.773 ± 0.002 | 0.590 ± 0.001 | 35.713 | ❌ **No** (también cambia dropout) |

**Comparación según criterio ±1 std**:

- **F1 vs C**: F1 = [0.770, 0.782], C = [0.780, 0.804]. **Se solapan ligeramente** en [0.780, 0.782]. La diferencia de 0.016 en accuracy es **marginal y no concluyente** con 3 semillas.
- **F2 vs C**: F2 = [0.778, 0.796], C = [0.780, 0.804]. **Se solapan ampliamente** → **diferencia no concluyente**. F2 y C son equivalentes.
- **C vs F3**: C = [0.780, 0.804], F3 = [0.771, 0.774]. **NO se solapan**. F3 es **claramente peor** que C según el criterio ±1 std. Pero como F3 también lleva dropout, no se puede atribuir exclusivamente a la anchura.

**Conclusión**: **F2 y C empatan en F1 puntual (0.603) y son equivalentes**. **F1 muestra peor accuracy puntual (0.776 vs 0.792), pero la diferencia es marginal y no concluyente** con 3 semillas. Los datos **sugieren** que 192-224 neuronas son suficientes para el mejor F1 observado; **no se puede afirmar con rigor que 96 sea insuficiente** según el criterio ±1 std. F3 no es comparable directamente con C.

---

### 5.3 Efecto del batch size (B vs G)

**Pregunta**: ¿Un batch más grande mejora el rendimiento?

**Respuesta**: **No se observa un efecto relevante del batch en el rendimiento final.**

| Modelo | Batch | Accuracy | F1 | Épocas |
|---|---|---|---|---|
| B | 32 | 0.782 ± 0.014 | 0.595 ± 0.012 | 21 |
| G | 64 | 0.788 ± 0.011 | 0.599 ± 0.015 | 31 |

**Interpretación técnica**:

- Con batch=32, hay 8000/32 = **250 actualizaciones por época**.
- Con batch=64, hay 8000/64 = **125 actualizaciones por época**.
- G necesita más épocas (31 vs 21) porque cada época tiene menos pasos de actualización.
- La diferencia en accuracy (0.006) está **dentro del ruido** (std ~0.01).

**Comparación según criterio ±1 std**: B = [0.768, 0.796], G = [0.777, 0.799]. **Se solapan ampliamente** → **diferencia no concluyente**.

**Conclusión**: **no se observa un efecto relevante del batch** en el rendimiento final. Con early stopping + weight decay, el efecto regularizador implícito del batch grande es redundante.

---

### 5.4 Efecto del learning rate (B vs H1 vs H2)

**Pregunta**: ¿Un learning rate más bajo mejora el rendimiento?

**Respuesta**: **No se observa un efecto relevante del LR en el rendimiento final, pero sí en el tiempo de entrenamiento.**

| Modelo | LR | Accuracy | F1 | Épocas | Tiempo relativo |
|---|---|---|---|---|---|
| B | 1e-3 | 0.782 ± 0.014 | 0.595 ± 0.012 | 21 | 1x |
| H1 | 5e-4 | 0.781 ± 0.005 | 0.590 ± 0.008 | 38 | 1.8x |
| H2 | 1e-4 | 0.785 ± 0.005 | 0.595 ± 0.008 | **108** | **5.1x** |

**Comparación según criterio ±1 std**: todos los rangos se solapan. **Diferencia no concluyente**.

**Conclusión**: **no se observa un efecto relevante del LR** en el rendimiento final. LR=1e-3 es el mejor compromiso de eficiencia: mismo resultado que LR más bajos, pero 5x menos tiempo.

---

### 5.5 Sobre el dropout (conclusión tentativa)

**Pregunta**: ¿El dropout mejora el rendimiento?

**Respuesta**: **Los modelos con dropout rinden peor, pero no hay una comparación limpia que aísle el dropout.**

| Modelo | Dropout | Arquitectura | F1 | Accuracy | Comparación con C |
|---|---|---|---|---|---|
| C | 0.0 | [128, 64, 32] | **0.603** | **0.792** | Referencia |
| E | 0.3 | [128, 64, 32, 16] | 0.595 | 0.781 | Cambia profundidad Y dropout |
| F3 | 0.3 | [128, 128, 128] | 0.590 | 0.773 | Cambia anchura Y dropout |

**Análisis**:

- **Entre C, E y F3**, C tiene el mejor F1 y accuracy. **Globalmente, F2 empata con C en F1 (0.603)**.
- **E (con dropout, 4 capas)** es peor. Pero E cambia **dos variables** respecto a C: profundidad (3→4) y dropout (0→0.3).
- **F3 (con dropout, 384 neuronas)** es el peor. Pero F3 cambia **dos variables** respecto a C: anchura (224→384) y dropout (0→0.3).

**Conclusión tentativa**: los modelos con dropout rinden peor, pero **no se puede atribuir exclusivamente al dropout**. Podría ser la profundidad (en E), la anchura (en F3), o el dropout. Para aislar el dropout haría falta un experimento `C con dropout=0.3` vs `C sin dropout`, que no existe en esta rejilla.

**Hipótesis (no confirmada)**: con early stopping + weight decay + pos_weight ya hay suficiente regularización. El dropout añade ruido sin aportar mejora. Pero esto es una hipótesis, no una conclusión. **Trabajo futuro**: añadir `C con dropout` para aislar el efecto.

---

## 6. Análisis de las curvas de pérdida individuales

Cada modelo tiene su `loss_modelo_X.png`. Estas curvas cuentan la historia del entrenamiento.

### 6.1 Modelo A (1 capa, 673 params, 31 épocas)

- **Train loss**: baja de ~0.99 a ~0.72 en 31 épocas.
- **Val loss**: baja de ~0.94 a ~0.75 en 31 épocas.
- **Forma**: train y val bajan juntos y se estabilizan.
- **Diagnóstico**: **sin overfitting**. El modelo es demasiado simple para memorizar.

### 6.2 Modelo C (3 capas, 12.929 params, 13 épocas)

- **Train loss**: baja de ~0.94 a **~0.73** en 13 épocas (valor final real: 0.733 ± 0.006).
- **Val loss**: baja de ~0.90 a **~0.74** en 13 épocas.
- **Forma**: train y val bajan juntos muy rápido.
- **Diagnóstico**: **sin overfitting**. El early stopping paró antes de que el modelo empezara a memorizar. **Esto es clave**: en la versión anterior, C overfiteaba porque se entrenaba 50 épocas fijas. Ahora para en 13 y generaliza bien.

### 6.3 Modelo H2 (LR=1e-4, 108 épocas)

- **Train loss**: baja de ~1.09 a ~0.70 en 108 épocas (valor final real: 0.700 ± 0.011).
- **Val loss**: baja de ~1.07 a ~0.75 en 108 épocas (valor final real: 0.746 ± 0.012).
- **Forma**: descenso muy lento y gradual.
- **Diagnóstico**: **convergencia lenta por LR bajo**. Necesita 5x más épocas que B para llegar al mismo sitio.

### 6.4 Modelo F3 (384 neuronas + dropout, 24 épocas)

- **Train loss**: baja de ~0.96 a ~0.73 en 24 épocas (valor final real: 0.728 ± 0.014).
- **Val loss**: baja de ~0.91 a ~0.75 en 24 épocas (valor final real: 0.747 ± 0.004).
- **Forma**: descenso más ruidoso por el dropout.
- **Diagnóstico**: **el dropout añade ruido**. La val loss tiene más varianza entre épocas.

---

## 7. El efecto del pos_weight: la clave del cambio de comportamiento

### 7.1 Comparación antes/después

| Métrica | Antes (sin pos_weight) | Ahora (con pos_weight=3.9) | Cambio |
|---|---|---|---|
| Accuracy | ~0.87 | ~0.79 | ↓ 8 puntos |
| Precision | ~0.77 | ~0.47 | ↓ 30 puntos |
| Recall | ~0.53 | **~0.79** | ↑ **26 puntos** |
| F1 | ~0.63 | ~0.60 | ≈ igual |
| AUC | (no medido) | **~0.865** | — |

### 7.2 ¿Qué significa esto?

Con `pos_weight=3.9`, la loss de los falsos negativos se multiplica por 3.9. Eso hace que el modelo **prefiera predecir churn aunque se equivoque**.

**Efecto técnico preciso**: `pos_weight` **desplaza las probabilidades predichas de los positivos hacia arriba**. El efecto neto es **equivalente** a bajar el umbral de decisión de 0.5 a ~0.35, pero **no es exactamente lo mismo**: el umbral sigue siendo 0.5, lo que cambia es la distribución de probabilidades que produce el modelo.

**Efecto práctico**:
- **Recall sube**: detecta el ~79% de los churns reales (antes ~53%).
- **Precision baja**: de los que predice churn, solo el ~47% lo son (antes ~77%).
- **Accuracy baja**: acierta menos globalmente porque hay más falsas alarmas.
- **AUC se mantiene**: la capacidad de discriminación no cambia.

### 7.3 ¿Por qué el AUC es igual?

El AUC mide la probabilidad de que un churn real tenga más probabilidad predicha que un no-churn real. **Es independiente del umbral**. El pos_weight desplaza las probabilidades, pero **no cambia la capacidad del modelo para ordenar** las muestras.

**Analogía**: el pos_weight **desplaza el punto de corte efectivo** de la decisión, no la capacidad de ordenación del modelo. La "balanza" sigue ordenando igual de bien (AUC), pero el "punto de equilibrio" se inclina hacia el churn.

### 7.4 ¿Es mejor o peor?

**Depende del negocio**:

- **Si el coste de perder un cliente es alto** (típico en churn bancario): **mejor ahora**. Recall ~0.79 significa que detectas 4 de cada 5 clientes que se van.
- **Si el coste de molestar a un cliente con una promo innecesaria es alto**: peor ahora. Precision ~0.47 significa que de cada 2 clientes que marcas como churn, 1 no lo es.

**En churn bancario, el coste de perder un cliente es mucho mayor que el coste de ofrecerle una promo**. Por eso **priorizar recall es la decisión correcta**.

> **Nota**: el dataset `Customer-Churn-Records.csv` contiene columnas como `CreditScore`, `Geography`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember` y `EstimatedSalary`, que son específicas del dominio bancario. Por eso se usa "churn bancario" y no genéricamente "churn".

---

## 8. Estabilidad entre semillas: análisis de la desviación estándar

| Modelo | Accuracy std | F1 std | Estabilidad |
|---|---|---|---|
| F3 | **0.0015** | **0.0006** | **El más estable en ambas métricas** |
| H2 | 0.0050 | 0.0076 | Muy estable |
| H1 | 0.0051 | 0.0081 | Muy estable |
| F1 | 0.0059 | 0.0038 | Estable |
| F2 | 0.0093 | 0.0071 | Moderadamente estable |
| A | 0.0102 | 0.0078 | Moderadamente estable |
| G | 0.0105 | 0.0145 | Moderadamente estable |
| C | 0.0124 | **0.0190** | **El menos estable en F1** |
| B | 0.0137 | 0.0122 | Poco estable |
| E | **0.0171** | 0.0150 | **El menos estable en accuracy** |

> **Nota**: los valores de esta tabla son los **sin redondear**. En la tabla de la sección 4 aparecen redondeados a 3 decimales (por ejemplo, F3: 0.002 y 0.001). Los rankings de estabilidad son idénticos con ambos redondeos.

**Interpretación**:

- **E es el menos estable en accuracy** (std 0.0171). La combinación de 4 capas + dropout introduce más varianza en accuracy.
- **C es el menos estable en F1** (std 0.0190). Su F1 varía más entre semillas por la sensibilidad del F1 al equilibrio precision/recall.
- **F3 (384 neuronas + dropout)** es el más estable en ambas métricas (std 0.0015 en accuracy, 0.0006 en F1). **Pero no se puede atribuir solo al dropout**: F3 también cambia la anchura (384 vs 224), y E, que también lleva dropout, es el **menos estable en accuracy**. La relación dropout-varianza queda como **hipótesis**, no como conclusión. Para aislarla haría falta `C+dropout` vs `C`.

**Lección (tentativa)**: el dropout podría reducir la varianza entre semillas, pero los datos no permiten aislarlo de otras variables (anchura en F3, profundidad en E). La relación queda como hipótesis a confirmar en trabajo futuro.

---

## 9. El caso de F3: AUC alto pero accuracy baja

F3 tiene el **mejor AUC (0.869)** pero el **peor accuracy (0.773)**. Esto no es contradictorio.

### 9.1 ¿Por qué ocurre?

- **AUC alto**: F3 **ordena muy bien las muestras**. Si tomas un churn real y un no-churn real, F3 asigna más probabilidad al churn el 86.9% de las veces. **Buena discriminación**, pero **no necesariamente buena calibración**. El AUC no mide calibración: un modelo puede tener AUC alto y probabilidades mal calibradas.
- **Accuracy bajo**: el umbral de decisión 0.5 no es el óptimo para F3. Con dropout, las probabilidades están más "comprimidas" hacia el centro. **Posiblemente** muchos churns reales tienen probabilidad ~0.45 (por debajo del umbral 0.5) y se clasifican como no-churn. **Esto es una hipótesis plausible, no una medición directa**.

### 9.2 ¿Qué pasaría si bajáramos el umbral?

Si bajáramos el umbral a 0.35:
- **Recall subiría** (más churns detectados).
- **Precision bajaría** (más falsas alarmas).
- **El efecto sobre accuracy es incierto** y probablemente negativo si el umbral añade muchos falsos positivos (con precision ya baja, añadir más FP puede empeorar el accuracy global). Habría que medirlo con threshold tuning explícito.

### 9.3 Lección

**El accuracy depende del umbral; el AUC no**. Para comparar modelos, el AUC es más robusto que el accuracy. F3 tiene una excelente capacidad de discriminación (AUC alto), pero su umbral óptimo no es 0.5. **AUC y calibración son cosas distintas**: el AUC mide ordenación, no calibración.

---

## 10. Hallazgos inesperados

### 10.1 El batch no muestra efecto relevante

B (batch 32) y G (batch 64) obtuvieron resultados equivalentes. **Lección**: en este dataset, no se observa un efecto relevante del batch en el rendimiento final.

### 10.2 El LR no muestra efecto relevante en el resultado, solo en el tiempo

B (LR 1e-3), H1 (LR 5e-4) y H2 (LR 1e-4) obtuvieron resultados equivalentes. Pero H2 necesitó **5x más épocas**. **Lección**: el LR afecta a la eficiencia, no al resultado final.

### 10.3 La anchura: 192-224 neuronas parecen suficientes

F2 (192) y C (224) empatan en F1 puntual (0.603). **F1 (96) queda algo por debajo (0.776 vs 0.792 en accuracy), pero la diferencia no es concluyente** según el criterio ±1 std. Los datos **sugieren** que 192-224 neuronas son suficientes; no se puede afirmar con rigor que 96 sea insuficiente. **Lección**: en este dataset, 192-224 neuronas parecen suficientes para el mejor F1 observado.

### 10.4 El overfitting de C era por épocas, no por profundidad

La conclusión anterior "3 capas → overfitting severo" era **falsa**. El overfitting era por **50 épocas fijas sin regularización**. Con early stopping, C converge en 13 épocas y generaliza bien. **Lección**: no atribuir el overfitting a la arquitectura sin aislar las variables.

### 10.5 El dropout no muestra mejora, pero la comparación no es limpia

Los modelos con dropout (E, F3) rinden peor, pero también cambian otras variables. **Lección**: no concluir sobre el dropout sin una comparación que lo aísle.

---

## 11. Comparación antes/después de la corrección

| Aspecto | Antes | Ahora |
|---|---|---|
| Experimentos | 4 (3 arquitecturas + 1 variante) | 10 (7 arquitecturas + 3 variantes) |
| Semillas | 1 | 3 |
| Early stopping | No | Sí (patience=8) |
| Dropout | No usado | Activo en E y F3 |
| pos_weight | No | 3.9 |
| AUC | No medido | 0.86-0.87 |
| Tabla de pérdidas finales | No | Sí (train/val/test) |
| Conclusión sobre C | "Overfitting por profundidad" | "Overfitting por 50 épocas fijas" |
| Significancia estadística | No | Media ± std con 3 semillas (criterio exploratorio) |
| Gráfico con barras de error | No | Sí |

**Esto demuestra que se han corregido los errores metodológicos señalados por el profesor.**

---

## 12. Lecciones aprendidas

1. **El early stopping es más importante que la arquitectura**. Sin él, cualquier red grande overfitea. Con él, incluso C (12.929 parámetros) generaliza bien.
2. **El pos_weight cambia el trade-off precision/recall sin cambiar la capacidad de discriminación (AUC constante)**. Es una decisión de negocio, no de estadística.
3. **La ablación es obligatoria**. Sin ella, no puedes atribuir causalidad. La versión anterior mezclaba batch y LR en el modelo D; ahora cada experimento cambia una variable.
4. **3 semillas permiten reportar media ± std y evitar conclusiones basadas en una única inicialización**, aunque no permiten tests formales con potencia suficiente. Para eso harían falta más semillas (≥10) o validación cruzada.
5. **En este dataset, 192-224 neuronas parecen suficientes para el mejor F1 observado**. No hay evidencia de que anchuras mayores aporten, pero **F3 confunde anchura y dropout**, así que la conclusión sobre el rango alto es tentativa.
6. **El LR afecta al tiempo, no al resultado**. LR=1e-3 converge en 21 épocas; LR=1e-4 necesita 108. Mismo resultado, 5x más tiempo.
7. **El AUC es más robusto que el accuracy para comparar modelos**. F3 tiene el mejor AUC pero el peor accuracy por el umbral.

---

## 13. Limitaciones

- **Una sola partición train/val/test**. No se hizo validación cruzada k-fold. Con 3 semillas se mitiga parcialmente, pero una CV daría estimaciones más robustas.
- **Dataset de 10.000 ejemplos**. Es pequeño para redes grandes. Los resultados podrían cambiar con más datos.
- **Desbalanceo mitigado con pos_weight pero no se exploró threshold tuning explícito**. Bajar el umbral a 0.35 o 0.4 podría mejorar el recall en todos los modelos, pero el efecto sobre accuracy y F1 requiere medición explícita.
- **No hay comparación limpia de dropout**. E y F3 cambian dropout + otra variable.
- **F3 no es comparable directamente con C** por el dropout.
- **No se probó BatchNorm**. Podría estabilizar el entrenamiento de las redes más profundas.
- **No se probaron otras funciones de activación** (tanh, LeakyReLU, GELU).
- **Criterio de comparación exploratorio**. Se usa solapamiento ±1 std, no tests estadísticos formales. Con 3 semillas no hay potencia suficiente para tests formales.

---

## 14. Trabajo futuro

1. **Validación cruzada 5-fold** para estimaciones más robustas.
2. **Threshold tuning explícito**: probar umbrales 0.3, 0.35, 0.4, 0.45, 0.5 y medir el trade-off precision/recall/accuracy/F1.
3. **Añadir `C+dropout`** para aislar el efecto del dropout de la profundidad y la anchura.
4. **Añadir `F3 sin dropout`** para aislar la anchura en el rango alto.
5. **Probar BatchNorm** en las redes profundas (C, E, F3).
6. **Probar otras activaciones** (LeakyReLU, GELU).
7. **Probar oversampling** de la clase minoritaria en lugar de pos_weight.

---

## 15. Conclusiones finales y recomendación

### 15.1 Conclusiones técnicas

1. **C y F2 comparten el mejor F1 puntual (0.603)**, aunque C tiene mejor accuracy puntual (0.792) y F2 es más eficiente (25% menos parámetros: 9.665 vs 12.929). Las diferencias con B, E, G y H2 están dentro del margen ±1 std. **En la práctica, todas las configuraciones con 2+ capas sin dropout (B, C, F1, F2, G, H1, H2) y E son equivalentes según el criterio ±1 std. F3 es claramente peor que C según este criterio, pero confunde anchura y dropout, así que no se puede atribuir solo a la arquitectura.**

2. **Si el criterio es eficiencia con rendimiento equivalente, F2 es preferible a C**: mismo F1 puntual con un 25% menos de parámetros.

3. **F3 tiene el mejor recall (0.802) y AUC (0.869)**, pero el peor accuracy (0.773) por el umbral. En un contexto de negocio donde el coste de perder un cliente es alto, es el candidato más alineado, **pero requiere threshold tuning antes de producción**.

4. **La arquitectura no es la palanca principal**. Las diferencias entre las configuraciones con 2+ capas son pequeñas (~0.02 puntos en el rango observado), aunque F3 queda por debajo de C fuera del margen ±1 std. **Lo que sí importa** es:
   - **Early stopping** (corrige overfitting).
   - **pos_weight** (ajusta el trade-off precision/recall hacia el negocio).
   - **Weight decay** (regulariza).

5. **El overfitting anterior de C no era por profundidad**: era por 50 épocas fijas sin regularización. Con early stopping, C converge en 13 épocas y generaliza correctamente.

### 15.2 Recomendación final explícita

**Dado que en churn bancario el coste de un falso negativo (cliente que se va y no detectas) supera al de un falso positivo (cliente que se queda y le ofreces una promo), la recomendación de negocio es F3.** Detecta el 80.2% de los churns reales, con un AUC de 0.869, aceptando el coste de una precision más baja (0.467).

**Importante**: F3 **no debería desplegarse con umbral 0.5 sin threshold tuning**. Su AUC alto con accuracy bajo sugiere que su umbral óptimo es más bajo (~0.35). Antes de producción habría que:
1. Hacer threshold tuning explícito (0.3, 0.35, 0.4, 0.45, 0.5).
2. Compararlo con C y F2 ajustando también sus umbrales.
3. Medir el coste de negocio de cada configuración (coste de falso positivo vs falso negativo).

**Si el criterio fuera únicamente F1 puntual (equilibrio precision/recall), la recomendación sería F2**: mismo F1 que C con 25% menos parámetros, más eficiente en producción.

**Si el criterio fuera accuracy puntual (no aplicable en churn desbalanceado), la recomendación sería C**: mejor accuracy (0.792).

**Recomendación operativa**:
- **F3 para producción, previo threshold tuning** (prioriza recall, alineado con el coste asimétrico del churn).
- **F2 como alternativa eficiente** (mismo F1 con menos parámetros).
- **C como referencia técnica** (mejor accuracy puntual).

---

## 16. Archivos generados

- `resultados/tables/comparacion_modelos.csv` → tabla agregada (10 filas, 25 columnas, media ± std).
- `resultados/tables/resultados_por_semilla.csv` → 30 filas crudas (10 experimentos × 3 semillas).
- `resultados/tables/resultados_por_semilla.json` → versión JSON.
- `resultados/figures/loss_modelo_*.png` → 10 curvas de pérdida.
- `resultados/figures/confusion_matrix_*.png` → 10 matrices de confusión.
- `resultados/figures/comparacion_modelos.png` → gráfico comparativo con barras de error (5 métricas).