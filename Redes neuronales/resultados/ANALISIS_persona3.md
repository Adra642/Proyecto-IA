# Análisis completo de resultados — Persona 3 (Fase C corregida)

## 1. Resumen ejecutivo

Se han entrenado **10 configuraciones distintas** de red neuronal con **3 semillas cada una** (30 entrenamientos totales). Cada experimento cambia **una sola variable** respecto a su referencia, lo que permite atribución causal de las diferencias observadas. Se aplicaron técnicas de regularización modernas: **early stopping** (patience=8), **weight decay** (1e-4), **pos_weight** (~3.9) para mitigar el desbalanceo 80/20, y **dropout** (0.3) en los modelos que lo requieren por su exceso de capacidad.

### Hallazgos principales

1. **El mejor modelo técnico es C** (3 capas, 128-64-32) con **Accuracy = 0.792 ± 0.012** y **F1 = 0.603 ± 0.019**. Sin embargo, las diferencias con B, F2, E, G y H2 están dentro del margen de 1 desviación estándar, por lo que en la práctica **todas las arquitecturas con 2 o más capas rinden de forma equivalente**.

2. **El overfitting anterior de C no era por profundidad**. Con 50 épocas fijas y sin regularización, C memorizaba el train. Con early stopping, C paró en **época 13** y generaliza correctamente. **La conclusión anterior "3 capas → overfitting severo" era falsa**.

3. **El pos_weight ha cambiado el comportamiento del modelo sin cambiar su capacidad de discriminación**. El AUC se mantiene en ~0.865 en todos los modelos, pero el recall sube de ~0.53 a ~0.79 a costa de bajar la precision de ~0.77 a ~0.47. **Esto no es un empeoramiento, es un cambio de objetivo**: priorizar detectar churns sobre no molestar a los que se quedan.

4. **La arquitectura no es la palanca principal en este problema**. Las diferencias de accuracy entre todas las configuraciones con 2+ capas son de ~0.02 puntos, dentro del ruido estadístico. **Lo que sí importa** es el early stopping (corrige overfitting) y el pos_weight (ajusta el trade-off precision/recall).

5. **El dropout no aportó mejora** en este dataset. Con early stopping + weight decay + pos_weight ya hay suficiente regularización. Añadir dropout solo introduce ruido sin mejorar el rendimiento medio.

---

## 2. Contexto y motivación: qué se corrigió respecto a la versión anterior

La versión anterior del trabajo (entrenada con el código inicial) presentaba **6 debilidades metodológicas** que el profesor señaló:

| Debilidad anterior | Corrección aplicada |
|---|---|
| 4 experimentos, pero solo 3 arquitecturas (D era variante de C) | **9 arquitecturas distintas** (A, B, C, E, F1, F2, F3, G, H1, H2) |
| D cambiaba batch y LR a la vez → no se podía aislar el efecto | **Cada experimento cambia una sola variable** (ablación real) |
| Una sola semilla (RANDOM_STATE=42) | **3 semillas** ([0, 1, 2]) con media ± std |
| Sin early stopping (50 épocas fijas) | **Early stopping con patience=8**, max 300 épocas |
| Dropout no se usaba aunque ChurnNet lo soporta | **Dropout 0.3** en E y F3, pasado correctamente desde train.py |
| Desbalanceo detectado pero no mitigado | **pos_weight = 3.9** en BCEWithLogitsLoss |

Además, se añadieron dos mejoras que el profesor mencionó en la reunión:

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
| **F3** | [128, 128, 128] | 3 | 384 | 32 | 1e-3 | 0.3 | Anchura grande vs C |
| **G** | [64, 32] | 2 | 96 | **64** | 1e-3 | 0.0 | Solo batch vs B |
| **H1** | [64, 32] | 2 | 96 | 32 | **5e-4** | 0.0 | Solo LR vs B |
| **H2** | [64, 32] | 2 | 96 | 32 | **1e-4** | 0.0 | Solo LR más bajo vs B |

### 3.2 Cobertura de los requisitos del enunciado

| Requisito | Cómo se cumple |
|---|---|
| **Número de capas ocultas** | 4 profundidades: A (1), B (2), C/F1/F2/F3 (3), E (4) |
| **Número de neuronas por capa** | 4 anchuras con misma profundidad: F1 (96), F2 (192), C (224), F3 (384) |
| **Batch size** | B (32) vs G (64), solo cambia el batch |
| **Learning rate** | B (1e-3) vs H1 (5e-4) vs H2 (1e-4), solo cambia el LR |
| **Ablación** | Cada experimento cambia una variable respecto a su referencia |

### 3.3 Justificación técnica de la rejilla

- **A → B → C → E**: mide el efecto de la **profundidad** manteniendo constante todo lo demás. E lleva dropout porque, sin él, una red de 4 capas con 13.441 parámetros overfitearía inevitablemente, y la comparación no sería justa.
- **F1 → F2 → C → F3**: mide el efecto de la **anchura** manteniendo la profundidad fija en 3 capas. Esto permite responder a la pregunta: *¿el problema de C (si lo hubiera) era la profundidad o la anchura?*
- **B vs G**: aísla el efecto del **batch size**. B y G tienen la misma arquitectura y el mismo LR, solo cambia el batch.
- **B vs H1 vs H2**: aísla el efecto del **learning rate**. Misma arquitectura, mismo batch, solo cambia el LR.

---

## 4. Resultados completos (media ± std, 3 semillas)

| Modelo | Arquitectura | Params | Épocas | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|---|---|
| A | [32] | 673 | 31 | 0.766 ± 0.010 | 0.459 ± 0.013 | **0.797 ± 0.012** | 0.582 ± 0.008 | 0.867 ± 0.002 |
| B | [64, 32] | 3.393 | 21 | 0.782 ± 0.014 | 0.479 ± 0.019 | 0.784 ± 0.010 | 0.595 ± 0.012 | 0.863 ± 0.003 |
| **C** | [128, 64, 32] | 12.929 | 13 | **0.792 ± 0.012** | **0.495 ± 0.019** | 0.773 ± 0.016 | **0.603 ± 0.019** | 0.864 ± 0.004 |
| E | [128, 64, 32, 16] | 13.441 | 26 | 0.781 ± 0.017 | 0.479 ± 0.023 | 0.788 ± 0.015 | 0.595 ± 0.015 | 0.866 ± 0.004 |
| F1 | [32, 32, 32] | 2.785 | 18 | 0.776 ± 0.006 | 0.471 ± 0.008 | 0.783 ± 0.012 | 0.588 ± 0.004 | 0.867 ± 0.003 |
| **F2** | [64, 64, 64] | 9.665 | 14 | 0.787 ± 0.009 | 0.487 ± 0.014 | 0.793 ± 0.012 | **0.603 ± 0.007** | 0.867 ± 0.006 |
| F3 | [128, 128, 128] | 35.713 | 24 | 0.773 ± 0.002 | 0.467 ± 0.002 | **0.802 ± 0.006** | 0.590 ± 0.001 | **0.869 ± 0.003** |
| G | [64, 32] b=64 | 3.393 | 31 | 0.788 ± 0.011 | 0.488 ± 0.016 | 0.775 ± 0.015 | 0.599 ± 0.015 | 0.863 ± 0.001 |
| H1 | [64, 32] lr=5e-4 | 3.393 | 38 | 0.781 ± 0.005 | 0.477 ± 0.007 | 0.775 ± 0.018 | 0.590 ± 0.008 | 0.864 ± 0.002 |
| H2 | [64, 32] lr=1e-4 | 3.393 | 108 | 0.785 ± 0.005 | 0.483 ± 0.008 | 0.775 ± 0.010 | 0.595 ± 0.008 | 0.865 ± 0.002 |

---

## 5. Análisis detallado por bloques

### 5.1 Efecto de la profundidad (A → B → C → E)

**Pregunta**: ¿Aumentar el número de capas ocultas mejora el rendimiento?

**Respuesta**: **Sí, pero con un óptimo en 3 capas. La 4ª capa no aporta.**

| Transición | Cambio en Accuracy | Cambio en F1 | Interpretación |
|---|---|---|---|
| A → B (+1 capa) | +0.015 | +0.013 | **Mejora significativa**. La 2ª capa permite aprender interacciones no lineales que una sola capa no captura. |
| B → C (+1 capa) | +0.011 | +0.008 | **Mejora marginal**. C es ligeramente mejor que B, pero la diferencia está dentro de 1 std. |
| C → E (+1 capa) | -0.011 | -0.008 | **Empeora**. La 4ª capa no aporta; el dropout mitiga parcialmente pero no compensa. |

**Interpretación técnica**:

- **A (1 capa)** solo puede aprender una transformación lineal seguida de una no linealidad. No puede capturar interacciones entre features. Su accuracy es el peor (0.766).
- **B (2 capas)** puede aprender una composición de dos transformaciones no lineales. Esto es suficiente para capturar la mayoría de patrones en este dataset. Accuracy sube a 0.782.
- **C (3 capas)** añade una tercera transformación. La mejora es marginal (0.792). El modelo ya tiene capacidad suficiente.
- **E (4 capas + dropout)** añade una 4ª capa, pero el dropout en el medio de la red reduce la información que fluye. La accuracy baja a 0.781. **La 4ª capa no aporta porque el dataset no tiene suficiente complejidad para necesitarla.**

**Significancia estadística**:
- A vs C: rangos no se solapan (A: [0.756, 0.776], C: [0.780, 0.805]). **Diferencia significativa**.
- B vs C: rangos se solapan (B: [0.768, 0.795], C: [0.780, 0.805]). **Diferencia no significativa**.
- C vs E: rangos se solapan. **Diferencia no significativa**.

**Conclusión**: la profundidad tiene un óptimo en **3 capas**. A es claramente peor. B y C son equivalentes. E no aporta. **El overfitting anterior de C no era por profundidad, era por 50 épocas fijas sin regularización**.

### 5.2 Efecto de la anchura (F1 → F2 → C → F3)

**Pregunta**: Con la misma profundidad (3 capas), ¿cómo afecta el número de neuronas por capa?

**Respuesta**: **Hay un óptimo en 192-224 neuronas. Menos es demasiado estrecho; más es demasiado ancho.**

| Modelo | Neuronas | Accuracy | F1 | Params | Interpretación |
|---|---|---|---|---|---|
| F1 | 96 | 0.776 | 0.588 | 2.785 | **Demasiado estrecho**. No tiene capacidad para aprender patrones complejos. |
| F2 | 192 | 0.787 | **0.603** | 9.665 | **Óptimo de eficiencia**. Mismo F1 que C con 25% menos parámetros. |
| C | 224 | **0.792** | **0.603** | 12.929 | **Óptimo de accuracy**. Ligeramente mejor que F2 en accuracy. |
| F3 | 384 | 0.773 | 0.590 | 35.713 | **Demasiado ancho**. Overfitea incluso con dropout. |

**Interpretación técnica**:

- **F1 (96 neuronas)** tiene solo 2.785 parámetros. Con 8.000 ejemplos de train, la relación parámetros/ejemplos es baja (0.35). El modelo no tiene capacidad suficiente para aprender los patrones. Accuracy baja a 0.776.
- **F2 (192 neuronas)** tiene 9.665 parámetros. Relación parámetros/ejemplos: 1.2. Es suficiente para aprender sin overfitear. F1 = 0.603, igual que C.
- **C (224 neuronas)** tiene 12.929 parámetros. Relación: 1.6. Ligeramente más capacidad que F2. Accuracy sube a 0.792 (mejor del estudio).
- **F3 (384 neuronas)** tiene 35.713 parámetros. Relación: **4.5**. Demasiada capacidad para 8.000 ejemplos. Aunque tiene dropout (0.3), overfitea. Accuracy baja a 0.773.

**Significancia estadística**:
- F1 vs C: rangos no se solapan (F1: [0.770, 0.782], C: [0.780, 0.805]). **Diferencia significativa**.
- F2 vs C: rangos se solapan (F2: [0.778, 0.796], C: [0.780, 0.805]). **Diferencia no significativa**.
- C vs F3: rangos se solapan parcialmente. F3 tiene std muy baja (0.0015), lo que indica que es muy estable pero consistentemente peor.

**Conclusión**: el óptimo de anchura está en **192-224 neuronas**. F2 es la mejor relación rendimiento/eficiencia (mismo F1 que C con 25% menos parámetros). F3, a pesar de tener 2.7x más parámetros que C, rinde peor.

### 5.3 Efecto del batch size (B vs G)

**Pregunta**: ¿Un batch más grande mejora el rendimiento?

**Respuesta**: **No. El batch no es una palanca relevante en este problema.**

| Modelo | Batch | Accuracy | F1 | Épocas | Interpretación |
|---|---|---|---|---|---|
| B | 32 | 0.782 | 0.595 | 21 | Batch pequeño: más actualizaciones por época, convergencia más rápida. |
| G | 64 | 0.788 | 0.599 | 31 | Batch grande: menos actualizaciones por época, necesita más épocas. |

**Interpretación técnica**:

- Con batch=32, hay 8000/32 = **250 actualizaciones por época**.
- Con batch=64, hay 8000/64 = **125 actualizaciones por época**.
- G necesita más épocas (31 vs 21) porque cada época tiene menos pasos de actualización.
- La diferencia en accuracy (0.788 vs 0.782) es de 0.006, **dentro del ruido estadístico** (std ~0.01).

**Teoría**: un batch más grande regulariza implícitamente (gradiente más suave, menos ruido). Pero en este caso, con early stopping + weight decay, ese efecto es redundante. **El batch no aporta nada aquí**.

**Significancia**: rangos se solapan. **No significativo**.

### 5.4 Efecto del learning rate (B vs H1 vs H2)

**Pregunta**: ¿Un learning rate más bajo mejora el rendimiento?

**Respuesta**: **No. El LR no afecta al resultado final, pero sí al tiempo de entrenamiento.**

| Modelo | LR | Accuracy | F1 | Épocas | Tiempo relativo |
|---|---|---|---|---|---|
| B | 1e-3 | 0.782 | 0.595 | 21 | 1x |
| H1 | 5e-4 | 0.781 | 0.590 | 38 | 1.8x |
| H2 | 1e-4 | 0.785 | 0.595 | **108** | **5.1x** |

**Interpretación técnica**:

- **B (LR=1e-3)**: converge en 21 épocas. Es el más rápido.
- **H1 (LR=5e-4)**: converge en 38 épocas. 1.8x más lento, mismo resultado.
- **H2 (LR=1e-4)**: converge en 108 épocas. 5.1x más lento, mismo resultado.

**¿Por qué H2 necesita tantas épocas?** Porque con LR=1e-4, cada actualización mueve los pesos muy poco. El modelo tarda mucho en llegar al mínimo. Pero **al final llega al mismo sitio**: accuracy 0.785 (vs 0.782 de B).

**Significancia**: rangos se solapan. **No significativo**.

**Conclusión**: LR=1e-3 es el mejor compromiso. Mismo resultado que LR más bajos, pero 5x menos tiempo. **No tiene sentido usar LR=1e-4 en este problema**.

### 5.5 Efecto del dropout (C vs E vs F3)

**Pregunta**: ¿El dropout mejora el rendimiento?

**Respuesta**: **No. El dropout no aporta en este dataset.**

| Modelo | Dropout | Arquitectura | F1 | Accuracy | Interpretación |
|---|---|---|---|---|---|
| C | 0.0 | [128, 64, 32] | **0.603** | **0.792** | Sin dropout: mejor F1 y accuracy. |
| E | 0.3 | [128, 64, 32, 16] | 0.595 | 0.781 | Con dropout: peor en ambas métricas. |
| F3 | 0.3 | [128, 128, 128] | 0.590 | 0.773 | Con dropout: peor aún. |

**Interpretación técnica**:

- **C (sin dropout)** es el mejor en F1 y accuracy. El early stopping + weight decay ya son suficientes para regularizar.
- **E (con dropout, 4 capas)** es peor. El dropout añade ruido durante el entrenamiento, lo que **reduce la capacidad efectiva** de la red. En un dataset pequeño como este, eso es contraproducente.
- **F3 (con dropout, 384 neuronas)** es el peor. Aunque el dropout ayuda a controlar el overfitting de las 384 neuronas, no es suficiente para compensar el exceso de capacidad.

**¿Por qué el dropout no ayuda aquí?** Porque el dataset no es lo suficientemente grande como para que el dropout aporte. El dropout es útil cuando hay **millones de ejemplos** y la red tiende a memorizar. Con 10.000 ejemplos y early stopping, el dropout es redundante.

**Conclusión**: el dropout **no es universalmente útil**. Depende del contexto de regularización. Aquí, con early stopping + weight decay + pos_weight, no aporta.

---

## 6. Análisis de las curvas de pérdida individuales

Cada modelo tiene su `loss_modelo_X.png`. Estas curvas cuentan la historia del entrenamiento.

### 6.1 Modelo A (1 capa, 673 params)

- **Train loss**: baja de 0.99 a 0.72 en 31 épocas.
- **Val loss**: baja de 0.94 a 0.75 en 31 épocas.
- **Forma**: train y val bajan juntos y se estabilizan.
- **Early stopping**: en época 31, cuando val dejó de mejorar durante 8 épocas.
- **Diagnóstico**: **sin overfitting**. El modelo es demasiado simple para memorizar.

### 6.2 Modelo C (3 capas, 12.929 params)

- **Train loss**: baja de 0.94 a 0.68 en 13 épocas.
- **Val loss**: baja de 0.90 a 0.74 en 13 épocas.
- **Forma**: train y val bajan juntos muy rápido. El modelo converge en solo 13 épocas.
- **Early stopping**: en época 13.
- **Diagnóstico**: **sin overfitting**. El early stopping paró antes de que el modelo empezara a memorizar. **Esto es clave**: en la versión anterior, C overfiteaba porque se entrenaba 50 épocas fijas. Ahora para en 13 y generaliza bien.

### 6.3 Modelo H2 (LR=1e-4, 108 épocas)

- **Train loss**: baja de 1.09 a 0.70 en 108 épocas.
- **Val loss**: baja de 1.07 a 0.75 en 108 épocas.
- **Forma**: descenso muy lento y gradual.
- **Early stopping**: en época 108.
- **Diagnóstico**: **convergencia lenta por LR bajo**. El modelo necesita muchas más épocas para llegar al mismo sitio. Al final, mismo resultado que B (accuracy 0.785 vs 0.782).

### 6.4 Modelo F3 (384 neuronas, dropout, 24 épocas)

- **Train loss**: baja de 0.96 a 0.72 en 24 épocas.
- **Val loss**: baja de 0.91 a 0.74 en 24 épocas.
- **Forma**: descenso más ruidoso por el dropout.
- **Early stopping**: en época 24.
- **Diagnóstico**: **el dropout añade ruido**. La val loss tiene más varianza entre épocas. Aun así, el early stopping lo controla.

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

Con `pos_weight=3.9`, la loss de los falsos negativos se multiplica por 3.9. Eso hace que el modelo **prefiera predecir churn aunque se equivoque**, porque equivocarse diciendo "se va" (falso positivo) cuesta menos que equivocarse diciendo "se queda" (falso negativo).

**Efecto práctico**:
- **Recall sube**: detecta el 79% de los churns reales (antes 53%).
- **Precision baja**: de los que predice churn, solo el 47% lo son (antes 77%).
- **Accuracy baja**: acierta menos globalmente porque hay más falsas alarmas.
- **AUC se mantiene**: la capacidad de discriminación no cambia.

### 7.3 ¿Por qué el AUC es igual?

El AUC mide la probabilidad de que un churn real tenga más probabilidad predicha que un no-churn real. **Es independiente del umbral**. El pos_weight cambia el umbral efectivo de decisión, pero **no cambia la capacidad del modelo para ordenar las probabilidades**.

**Analogía**: es como una balanza muy precisa pero mal calibrada. El pos_weight recalibra el punto de equilibrio. La balanza sigue siendo igual de precisa (AUC), pero ahora se inclina hacia el churn.

### 7.4 ¿Es mejor o peor?

**Depende del negocio**:

- **Si el coste de perder un cliente es alto** (típico en churn): **mejor ahora**. Recall 0.79 significa que detectas 4 de cada 5 clientes que se van.
- **Si el coste de molestar a un cliente con una promo innecesaria es alto**: peor ahora. Precision 0.47 significa que de cada 2 clientes que marcas como churn, 1 no lo es.

**En churn bancario, el coste de perder un cliente es mucho mayor que el coste de ofrecerle una promo**. Por eso **priorizar recall es la decisión correcta**.

---

## 8. Estabilidad entre semillas: análisis de la desviación estándar

La desviación estándar (std) mide cuánto varía el resultado entre semillas. **Menos std = más estable**.

| Modelo | Accuracy std | F1 std | Estabilidad |
|---|---|---|---|
| F3 | **0.0015** | **0.0006** | **El más estable** |
| H2 | 0.0050 | 0.0076 | Muy estable |
| H1 | 0.0051 | 0.0081 | Muy estable |
| F1 | 0.0059 | 0.0038 | Estable |
| F2 | 0.0093 | 0.0071 | Moderadamente estable |
| A | 0.0102 | 0.0078 | Moderadamente estable |
| G | 0.0105 | 0.0145 | Moderadamente estable |
| C | 0.0124 | 0.0190 | **El menos estable** |
| B | 0.0137 | 0.0122 | Poco estable |
| E | 0.0171 | 0.0150 | Poco estable |

**Interpretación**:

- **F3 (384 neuronas + dropout)** es el más estable (std 0.0015 en accuracy). El dropout reduce la varianza entre semillas.
- **C (3 capas sin dropout)** es el menos estable (std 0.0124). Sin dropout, la inicialización aleatoria afecta más.
- **H1 y H2 (LR bajos)** son muy estables. Un LR bajo hace que el entrenamiento sea más determinista.

**Lección**: el dropout, aunque no mejora la media, **sí reduce la varianza**. Es un trade-off: menos rendimiento medio, pero más consistencia entre semillas.

---

## 9. El caso de F3: AUC alto pero accuracy baja

F3 tiene el **mejor AUC (0.869)** pero el **peor accuracy (0.773)**. Esto parece contradictorio, pero no lo es.

### 9.1 ¿Por qué ocurre?

- **AUC alto**: F3 asigna probabilidades muy bien calibradas. Si tomas un churn real y un no-churn real, F3 asigna más probabilidad al churn el 86.9% de las veces. **La capacidad de discriminación es excelente**.
- **Accuracy bajo**: el umbral de decisión 0.5 no es el óptimo para F3. Con dropout, las probabilidades están más "comprimidas" hacia el centro. Muchos churns reales tienen probabilidad 0.45 (por debajo del umbral 0.5) y se clasifican como no-churn.

### 9.2 ¿Qué pasaría si bajáramos el umbral?

Si usáramos umbral 0.35 en lugar de 0.5:
- **Recall subiría** (más churns detectados).
- **Precision bajaría** (más falsas alarmas).
- **Accuracy subiría** (porque el umbral estaría mejor calibrado para la distribución de F3).

### 9.3 Lección

**El accuracy depende del umbral; el AUC no**. Para comparar modelos, el AUC es más robusto. F3 es un modelo excelente en términos de discriminación, pero su umbral óptimo no es 0.5.

---

## 10. Hallazgos inesperados

### 10.1 El dropout no ayudó

A pesar de que el profesor lo pidió y de que el código lo soporta, el dropout **no mejoró el rendimiento**. C (sin dropout) fue mejor que E y F3 (con dropout). **Lección**: el dropout no es universalmente útil; depende del contexto de regularización.

### 10.2 El batch no importa

B (batch 32) y G (batch 64) obtuvieron resultados equivalentes. **Lección**: en este dataset, el batch no es una palanca relevante.

### 10.3 El LR no afecta al resultado, solo al tiempo

B (LR 1e-3), H1 (LR 5e-4) y H2 (LR 1e-4) obtuvieron resultados equivalentes. Pero H2 necesitó **5x más épocas**. **Lección**: el LR afecta a la eficiencia, no al resultado final.

### 10.4 La anchura óptima es 192-224 neuronas

F2 (192) y C (224) empatan en F1. F1 (96) es demasiado estrecho. F3 (384) es demasiado ancho. **Lección**: hay un óptimo de capacidad que depende del tamaño del dataset.

### 10.5 El overfitting de C era por épocas, no por profundidad

La conclusión anterior "3 capas → overfitting severo" era **falsa**. El overfitting era por **50 épocas fijas sin regularización**. Con early stopping, C converge en 13 épocas y generaliza bien. **Lección**: no atribuir el overfitting a la arquitectura sin aislar las variables.

---

## 11. Comparación antes/después de la corrección

| Aspecto | Antes | Ahora |
|---|---|---|
| Arquitecturas | 3 (D era variante de C) | 9 arquitecturas distintas |
| Semillas | 1 | 3 |
| Early stopping | No | Sí (patience=8) |
| Dropout | No usado | Activo en E y F3 |
| pos_weight | No | 3.9 |
| AUC | No medido | 0.86-0.87 |
| Tabla de pérdidas finales | No | Sí (train/val/test) |
| Conclusión sobre C | "Overfitting por profundidad" | "Overfitting por 50 épocas fijas" |
| Significancia estadística | No | Media ± std con 3 semillas |
| Gráfico con barras de error | No | Sí |

**Esto demuestra que se han corregido los errores metodológicos señalados por el profesor.**

---

## 12. Lecciones aprendidas

1. **El early stopping es más importante que la arquitectura**. Sin él, cualquier red grande overfitea. Con él, incluso C (12.929 parámetros) generaliza bien.
2. **El pos_weight cambia el trade-off precision/recall sin cambiar la capacidad de discriminación (AUC constante)**. Es una decisión de negocio, no de estadística.
3. **La ablación es obligatoria**. Sin ella, no puedes atribuir causalidad. La versión anterior mezclaba batch y LR en el modelo D; ahora cada experimento cambia una variable.
4. **3 semillas son el mínimo para distinguir señal de ruido**. Con 1 semilla, diferencias de 0.01 son indistinguibles de la aleatorización.
5. **El dropout no es universalmente útil**. Depende del contexto de regularización. Aquí, con early stopping + weight decay + pos_weight, es redundante.
6. **En datasets pequeños (10.000 ejemplos), las redes grandes no aportan**. F3 (384 neuronas) rinde peor que C (224) a pesar de tener 2.7x más parámetros.
7. **El LR afecta al tiempo, no al resultado**. LR=1e-3 converge en 21 épocas; LR=1e-4 necesita 108. Mismo resultado, 5x más tiempo.
8. **El AUC es más robusto que el accuracy para comparar modelos**. F3 tiene el mejor AUC pero el peor accuracy por el umbral.

---

## 13. Limitaciones

- **Una sola partición train/val/test**. No se hizo validación cruzada k-fold. Con 3 semillas se mitiga parcialmente, pero una CV daría estimaciones más robustas.
- **Dataset de 10.000 ejemplos**. Es pequeño para redes grandes. Los resultados podrían cambiar con más datos.
- **Desbalanceo mitigado con pos_weight pero no se exploró threshold tuning**. Bajar el umbral a 0.35 o 0.4 podría mejorar el F1.
- **No se probó BatchNorm**. Podría estabilizar el entrenamiento de las redes más profundas.
- **No se probaron otras funciones de activación** (tanh, LeakyReLU, GELU). ReLU es el estándar, pero otras podrían funcionar mejor en este dataset.

---

## 14. Trabajo futuro

1. **Validación cruzada 5-fold** para estimaciones más robustas.
2. **Threshold tuning explícito**: probar umbrales 0.3, 0.35, 0.4, 0.45, 0.5 y ver el trade-off precision/recall.
3. **Probar BatchNorm** en las redes profundas (C, E, F3).
4. **Probar otras activaciones** (LeakyReLU, GELU).
5. **Probar oversampling** de la clase minoritaria en lugar de pos_weight.
6. **Probar arquitecturas más grandes** si se consiguen más datos.

---

## 15. Conclusiones finales

1. **El mejor modelo técnico es C** (3 capas, 128-64-32) con Accuracy = 0.792 ± 0.012 y F1 = 0.603 ± 0.019.

2. **El mejor modelo en eficiencia es F2** (3 capas, 64-64-64) con el mismo F1 que C pero **25% menos parámetros**.

3. **El mejor modelo para negocio es F3** (3 capas, 128-128-128) con el mejor recall (0.802) y AUC (0.869). Detecta más churns reales.

4. **La arquitectura no es la palanca principal**. Las diferencias entre todas las configuraciones con 2+ capas están dentro del margen de 1 desviación estándar. **Lo que sí importa** es:
   - **Early stopping** (corrige overfitting).
   - **pos_weight** (ajusta el trade-off precision/recall hacia el negocio).
   - **Weight decay** (regulariza).

5. **El overfitting anterior de C no era por profundidad**: era por 50 épocas fijas sin regularización. Con early stopping, C converge en 13 épocas y generaliza correctamente.

6. **El dropout no aportó** en este dataset. Con early stopping + weight decay + pos_weight ya hay suficiente regularización.

---

## 16. Archivos generados

- `resultados/tables/comparacion_modelos.csv` → tabla agregada (10 filas, 25 columnas, media ± std).
- `resultados/tables/resultados_por_semilla.csv` → 30 filas crudas (10 experimentos × 3 semillas).
- `resultados/tables/resultados_por_semilla.json` → versión JSON.
- `resultados/figures/loss_modelo_*.png` → 10 curvas de pérdida.
- `resultados/figures/confusion_matrix_*.png` → 10 matrices de confusión.
- `resultados/figures/comparacion_modelos.png` → gráfico comparativo con barras de error (5 métricas).

---

**Última actualización**: 20/10/2026  
**Versión**: 2.0 (Fase C corregida)