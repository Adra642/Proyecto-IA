# 📚 Glosario completo del proyecto

**Redes Neuronales · Clasificación de Churn Bancario**

Para: Persona 1, Persona 2, Persona 3, Persona 4 — todos novatos en IA.

Este documento define **todos** los términos técnicos que aparecen en `src/preprocessing.py` y en el resto del proyecto. Está pensado para que cualquier persona del equipo pueda leer el código sin perderse.

> 💡 **Cómo previsualizarlo con colores en VS Code:**
> Pulsa `Ctrl + Shift + V` (o `Cmd + Shift + V` en Mac) para abrir la vista previa al lado.

---

## Índice

1. [Conceptos generales de Machine Learning / IA](#1-conceptos-generales-de-machine-learning--ia)
2. [El problema: churn bancario](#2-el-problema-churn-bancario)
3. [Python general](#3-python-general)
4. [Sistema de archivos y rutas (pathlib)](#4-sistema-de-archivos-y-rutas-pathlib)
5. [NumPy](#5-numpy)
6. [Pandas](#6-pandas)
7. [PyTorch](#7-pytorch)
8. [PyTorch Lightning](#8-pytorch-lightning)
9. [Scikit-learn (sklearn)](#9-scikit-learn-sklearn)
10. [Estadística y matemáticas](#10-estadística-y-matemáticas)
11. [Ingeniería de software y buenas prácticas](#11-ingeniería-de-software-y-buenas-prácticas)
12. [Convenciones específicas del proyecto](#12-convenciones-específicas-del-proyecto)

---

## 1. Conceptos generales de Machine Learning / IA


### Machine Learning (ML) / Aprendizaje automático
Subcampo de la IA. En lugar de programar reglas a mano (*"si X entonces Y"*), le damos datos al ordenador y él "aprende" los patrones solo. En este proyecto, el modelo aprende de 10.000 clientes pasados para predecir si uno nuevo se irá.

### Deep Learning / Aprendizaje profundo
Subcampo del ML basado en redes neuronales con muchas capas. Es lo que usamos aquí (PyTorch).

### Red neuronal
Modelo matemático inspirado en el cerebro. Tiene "neuronas" organizadas en capas que se pasan información entre sí. Ajustando los pesos de las conexiones entre neuronas, el modelo aprende.

### Modelo
El resultado del entrenamiento. En nuestro caso, un archivo con los pesos ajustados que sirve para predecir. Antes de entrenar, el modelo es una cáscara vacía con pesos aleatorios.

### Entrenamiento / Training
Proceso de ajustar los pesos del modelo mostrándole ejemplos. En cada iteración, el modelo predice, se compara con la realidad, y se corrige.

### Predicción / Inference
Usar un modelo ya entrenado para dar una respuesta sobre datos nuevos. Ejemplo: dado un cliente nuevo, predecir si se irá o no.

### Features / Características / Variables predictoras / X
Las columnas de entrada del modelo. Lo que sabe del cliente: edad, salario, si tiene tarjeta, etc. En el código aparecen como `X`.

### Target / Etiqueta / Variable objetivo / y
Lo que queremos predecir. En nuestro caso, `Exited` (0 = se queda, 1 = se va). En el código aparece como `y`.

### Clasificación
Tarea de ML donde la salida es una categoría. Ejemplo: "gato" o "perro". En nuestro caso: "se va" o "se queda".

### Clasificación binaria
Clasificación con exactamente 2 categorías. Es nuestro caso.

### Clasificación multiclase
Clasificación con 3 o más categorías. **NO** es nuestro caso.

### Regresión
Tarea de ML donde la salida es un número continuo. Ejemplo: predecir el precio de una casa. **NO** es nuestro caso.

### Split / División (Train / Validation / Test)
Dividir el dataset en tres trozos:

- **Train** (entrenamiento): el modelo aprende de aquí. 80%.
- **Validation** (validación): para ajustar hiperparámetros. 10%.
- **Test** (prueba): evaluación final, nunca vista antes. 10%.

Nunca se debe evaluar en el mismo conjunto con el que se entrenó.

### Leakage / Fuga de información / Data leakage
Cuando el modelo, durante el entrenamiento, tiene acceso a información que no debería (porque en producción no la tendría). Ejemplo clásico: si una columna se rellenó **después** de saber el resultado, el modelo la usa como atajo y en producción falla. En este proyecto sospechamos de `Complain` (queja del cliente): si se registró después de que el cliente se fuera, es leakage.

### Overfitting / Sobreajuste
Cuando el modelo aprende de memoria los ejemplos de entrenamiento y luego falla en datos nuevos. Es como estudiar solo las respuestas del examen anterior: en el mismo examen sacas 10, en uno nuevo sacas 3.

### Underfitting / Subajuste
Cuando el modelo es tan simple que ni siquiera aprende de los datos de entrenamiento. Falla en todo.

### Hiperparámetro
Valor que se elige **antes** de entrenar y no se aprende de los datos. Ejemplos: learning rate, batch size, número de capas, número de neuronas por capa.

### Parámetro
Valor que el modelo **aprende** durante el entrenamiento. Son los pesos y los sesgos de las neuronas.

### Época / Epoch
Una pasada completa por todo el conjunto de entrenamiento. Si tienes 8.000 ejemplos y `batch_size=64`, una época son 125 batches.

### Batch / Lote
Subconjunto pequeño de ejemplos que se procesa de golpe. En vez de mirar los 8.000 ejemplos y actualizar los pesos, miramos 64, actualizamos, miramos otros 64... Es más eficiente y generaliza mejor.

### Batch size / Tamaño de lote
Cuántos ejemplos por batch. En el proyecto: `64`.

### Learning rate / Tasa de aprendizaje / LR
Cuánto se corrige el modelo en cada paso. Muy alto → no converge. Muy bajo → tarda muchísimo. Típico: `0.001`.

### Loss / Pérdida / Función de pérdida
Número que mide cuánto se equivoca el modelo. El objetivo es minimizarlo. En clasificación binaria se usa `BCEWithLogitsLoss`.

### Optimizer / Optimizador
Algoritmo que actualiza los pesos para reducir la loss. Los más usados: SGD y Adam. Nosotros usaremos **Adam**.

### Métricas
Números que miden el rendimiento del modelo:

- **Accuracy**: % de aciertos.
- **Precision**: de los que predije "se va", cuántos se fueron de verdad.
- **Recall**: de los que se fueron de verdad, cuántos detecté.
- **F1**: media armónica de precision y recall.
- **AUC**: área bajo la curva ROC, mide discriminación global.

### Desbalanceo de clases / Class imbalance
Cuando una clase tiene muchos más ejemplos que otra. En churn, ~80% no se va, ~20% se va. Esto hay que gestionarlo con `stratify` y eligiendo bien las métricas.

### Stratify / Estratificar
Al hacer un split, garantizar que la proporción de clases es la misma en train, val y test que en el dataset original. Se activa con `stratify=y` en `train_test_split`.

### Semilla / Seed / Random state
Número que fija la aleatoriedad. Si usas la misma semilla, obtienes siempre el mismo resultado. En el proyecto: `RANDOM_STATE = 42`. Es una tradición friki (42 = *"La respuesta a la vida, el universo y todo lo demás"*, de Douglas Adams).

### Cardinalidad
Número de valores distintos que tiene una columna. Si es igual al número de filas → es un identificador (no sirve para ML). `CustomerId` tiene cardinalidad 10.000 en este dataset.

### Proxy
Variable que sustituye a otra. Ejemplo: si el apellido predice el país de origen, es un "proxy de geografía" pero disfrazado. Es peligroso porque el modelo aprende algo que no quieres.

### Entrenamiento end-to-end
Entrenar todo el sistema de golpe (preprocesamiento + modelo) sin pasos manuales intermedios. Es lo que queremos aquí.

### Pipeline
Cadena de transformaciones aplicadas en orden. Los datos pasan por cada paso y salen transformados. Ejemplo: imputar → escalar → modelo.

### Dataset
Conjunto de datos. En nuestro caso, `Customer-Churn-Records.csv`.

### Churn
Palabra inglesa para "abandono de clientes". Un cliente hace "churn" cuando se va de la empresa.

### Exited
Nombre de la columna target en este dataset. Vale 0 si el cliente se quedó y 1 si se fue.

---

## 2. El problema: churn bancario

### Churn / Abandono
Cliente que deja de usar los servicios del banco.

### Churn rate / Tasa de churn
Porcentaje de clientes que se van. En este dataset: ~20%.

### Cliente activo / IsActiveMember
Columna binaria (0/1) que indica si el cliente usa activamente el banco.

### NumOfProducts
Número de productos bancarios que tiene el cliente (cuenta, tarjeta, hipoteca, etc.). A más productos, menos probable que se vaya.

### Balance
Saldo en la cuenta del cliente. Tiene muchos ceros y outliers.

### EstimatedSalary
Salario anual estimado del cliente. También tiene outliers.

### Complain / Queja
Columna binaria (0/1): si el cliente ha puesto una queja. Sospechosa de leakage porque puede estar muy correlacionada con `Exited`.

### Satisfaction Score
Puntuación de satisfacción del cliente, de 1 a 5.

### Card Type
Tipo de tarjeta del cliente: `DIAMOND`, `GOLD`, `SILVER`, `PLATINUM`.

### Point Earned
Puntos acumulados por el cliente en programas de fidelización.

### Geography
País del cliente: `France`, `Spain`, `Germany`.

### Gender
Género del cliente: `Female`, `Male`.

---

## 3. Python general

### Python
Lenguaje de programación interpretado, muy usado en data science e IA. Es el lenguaje del proyecto.

### Comentario
Texto que el ordenador ignora. En Python empieza con `#`.

```python
# esto es un comentario
```

### Docstring / Documentación
Comentario especial entre triples comillas `"""..."""` justo después de definir una función o clase. Explica qué hace.

### Módulo
Archivo `.py` que se puede importar desde otros archivos. `src/preprocessing.py` es un módulo.

### Paquete
Carpeta con varios módulos. Se reconoce porque tiene un archivo `__init__.py` dentro. En el proyecto, `src/` es un paquete.

### Import / Importar
Traer código de otro módulo para usarlo.

```python
import pandas as pd
```

### `from ... import ...`
Forma alternativa: importar solo una parte.

```python
from pathlib import Path
```

### `as` / Alias
Renombrar algo al importarlo. `import numpy as np` → usamos numpy como `np` para escribir menos.

### Función
Bloque de código reutilizable. Se define con `def nombre():`.

```python
def clean_raw(df_raw): ...
```

### Parámetro
Variable que recibe una función. En `clean_raw(df_raw)`, `df_raw` es el parámetro.

### Argumento
Valor concreto que se pasa a la función al llamarla. En `clean_raw(mi_dataframe)`, `mi_dataframe` es el argumento.

### Return / Retorno
Valor que devuelve una función con `return`. Sin return, devuelve `None`.

### Clase / Class
Plantilla para crear objetos. Se define con `class Nombre:`.

### Objeto / Instancia
Ejemplar concreto de una clase. `IQRClipper()` crea un objeto de la clase `IQRClipper`.

### Método
Función que pertenece a una clase. Se define dentro con `def` pero lleva `self` como primer parámetro.

### Atributo
Variable que pertenece a un objeto. Ejemplo: `self.factor = 1.5`.

### `self`
Palabra clave que representa al propio objeto. Sirve para acceder a sus atributos y métodos desde dentro.

### `__init__`
Método especial que se ejecuta al crear un objeto. Es el "constructor".

### `__main__`
Nombre del módulo cuando se ejecuta directamente. El bloque `if __name__ == "__main__":` solo se ejecuta si el archivo se lanza con `python script.py`, no si se importa.

### `def` / Definir
Palabra clave para definir una función o método.

### Clase base / Superclase / Herencia
Una clase puede heredar de otra. Ejemplo: `ChurnDataModule` hereda de `pl.LightningDataModule`. Heredar significa "coger todo lo que tiene la clase padre y añadir lo propio".

### `super().__init__()`
Llamar al constructor de la clase padre. Necesario para que la herencia funcione bien.

### Try / Except
Bloque para manejar errores. Intenta hacer algo (`try`); si falla, ejecuta otra cosa (`except`).

```python
try:
    import pytorch_lightning
except ImportError:
    # si no está instalado, hacemos otra cosa
```

### `ImportError`
Error que salta cuando Python no encuentra un módulo que intentas importar.

### Assert / Assertion
Comprobación que se hace en medio del código. Si falla, el programa se detiene con un error.

```python
assert x > 0, "x debe ser positivo"
```

Sirve como "contrato": si las condiciones no se cumplen, mejor parar que seguir con datos raros.

### `AssertionError`
Error que se lanza cuando un assert falla.

### `None`
Valor especial que significa "nada" o "vacío". Es lo que devuelve una función sin return.

### True / False
Valores booleanos. Verdadero y falso.

### Type hint / Anotación de tipo
Indicar de qué tipo es un parámetro o return. No es obligatorio, es documentación.

```python
def f(x: int) -> str:
```

Pero Python no lo fuerza en tiempo de ejecución.

### F-string
Cadena de texto con formato. Se pone una `f` delante y dentro se meten variables con `{variable}`.

```python
f"Shape: {df.shape}"
```

### Float / Número decimal
Número con decimales. Ejemplo: `3.14`.

### Int / Entero
Número sin decimales. Ejemplo: `42`.

### Bool / Booleano
`True` o `False`.

### Str / String / Cadena
Texto. Ejemplo: `"Hola"`.

### Lista / List
Colección ordenada de elementos. Ejemplo: `[1, 2, 3]`. `ID_COLUMNS_TO_DROP = [...]` es una lista.

### Tupla / Tuple
Colección ordenada pero **inmutable**. Ejemplo: `(1, 2, 3)`. En el `Pipeline`:

```python
("imputer", SimpleImputer(strategy="median"))
```

Cada paso del pipeline es una tupla `(nombre, transformador)`.

### Diccionario / Dict
Colección de pares clave→valor. Ejemplo: `{"a": 1, "b": 2}`. `split_indices = {"train": ..., "val": ..., "test": ...}` es un dict.

### Idempotente
Operación que, si la ejecutas dos veces seguidas, da el mismo resultado que ejecutarla una sola vez. `clean_raw` es idempotente porque si le pasas un DataFrame ya limpio, no rompe.

### Defensivo / Defensive programming
Escribir código que anticipa casos raros y los maneja sin romperse. Ejemplo: `[c for c in ID_COLUMNS_TO_DROP if c in df_raw.columns]` se protege ante el caso de que alguna columna ya no esté.

### Guard clause / Guardia
Condición al principio de una función que sale rápido si no hay nada que hacer. Ejemplo:

```python
if self._is_setup:
    return
```

---

## 4. Sistema de archivos y rutas (pathlib)

### Path / Ruta
Dirección de un archivo o carpeta en el sistema. Ejemplo: `data/Customer-Churn-Records.csv`.

### Ruta relativa
Empieza desde donde estás. Ejemplo: `data/file.csv` (asume que estás en la raíz del proyecto).

### Ruta absoluta
Empieza desde la raíz del sistema. Ejemplo en Windows: `C:\Users\yo\Proyecto\data\file.csv`.

### `pathlib`
Librería de Python para trabajar con rutas de archivos. Sustituye a la antigua `os.path`. Es más limpia y portable.

### `Path`
Clase principal de pathlib. Representa una ruta.

```python
Path("data/Customer-Churn-Records.csv")
```

### `resolve()`
Convierte una ruta relativa en absoluta. Útil para mostrar dónde está un archivo cuando hay errores.

### `exists()`
Comprueba si un archivo o carpeta existe:

```python
if not path.exists(): raise FileNotFoundError(...)
```

### `mkdir` / Make directory
Crear una carpeta.

### `parents=True`
Argumento de `mkdir`: si faltan carpetas intermedias, las crea también. Ejemplo: crear `data/processed/` cuando `data/` ya existe. Si `data/` no existiera, `parents=True` lo crea también.

### `exist_ok=True`
Argumento de `mkdir`: si la carpeta ya existe, no da error. Sin esto, `mkdir` peta si la carpeta ya está creada.

### `parent`
Atributo de una ruta que devuelve la carpeta contenedora.

```python
Path("data/processed/df.csv").parent == Path("data/processed")
```

### File / Archivo
Documento en disco. Puede ser `.py`, `.csv`, `.txt`, etc.

### Carpeta / Directorio
Contenedor de archivos.

### CSV / Comma-separated values
Formato de archivo de datos. Cada línea es una fila, los valores se separan por comas. Es el formato del dataset.

### Raíz del proyecto
Carpeta donde está el `README.md`. Todos los scripts se ejecutan desde aquí.

### `os.path`
Módulo antiguo de Python para rutas. Sustituido por `pathlib`.

---

## 5. NumPy

### NumPy / np
Librería fundamental para cálculo numérico en Python. Maneja arrays (matrices) de forma muy eficiente. Se importa como `np`.

### Array
Estructura de datos de NumPy. Como una lista pero mucho más rápida y con operaciones matemáticas vectorizadas.

### `np.asarray(X, dtype=np.float64)`
Convierte `X` (que puede ser lista, DataFrame, etc.) en un array de NumPy. `dtype` indica el tipo de los números.

### `np.float64`
Número decimal con doble precisión (64 bits). Más preciso que `float32` pero ocupa el doble de memoria. Se usa para cálculo de cuantiles.

### `np.nanquantile(v, 0.25, axis=0)`
Calcula cuantiles ignorando NaN. `v`: array. `0.25`: cuantil. `axis=0`: por columnas.

### `np.clip(v, a, b)`
Recorta los valores de `v` para que estén entre `a` y `b`. Los que estén por debajo de `a` se vuelven `a`, los que estén por encima de `b` se vuelven `b`.

### `np.log1p(x)`
Calcula `log(1 + x)`. Igual que `np.log(1 + x)` pero más preciso para valores pequeños.

### `np.arange(n)`
Devuelve un array `[0, 1, 2, ..., n-1]`.

### `np.bincount(x, minlength=2)`
Cuenta cuántas veces aparece cada valor entero en `x`. Ejemplo: `np.bincount([0, 0, 1, 1, 1], minlength=2)` → `[2, 3]`.

### Axis
Eje de un array. En una matriz 2D: `axis=0` → por columnas; `axis=1` → por filas.

### NaN
*"Not a Number"*. Valor especial para representar vacíos o indefinidos en NumPy/Pandas.

### Inf / Infinito
Valor especial que representa infinito. `log(0) = -inf`.

### Vectorización
Aplicar operaciones a todo un array de golpe, sin bucles. Es lo que hace NumPy internamente y por eso es tan rápido.

### float32 / float64
Tipos de número decimal. `float32` ocupa 32 bits (4 bytes) y es más rápido en GPU. `float64` ocupa 64 bits (8 bytes) y tiene más precisión. PyTorch usa `float32` por defecto; Python usa `float64` por defecto.

### int64
Entero de 64 bits. Es el tipo que devuelve `np.bincount` y las etiquetas de clase.

---

## 6. Pandas

### Pandas / pd
Librería para manipular datos tabulares (como Excel pero en código). Se importa como `pd`.

### DataFrame
Estructura de Pandas. Como una tabla de Excel: filas y columnas con nombres. `df` es un DataFrame.

### Series
Una sola columna de un DataFrame. `df["Age"]` es una Series.

### Index / Índice
Identificador de cada fila. Por defecto `0, 1, 2, ..., n-1`.

### `df.shape`
Tupla `(n_filas, n_columnas)`. Ejemplo: `(10000, 18)`. `df.shape[0]` → número de filas. `df.shape[1]` → número de columnas.

### `df.columns`
Lista con los nombres de las columnas.

### `pd.read_csv(path)`
Lee un CSV y devuelve un DataFrame.

### `df.to_csv(path, index=False)`
Guarda un DataFrame como CSV. `index=False` → no guardar el índice como primera columna.

### `df.drop(columns=[...])`
Elimina columnas.

### `df.copy()`
Hace una copia independiente. Sin `copy`, los cambios en uno afectan al otro.

### `df.select_dtypes(include=[...])`
Selecciona columnas por tipo.

```python
df.select_dtypes(include=["object", "category"])  # texto/categoría
df.select_dtypes(include=np.number)               # numéricas
```

### `object`
Tipo de Pandas para columnas de texto (strings).

### `category`
Tipo de Pandas para columnas categóricas (pocos valores repetidos).

### `df.isnull()`
Devuelve un DataFrame booleano: `True` donde hay NaN.

### `df.isnull().sum()`
Cuenta cuántos NaN hay por columna.

### `df.isnull().sum().sum()`
Suma total de NaN en todo el DataFrame.

### `df.isin([0, 1])`
Devuelve máscara booleana: `True` donde el valor está en la lista.

### `df.duplicated().sum()`
Cuenta filas duplicadas.

### `df.to_numpy()`
Convierte el DataFrame en array de NumPy.

### `df.iloc[idx]`
Selecciona filas por posición (no por etiqueta). El "i" de `iloc` significa "integer".

### `df.loc[idx]`
Selecciona filas por etiqueta del índice.

### `df.reset_index(drop=True)`
Reinicia el índice a `0, 1, 2, ...`. `drop=True` → no guarda el índice viejo como columna.

### `df.mean()`
Media de cada columna numérica.

### NaN / Missing value / Valor nulo
Celda vacía. En Pandas se ve como `NaN`.

### Imputación
Rellenar valores nulos con algo (media, mediana, moda, constante). Es lo que hace `SimpleImputer`.

---

## 7. PyTorch

### PyTorch / torch
Librería de deep learning. Alternativa a TensorFlow. La usa Meta (Facebook).

### Tensor
Estructura de datos principal de PyTorch. Como un array de NumPy pero que puede vivir en GPU y calcular gradientes.

### `torch.tensor(data, dtype=...)`
Crea un tensor a partir de una lista o array. `dtype` indica el tipo de los números.

### `torch.float32`
Tipo de número decimal de 32 bits. Es el estándar en PyTorch.

### `torch.long`
Tipo entero de 64 bits. Se usa para índices y etiquetas de clasificación multiclase.

### `torch.cuda.is_available()`
Devuelve `True` si hay GPU NVIDIA disponible para usar con CUDA.

### CUDA
Plataforma de NVIDIA para cálculo en GPU.

### GPU
Tarjeta gráfica. Mucho más rápida que la CPU para operaciones matriciales, que es lo que hace una red neuronal.

### CPU
Procesador principal del ordenador.

### TensorDataset
Clase de PyTorch que empaqueta `X` e `y` juntos. Cada elemento del dataset es un par `(x_i, y_i)`.

### DataLoader
Envuelve un Dataset y lo recorre en batches. Se encarga de mezclar (`shuffle`), paralelizar la carga (`num_workers`) y optimizar el transporte a GPU (`pin_memory`).

### `batch_size`
Número de ejemplos por batch.

### Shuffle
Barajar los datos antes de cada época. Solo se hace en train, nunca en val/test.

### `num_workers`
Número de procesos paralelos que cargan datos. Con 0, todo en el proceso principal (más lento pero sin problemas en Windows).

### `pin_memory`
Reservar los datos en memoria "pinned" para transferirlos a GPU más rápido. Solo tiene sentido si hay GPU.

### `persistent_workers`
Mantener vivos los workers entre épocas. Ahorra tiempo pero solo funciona si `num_workers > 0`.

### `reshape(-1, 1)`
Cambiar la forma de un array. `-1` significa "deduce esa dimensión". Ejemplo: `(8000,)` → `reshape(-1, 1)` → `(8000, 1)`.

### Shape / Forma
Tupla con las dimensiones de un tensor. Ejemplo: `(64, 26)` = 64 filas y 26 columnas.

### `nn` / `torch.nn`
Submódulo de PyTorch con bloques para redes neuronales (capas, funciones de pérdida, etc.).

### `nn.Linear(in, out)`
Capa lineal (fully connected) con `in` entradas y `out` salidas.

### `nn.ReLU()`
Función de activación que pone a 0 los negativos. Introduce no linealidad en la red.

### `nn.BCEWithLogitsLoss`
Función de pérdida para clasificación **binaria**. Espera logits (números reales sin sigmoide) y etiquetas en `{0, 1}` con shape `(N, 1)`.

### `nn.CrossEntropyLoss`
Función de pérdida para clasificación **multiclase**. Espera logits y etiquetas enteras (`long`).

### Logit
Valor real (antes de aplicar sigmoide o softmax). La red devuelve logits; la función de pérdida los convierte a probabilidades.

### Sigmoide / Sigmoid
Función matemática que convierte cualquier número en un valor entre 0 y 1. Se usa para clasificación binaria.

### Forward / Pass
Pasar datos por la red para obtener una predicción.

### Backward / Backpropagation
Calcular los gradientes para actualizar los pesos. PyTorch lo hace solo con `loss.backward()`.

### Gradiente
Vector que indica cómo cambia la loss al cambiar cada peso. Se usa para actualizar los pesos en la dirección que reduce la loss.

### Peso / Weight
Número que multiplica una entrada en una neurona. Se ajusta durante el entrenamiento.

### Sesgo / Bias
Número que se suma a la salida de una neurona. También se ajusta.

### Optimizador
Algoritmo que usa los gradientes para actualizar los pesos. Ejemplos: SGD, Adam.

### `torch.optim.Adam`
Optimizador Adam. Es el estándar por defecto para muchas tareas.

### `lr` / Learning rate
Cuánto se corrige el modelo en cada paso. Ejemplo: `lr=1e-3` = `0.001`.

---

## 8. PyTorch Lightning

### PyTorch Lightning / pl
Capa por encima de PyTorch que organiza el código de entrenamiento en clases y reduce boilerplate.

### `pl.LightningDataModule`
Clase base para encapsular datos. Tiene métodos `setup()`, `train_dataloader()`, `val_dataloader()`, `test_dataloader()`.

### `pl.LightningModule`
Clase base para el modelo + entrenamiento + validación + test. Agrupa el modelo, la loss, el optimizador y los pasos de cada fase.

### Trainer
Clase de Lightning que ejecuta el entrenamiento. Se le pasa el `LightningModule` y el `DataModule`.

### Stage
Fase del entrenamiento: `"fit"`, `"validate"`, `"test"`, `"predict"`. El método `setup()` recibe el stage para saber en qué fase está.

### `setup()`
Método del DataModule donde se preparan los datasets (split, preprocesado, tensores). Se llama una vez por stage.

### Idempotente (aplicado a setup)
Si llamas `setup()` dos veces, la segunda no hace nada. Se consigue con la bandera `_is_setup`.

### Callback
Función que se ejecuta automáticamente en ciertos momentos del entrenamiento. Ejemplo: `EarlyStopping`.

### Early stopping
Parar el entrenamiento si la métrica de validación no mejora durante N épocas.

### Checkpoint
Guardar el estado del modelo en disco para poder reanudar o usar después.

### Logger
Objeto que registra métricas durante el entrenamiento para visualizarlas (TensorBoard, CSV, etc.).

---

## 9. Scikit-learn (sklearn)

### Scikit-learn / sklearn
Librería de ML clásico (no redes neuronales). Trae transformadores, métricas, modelos lineales, árboles, etc.

### Estimator / Estimador
Cualquier objeto de sklearn que se ajuste con `fit()` y se aplique con `transform()` o `predict()`.

### BaseEstimator
Clase base de sklearn. Heredar de ella da funcionalidad común (`get_params`, `set_params`, etc.).

### TransformerMixin
Mezclador de sklearn que añade el método `fit_transform()`.

### Fit
Ajustar un transformador o modelo a datos. Aprende los parámetros internos (medias, cuantiles, etc.). **Importante:** `fit` **solo** con train.

### Transform
Aplicar el transformador ya ajustado a datos nuevos. **No** aprende nada, solo aplica lo aprendido. `transform` con train, val o test.

### `fit_transform`
Hace `fit` y `transform` en un solo paso. Se usa con train.

### ColumnTransformer
Transformador de sklearn que aplica pipelines distintos a grupos de columnas distintos, y luego concatena los resultados.

### Pipeline
Cadena de transformadores que se aplican en orden.

### SimpleImputer
Rellena valores nulos. Estrategias: `"mean"`, `"median"`, `"most_frequent"`, `"constant"`.

### `strategy="median"`
Rellenar con la mediana de cada columna. Robusto frente a outliers.

### `strategy="most_frequent"`
Rellenar con el valor más común. Útil para categóricas.

### FunctionTransformer
Envuelve una función Python para que sklearn la trate como transformador.

### `feature_names_out="one-to-one"`
Indica que el transformador no cambia el número de columnas. Necesario en sklearn moderno para que los nombres de columnas se propaguen.

### OneHotEncoder
Codifica categóricas como columnas binarias (0/1). Ejemplo: `"France"`, `"Spain"`, `"Germany"` → 3 columnas.

### `handle_unknown="ignore"`
Si en test/producción aparece una categoría que no estaba en train, en vez de petar, devuelve todo ceros.

### `sparse_output=False`
Devolver una matriz densa (todos los ceros explícitos) en vez de una matriz dispersa (solo los no-ceros). Necesario para PyTorch.

### RobustScaler
Escala las columnas restando la mediana y dividiendo por el IQR. Robusto a outliers, a diferencia de `StandardScaler`.

### StandardScaler
Escala restando la media y dividiendo por la desviación típica. Sensible a outliers.

### MinMaxScaler
Escala al rango `[0, 1]`.

### `train_test_split`
Función de sklearn que divide un dataset en train y test (o train y temporal).

### `test_size=0.20`
Proporción del conjunto que va al split de test/temporal.

### `random_state=RANDOM_STATE`
Semilla para reproducibilidad.

### `stratify=y`
Mantener la proporción de clases en ambos lados del split.

### `remainder="drop"`
En `ColumnTransformer`: qué hacer con las columnas que no están en ningún grupo. `"drop"` las elimina.

### `verbose_feature_names_out=False`
En `ColumnTransformer`: no añadir prefijos del grupo a los nombres de salida.

### Regressor / Classifier
Modelos de sklearn para regresión o clasificación. No los usamos directamente (usamos PyTorch), pero el concepto es el mismo.

### Trailing underscore / Guion bajo final
Convención de sklearn: atributos que se aprenden en `fit()` acaban en `_`. Ejemplo: `self.lower_`, `self.upper_`.

### Parámetro aprendido
Valor que se calcula durante `fit()` y se usa en `transform()`. No se pasa como argumento al constructor.

---

## 10. Estadística y matemáticas

### Media / Mean / Promedio
Suma de todos los valores dividida entre cuántos hay. Sensible a outliers.

### Mediana
Valor central cuando ordenas los datos. El 50% está por debajo, el 50% por encima. Robusta a outliers.

### Moda
Valor más frecuente. Se usa para categóricas.

### Cuantil
Punto que divide una distribución. Ejemplo: el cuantil 0.25 (Q1) es el valor por debajo del cual está el 25% de los datos.

### Q1 / Primer cuartil
Cuantil 0.25. Percentil 25.

### Q3 / Tercer cuartil
Cuantil 0.75. Percentil 75.

### IQR / Interquartile range / Rango intercuartílico
Diferencia entre Q3 y Q1: `IQR = Q3 - Q1`. Mide la dispersión del 50% central de los datos. Es robusto frente a outliers.

### Outlier / Valor atípico
Valor que se aleja mucho del resto. Regla estándar: valores fuera de `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]` son outliers.

### Winsorización / Winsorization
Recortar los valores extremos a un límite. En vez de eliminar los outliers, los sustituimos por el valor del límite.

### Clipping / Recorte
Igual que winsorización. Limitar valores a un rango.

### Logaritmo / Log
Operación matemática que comprime valores grandes. `log(1) = 0`, `log(10) ≈ 2.3`, `log(1000) ≈ 6.9`.

### `log1p`
`log(1 + x)`. Se usa cuando `x` puede ser 0, porque `log(0) = -infinito`.

### Distribución
Cómo se reparten los valores de una variable. Puede ser normal, uniforme, sesgada, etc.

### Asimetría / Skewness
Cuánto se desvía una distribución de la simetría. `Balance` tiene mucha asimetría porque hay muchos ceros y algunos valores muy altos.

### Cola larga / Long tail
Distribución con muchos valores pequeños y unos pocos muy grandes. Ejemplo: salarios, ingresos, precios.

### Correlación
Cuánto se mueven dos variables juntas. Va de -1 a 1.
- `+1` → cuando una sube, la otra también.
- `-1` → cuando una sube, la otra baja.
- `0` → no relación lineal.

### Correlación de Pearson
Mide relación **lineal**. Sensible a outliers.

### Correlación de Spearman
Mide relación **monótona**. Más robusta.

### Crosstab / Tabla cruzada
Tabla que cruza dos variables categóricas. Útil para ver relaciones. Ejemplo: `Complain` vs `Exited`.

### Frecuencia / Frequency
Cuántas veces aparece un valor.

### Proporción / Porcentaje
Frecuencia dividida entre el total.

### Estratificación
Garantizar que las proporciones de una variable categórica se mantienen en todos los subconjuntos.

### Desviación típica / Desviación estándar / std
Medida de cuánto se dispersan los valores respecto a la media.

### Varianza
Cuadrado de la desviación típica.

### Normalización
Escalar valores a un rango concreto, normalmente `[0, 1]`.

### Estandarización
Escalar valores para que media = 0 y desviación = 1.

### Escalado robusto
Escalar usando mediana e IQR en lugar de media y desviación.

### Dimensionalidad
Número de columnas/features que tiene el dataset.

### Alta dimensionalidad
Cuando hay muchísimas columnas. Peligro de overfitting.

---

## 11. Ingeniería de software y buenas prácticas

### Pipeline (de ingeniería de datos)
Cadena de pasos que transforman los datos desde el formato crudo hasta el formato que consume el modelo.

### Fuente de verdad / Source of truth
El único sitio del que se considera válido un dato o decisión. En este proyecto, **el código es la fuente de verdad**, no el CSV limpio.

### Caché
Copia de un resultado para no recalcularlo. `df_limpio.csv` es una caché derivada del CSV crudo.

### Artefacto
Producto derivado. El CSV limpio, el modelo entrenado, un gráfico, todo son artefactos.

### Contrato
Acuerdo explícito entre dos partes del sistema. Los `assert` de `clean_raw` son un contrato: *"el CSV limpio debe tener 15 columnas"*.

### Magic number / Número mágico
Número suelto en el código sin explicación. Se arregla con constantes globales.

### Constante global
Variable que se define arriba del archivo y se usa en varios sitios. Ejemplo: `RANDOM_STATE = 42`.

### Refactorización
Reorganizar código sin cambiar su comportamiento.

### DRY / Don't repeat yourself
Principio: no duplicar código. Si dos personas necesitan la misma transformación, se pone en un módulo compartido.

### KISS / Keep it simple, stupid
Principio: preferir lo simple. No hacer pipelines de 10 pasos si con 4 funciona.

### YAGNI / You aren't gonna need it
Principio: no añadir cosas "por si acaso". Solo lo que hace falta.

### Blindaje / Hardening
Hacer el código resistente a casos raros y errores.

### Reproducibilidad
Poder obtener exactamente los mismos resultados ejecutando el mismo código con los mismos datos.

### Trazabilidad
Poder rastrear de dónde viene cada resultado. Los cambios en git son la trazabilidad de este proyecto.

### Auditoría
Revisar el código de otra persona para detectar problemas.

### Mock / Mockear
En tests: sustituir una dependencia real (una base de datos, un archivo) por una falsa para no depender de ella.

### Git
Sistema de control de versiones. Guarda el historial de cambios.

### Rama / Branch
Versión paralela del proyecto. Cada persona trabaja en su rama (`preprocessing-persona2`) y luego se fusiona a `main`.

### Commit
Guardar un conjunto de cambios con un mensaje.

### Push
Subir los commits al repositorio remoto (GitHub).

### Merge / Fusionar
Juntar los cambios de una rama con otra.

### Pull Request / PR
Propuesta de fusionar tu rama con `main`. Se revisa antes de aceptarla.

### Main / Master
Rama principal del repositorio.

### README
Archivo de documentación que explica el proyecto.

### `.gitignore`
Archivo que le dice a git qué **no** debe rastrear (por ejemplo, `.venv/`, `__pycache__/`, `data/processed/*.csv`).

### Venv / Entorno virtual
Carpeta con una instalación aislada de Python y sus librerías. Evita que un proyecto rompa a otro.

### Requerimientos / requirements.txt
Lista de librerías que necesita el proyecto con sus versiones. En este proyecto se llama `requerimientos.txt`.

### Pytest
Framework de tests de Python. Ejecuta archivos `test_*.py`.

### Test unitario
Comprobación automática de que una función hace lo que debe.

### Fixture (pytest)
Función que prepara algo para varios tests (por ejemplo, cargar el dataset una vez y reutilizarlo).

### Módulo ejecutable / `__main__`
Un archivo `.py` se puede importar (como módulo) o ejecutar (con `python script.py`). El bloque `if __name__ == "__main__":` solo se ejecuta en el segundo caso.

---

## 12. Convenciones específicas del proyecto

### P1 / Persona 1
Encargada del dataset y el EDA. Genera gráficos y decide qué columnas se eliminan.

### P2 / Persona 2 *(tú)*
Encargada del preprocesamiento. Escribe `src/preprocessing.py` y los tests.

### P3 / Persona 3
Encargada del modelo PyTorch y los experimentos.

### P4 / Persona 4
Encargada del Quarto y la integración final del informe.

### `RANDOM_STATE = 42`
Semilla global del proyecto. Todos los splits y procesos aleatorios usan esta semilla para que sean reproducibles.

### `DROP_COMPLAIN = False`
Bandera que decide si se elimina la columna `Complain` por posible leakage. Pendiente de decisión con P1.

### `ID_COLUMNS_TO_DROP`
Lista `["RowNumber", "CustomerId", "Surname"]` de columnas que se eliminan por ser identificadores.

### `EXPECTED_COLUMNS_AFTER_CLEAN = 15`
Contrato: el CSV limpio debe tener exactamente 15 columnas.

### `TARGET_COLUMN = "Exited"`
Nombre de la columna que queremos predecir.

### `DEFAULT_RAW_PATH`
Ruta por defecto al CSV crudo.

### `DEFAULT_CLEAN_PATH`
Ruta por defecto a la caché del CSV limpio.

### Churn
Abandono de clientes. El problema que resolvemos.

### `input_dim`
Dimensión de entrada de la red neuronal. Se calcula tras el preprocesamiento porque el OneHot cambia el número de columnas. P3 lo lee de `dm.input_dim`.

### Split 80/10/10
Convención del proyecto: 80% train, 10% val, 10% test.

### Winsorización IQR
Recortar a `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]`.

### `log1p`
Aplicado a `Balance` y `EstimatedSalary`.

### `RobustScaler`
Escalado final de todas las numéricas.

### OneHot
Codificación de `Geography`, `Gender`, `Card Type`.

### `handle_unknown="ignore"`
Los OneHot toleran categorías nuevas en producción.

### `feature_names_out="one-to-one"`
El `FunctionTransformer` no cambia el número de columnas.

### DataModule
`ChurnDataModule` encapsula split + preprocesador + DataLoaders.

### Checkpoint de no-leakage
El `preprocessor.fit()` se llama **solo** con `train_df`. Cualquier cambio aquí rompe los tests y las métricas.

### Quarto / `.qmd`
Formato de documento del informe final. Se renderiza a HTML.

### Render
Proceso de generar el HTML del Quarto: `quarto render`.

### `resultados/figures/`
Carpeta donde van los gráficos. **Ojo:** en este proyecto la carpeta se llama `resultados/`, no `results/`.

### `requerimientos.txt`
Archivo con las dependencias. **Ojo:** no se llama `requirements.txt` en este proyecto.

---

## Fin del glosario

Si algún término del código no aparece aquí, avisa a P2 para que lo añada.

**Última actualización:** _por definir_