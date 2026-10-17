# 🚀 Entrega de Preprocesamiento (Persona 2)

¡Hola equipo! Aquí tenéis toda la información sobre el módulo de preprocesamiento. 
Ya está todo subido a la rama `preprocessing-persona2`, probado y con los tests en verde. 

A continuación, os dejo los detalles específicos para cada uno.

---

## 🧠 Para la Persona 3 (CARLOS ROJAS)(Red Neuronal)

El `DataModule` ya se encarga de hacer el split (80/10/10), de limpiar los datos y de darte los datos en "tensores" listos para PyTorch. No tienes que preocuparte por el preprocesamiento.

### 1. La arquitectura de tu red
El número de variables de entrada (features) es **19**. 
*   **Nota importante:** Originalmente eran 20, pero hemos eliminado la columna `Complain` porque producía *leakage* (fuga de información, hacía trampa). 
*   Tu primera capa debe ser: `nn.Linear(dm.input_dim, ...)` que equivale a `nn.Linear(19, ...)`.

### 2. Cómo obtener los datos
Así es como debes inicializar los datos en tu script de entrenamiento:

```python
from src.preprocessing import load_clean, ChurnDataModule

# 1. Cargamos el CSV limpio
df = load_clean()

# 2. Inicializamos el DataModule (batch_size de 64)
dm = ChurnDataModule(df, batch_size=64)
dm.setup() # Hace el split y el fit del preprocesador

3. Los números que necesitas para el entrenamiento
dm.input_dim: Devuelve 19.

dm.class_counts: Devuelve [6370, 1630]. Es decir, en el conjunto de entrenamiento hay 6370 personas que se quedan (clase 0) y 1630 que se van (clase 1). El 79.6% son de la clase 0 y el 20.4% de la clase 1. (Puedes usar esto para calcular el pos_weight en tu función de pérdida si quieres balancear).

4. Formato de salida de la red
Tu modelo debe escupir 1 solo logit (un número suelto, no una probabilidad).

Las etiquetas (y_batch) ya vienen con shape (N, 1) y tipo float32.

Usa la función de pérdida nn.BCEWithLogitsLoss() (es la más estable para este tipo de problemas).

5. Bucle de entrenamiento
Así se ve un paso de entrenamiento:


for x_batch, y_batch in dm.train_dataloader():
    # x_batch.shape será (64, 19)
    # y_batch.shape será (64, 1)
    
    logits = model(x_batch)              # salida shape (64, 1)
    loss = criterion(logits, y_batch)    # BCEWithLogitsLoss
    
    # ... backpropagation ...3. Los números que necesitas para el entrenamiento
dm.input_dim: Devuelve 19.

dm.class_counts: Devuelve [6370, 1630]. Es decir, en el conjunto de entrenamiento hay 6370 personas que se quedan (clase 0) y 1630 que se van (clase 1). El 79.6% son de la clase 0 y el 20.4% de la clase 1. (Puedes usar esto para calcular el pos_weight en tu función de pérdida si quieres balancear).

4. Formato de salida de la red
Tu modelo debe escupir 1 solo logit (un número suelto, no una probabilidad).

Las etiquetas (y_batch) ya vienen con shape (N, 1) y tipo float32.

Usa la función de pérdida nn.BCEWithLogitsLoss() (es la más estable para este tipo de problemas).

5. Bucle de entrenamiento

Así se ve un paso de entrenamiento


for x_batch, y_batch in dm.train_dataloader():
    # x_batch.shape será (64, 19)
    # y_batch.shape será (64, 1)
    
    logits = model(x_batch)              # salida shape (64, 1)
    loss = criterion(logits, y_batch)    # BCEWithLogitsLoss
    
    # ... backpropagation ...