# tests/test_preprocessing.py
"""
================================================================================
¿QUÉ ES ESTE ARCHIVO?
Este archivo es el "Control de Calidad" de nuestra fábrica de datos.
Imagina que eres un inspector en una fábrica de coches. Antes de que el coche 
salga a la calle, tienes una lista de comprobaciones: ¿tiene ruedas? ¿tiene volante?
Este archivo hace exactamente lo mismo, pero con los datos antes de dárselos a la IA.
Cada función que empieza por "test_" es una comprobación. Si alguna falla, se 
enciende una luz roja y el programa nos avisa.
================================================================================
"""

# --- IMPORTACIONES: Traemos las herramientas ---
import numpy as np
import pandas as pd
import pytest # Pytest es el "inspector jefe". Es la herramienta que ejecuta estas pruebas.

# Traemos las funciones de nuestro archivo principal para poder examinarlas.
from src.preprocessing import (
    clean_raw, load_clean, build_preprocessor, ChurnDataModule,
    TARGET_COLUMN, EXPECTED_COLUMNS_AFTER_CLEAN,
)


# ==============================================================================
# --- EL "PREPARATIVO" (FIXTURE) ---
# ==============================================================================

@pytest.fixture(scope="module")
def df_clean():
    """
    ¿Qué es esto? Imagina que vas a hacer 9 exámenes distintos. En vez de 
    preparar los datos desde cero antes de cada examen, los preparamos UNA VEZ 
    y los dejamos listos para todos.
    Esta función llama a nuestro limpiador de datos y se lo pasa a cada test.
    """
    return load_clean()


# ==============================================================================
# --- LAS COMPROBACIONES (LOS TESTS) ---
# ==============================================================================

def test_clean_raw_drops_id_columns(df_clean):
    """
    Test 1: Comprobar que hemos tirado la basura.
    No queremos que columnas que no sirven (como el número de fila o el DNI) 
    se cuelen en nuestro examen final.
    """
    # Revisamos que "RowNumber", "CustomerId" y "Surname" NO estén en la tabla.
    for col in ["RowNumber", "CustomerId", "Surname"]:
        assert col not in df_clean.columns


def test_clean_raw_has_expected_shape(df_clean):
    """
    Test 2: Contar las piezas del puzzle.
    Queremos estar seguros de que la tabla final tiene exactamente 15 columnas.
    Ni una más, ni una menos. Si tiene 14, es que hemos borrado algo que servía.
    """
    assert df_clean.shape[1] == EXPECTED_COLUMNS_AFTER_CLEAN


def test_target_is_binary(df_clean):
    """
    Test 3: Comprobar que la respuesta es un "Sí" o un "No".
    En nuestro caso, "Exited" (si el cliente se va del banco) solo puede ser:
    0 (Se queda) o 1 (Se va). No puede haber un 2, un 5 o un texto.
    """
    assert df_clean[TARGET_COLUMN].isin([0, 1]).all()


def test_no_missing_values(df_clean):
    """
    Test 4: Buscar huecos vacíos.
    La IA se vuelve loca si hay celdas en blanco (nulos). 
    Este test se asegura de que no haya ni un solo hueco en toda la tabla.
    """
    assert df_clean.isnull().sum().sum() == 0


def test_no_duplicates(df_clean):
    """
    Test 5: Buscar clones (Filas duplicadas).
    Si por error hemos copiado la misma fila del cliente dos veces, 
    la IA pensará que ese cliente vale el doble. Queremos datos únicos.
    """
    assert df_clean.duplicated().sum() == 0


def test_split_is_stratified(df_clean):
    """
    Test 6: Repartir la tarta de forma justa.
    Vamos a dividir los datos en 3 grupos (Train, Val, Test).
    Este test comprueba que el "porcentaje de gente que se va" (la tasa de churn)
    sea casi idéntico en los tres grupos. Si en uno es del 20% y en otro del 5%, 
    es que hemos repartido mal los datos.
    """
    # Creamos el organizador de datos (DataModule) y lo ponemos a trabajar.
    dm = ChurnDataModule(df_clean, batch_size=64)
    dm.setup()
    
    rates = []
    # Miramos qué porcentaje de gente se va en cada uno de los 3 grupos.
    for split in ["train", "val", "test"]:
        idx = dm.split_indices[split] # Las filas que le tocaron a ese grupo
        rates.append(df_clean.iloc[idx][TARGET_COLUMN].mean())
        
    # Si la diferencia entre el grupo con más % y el de menos % es mayor a 0.02 (2%),
    # saltará un error. Queremos que estén muy igualados.
    assert max(rates) - min(rates) < 0.02, f"Tasas muy dispares: {rates}"


def test_tensor_shapes_match(df_clean):
    """
    Test 7: Comprobar las cajas (Tensores).
    La IA no lee tablas, lee "cajas" de números llamadas Tensores.
    Este test comprueba que la caja de preguntas (X) y la caja de respuestas (Y) 
    tienen exactamente el mismo número de filas. No puede haber 100 preguntas y 99 respuestas.
    También comprueba que X tiene las columnas que esperamos.
    """
    dm = ChurnDataModule(df_clean, batch_size=64)
    dm.setup()
    x, y = dm.train_dataset.tensors
    
    # Mismo número de filas
    assert x.shape[0] == y.shape[0]
    # X tiene el número correcto de columnas (features)
    assert x.shape[1] == dm.input_dim


def test_complain_is_not_in_features(df_clean):
    """
    Test 8: ¡La prueba del "chivato" (Leakage)!
    Habíamos decidido eliminar la columna "Complain" (queja) porque hacía trampa.
    Este test se asegura al 100% de que esa columna ha desaparecido de los datos 
    que le vamos a dar a la IA. Si sigue ahí, la IA sacará un 99% de acierto falso.
    """
    dm = ChurnDataModule(df_clean, batch_size=64)
    dm.setup()
    assert "Complain" not in dm.frame.columns


def test_input_dim_is_19(df_clean):
    """
    Test 9: El número mágico final.
    Después de quitar IDs y quitar "Complain", y de convertir textos a números,
    a la IA le deben llegar exactamente 19 columnas.
    Este es el número que usaremos para construir la red neuronal.
    """
    dm = ChurnDataModule(df_clean, batch_size=64)
    dm.setup()
    assert dm.input_dim == 19