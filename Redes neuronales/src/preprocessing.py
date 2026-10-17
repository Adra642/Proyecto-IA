# src/preprocessing.py
"""
================================================================================
¿QUÉ ES ESTE ARCHIVO?
Este es el "limpiador" oficial de los datos del equipo.
Imagina que los datos crudos son harina, agua y huevos por separado.
Este archivo es la receta que los mezcla, quita las cáscaras (datos basura)
y los deja listos en un molde para que el "horno" (la red neuronal) los cocine.
================================================================================
"""

# --- IMPORTACIONES: Traemos las herramientas que necesitamos ---
from pathlib import Path  # Herramienta para manejar rutas de archivos (como carpetas y textos)
import numpy as np        # Herramienta matemática para hacer cálculos con listas de números
import pandas as pd       # Herramienta para manejar tablas de datos (como un Excel gigante)
import torch              # La librería de Inteligencia Artificial que usa tensores (los "ladrillos" de la IA)
from torch.utils.data import DataLoader, TensorDataset # Herramientas para dar los datos a la IA en trocitos (batches)

# Intentamos importar PyTorch Lightning. Si no está instalado, creamos un "simulacro" 
# para que el archivo no explote y pueda seguir funcionando.
try:
    import pytorch_lightning as pl
    HAS_PL = True
except ImportError:
    HAS_PL = False
    class pl:
        class LightningDataModule:
            pass

# Herramientas de scikit-learn (la caja de herramientas clásica de ciencia de datos)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler


# ==============================================================================
# --- CONFIGURACIÓN GLOBAL: Los ajustes de nuestro proyecto ---
# ==============================================================================

# Semilla aleatoria. Sirve para que si tú y yo ejecutamos el código, nos salga
# exactamente lo mismo. Sin esto, la IA dividiría los datos de forma distinta cada vez.
RANDOM_STATE = 42

# ¡DECISIÓN CLAVE DEL EQUIPO! 
# Eliminamos la columna "Complain" (queja) porque descubrimos que hacía "trampa".
# Si un cliente se queja, se va. Pero en la vida real, no sabes que se va a quejar
# hasta que ya se ha ido. Para que la IA sea útil, no puede usar esta columna.
DROP_COMPLAIN = True

# Columnas que son solo "etiquetas" (como el número de fila o el DNI del cliente).
# No sirven para predecir, así que las tiramos a la basura.
ID_COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]

# ¿Cuántas columnas nos deben quedar después de limpiar?
EXPECTED_COLUMNS_AFTER_CLEAN = 15

# El nombre de la columna que queremos predecir. 
# "Exited" significa: 0 (se queda) o 1 (se va del banco).
TARGET_COLUMN = "Exited"

# Dónde está el archivo original y dónde guardaremos el archivo limpio.
DEFAULT_RAW_PATH = Path("data/Customer-Churn-Records.csv")
DEFAULT_CLEAN_PATH = Path("data/processed/df_limpio.csv")


# ==============================================================================
# --- 1. LIMPIEZA MÍNIMA: Quitar la basura y comprobar que todo está bien ---
# ==============================================================================

def clean_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Esta función toma la tabla original (sucia) y le quita las columnas que no sirven.
    """
    # 1. Buscamos cuáles de las columnas basura existen realmente en la tabla
    drop = [c for c in ID_COLUMNS_TO_DROP if c in df_raw.columns]
    
    # 2. Las borramos. El ".copy()" es para no modificar la tabla original por accidente.
    df = df_raw.drop(columns=drop).copy()

    # 3. Hacemos comprobaciones de seguridad (como los seguros de un coche).
    # Si alguna de estas cosas falla, el programa se para y nos avisa.
    
    # ¿Existe la columna que queremos predecir?
    assert TARGET_COLUMN in df.columns, f"Falta la columna target '{TARGET_COLUMN}'"
    
    # ¿Los valores son solo 0 y 1? (No puede haber un 2 o un 5)
    assert df[TARGET_COLUMN].isin([0, 1]).all(), "Exited debe ser binaria (0/1)"
    
    # ¿Hay algún hueco vacío (nulo) en la tabla? La IA no sabe qué hacer con huecos.
    assert df.isnull().sum().sum() == 0, "No debe haber valores nulos"
    
    # ¿La tabla tiene exactamente 15 columnas como esperábamos?
    assert df.shape[1] == EXPECTED_COLUMNS_AFTER_CLEAN, (
        f"Se esperaban {EXPECTED_COLUMNS_AFTER_CLEAN} columnas, hay {df.shape[1]}"
    )
    
    return df


def load_raw(path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """
    Esta función simplemente va a la carpeta, coge el archivo CSV y lo convierte en tabla.
    Si el archivo no existe, da un error claro (en vez de un error feo de Python).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el CSV en {path.resolve()}")
    return pd.read_csv(path)


def load_clean(raw_path: Path = DEFAULT_RAW_PATH,
               cache_path: Path = DEFAULT_CLEAN_PATH,
               use_cache: bool = False) -> pd.DataFrame:
    """
    La función que usará todo el equipo. 
    Coge el CSV crudo, lo limpia, y lo guarda en la carpeta "processed".
    """
    # Si ya existe una copia limpia y nos dicen que la usemos, la usamos (es más rápido).
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path)
    
    # Si no, limpiamos desde cero.
    df = clean_raw(load_raw(raw_path))
    
    # Nos aseguramos de que la carpeta "processed" exista antes de guardar.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardamos la tabla limpia en un nuevo archivo CSV.
    # index=False significa que no queremos una columna extra con números de fila.
    df.to_csv(cache_path, index=False)
    
    return df


# ==============================================================================
# --- 2. TRANSFORMADORES: Poner los datos en el "idioma" que entiende la IA ---
# ==============================================================================

class IQRClipper(BaseEstimator, TransformerMixin):
    """
    ¿Qué es esto? Imagina que tienes las edades de los clientes, y de repente
    hay un cliente con 150 años (un error). La IA se asusta con esos valores extremos.
    Esta herramienta "recorta" los valores extremos para que no molesten.
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        # Aquí la herramienta "aprende" cuál es el límite normal (usando matemáticas).
        v = np.asarray(X, dtype=np.float64)
        q1 = np.nanquantile(v, 0.25, axis=0) # El 25% de los datos
        q3 = np.nanquantile(v, 0.75, axis=0) # El 75% de los datos
        iqr = q3 - q1                        # La distancia entre ellos
        self.lower_ = q1 - self.factor * iqr # El límite de abajo
        self.upper_ = q3 + self.factor * iqr # El límite de arriba
        return self

    def transform(self, X):
        # Aquí la herramienta recorta los valores que se pasen de los límites.
        v = np.asarray(X, dtype=np.float64)
        return np.clip(v, self.lower_, self.upper_)


def log1p_nonnegative(values):
    """
    El dinero (Balance, Salary) puede ser 1, 100, o 100.000. 
    Los números tan grandes confunden a la IA. Esta función "comprime" esos números
    para que todos estén en una escala parecida, sin perder la información.
    """
    return np.log1p(np.clip(values, a_min=0, a_max=None))


# ==============================================================================
# --- 3. PIPELINE DE TRANSFORMACIÓN: La "cadena de montaje" ---
# ==============================================================================

def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """
    En una tabla hay columnas de distinto tipo: números normales, columnas de dinero
    y columnas de texto (como el país o el género).
    Esta función crea una "cadena de montaje" distinta para cada tipo de columna.
    """
    
    # 1. Separamos las columnas por tipos.
    # Categorical: son las que tienen texto (ej: "Spain", "France", "Male", "Female")
    categorical = feature_frame.select_dtypes(include=["object", "category"]).columns.tolist()
    # Numeric: son las que tienen números (ej: Edad, Puntuación)
    numeric = feature_frame.select_dtypes(include=np.number).columns.tolist()
    
    # 2. Separamos las columnas de dinero (Balance, EstimatedSalary) del resto de números.
    financial = [c for c in ["Balance", "EstimatedSalary"] if c in numeric]
    other_numeric = [c for c in numeric if c not in financial]

    # 3. Cadena de montaje para el DINERO:
    # - Si hay huecos vacíos, los rellena con la mediana (el valor del medio).
    # - Comprime los números muy grandes con "log1p".
    # - Recorta los valores extremos (IQR).
    # - Escala todo para que esté entre -1 y 1 (RobustScaler).
    financial_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(log1p_nonnegative, feature_names_out="one-to-one")),
        ("winsorize", IQRClipper()),
        ("scale", RobustScaler()),
    ])

    # 4. Cadena de montaje para el RESTO DE NÚMEROS (Edad, Puntuación, etc.):
    # Igual que el dinero, pero sin comprimir (porque no son tan grandes).
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorize", IQRClipper()),
        ("scale", RobustScaler()),
    ])

    # 5. Cadena de montaje para el TEXTO (País, Género):
    # La IA no entiende la palabra "Spain". Entiende 1s y 0s.
    # OneHotEncoder convierte "Spain" en una columna nueva (1 si es Spain, 0 si no).
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")), # Rellena huecos con el país más común.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])

    # 6. El Director de la cadena de montaje (ColumnTransformer).
    # Le dice a cada columna por qué cadena debe pasar.
    return ColumnTransformer(
        [
            ("financial", financial_pipe, financial),
            ("numeric", numeric_pipe, other_numeric),
            ("categorical", cat_pipe, categorical),
        ],
        remainder="drop", # Si hay alguna columna que no hemos nombrado, la tiramos.
        verbose_feature_names_out=False,
    )


# ==============================================================================
# --- 4. DATAMODULE: El organizador del examen ---
# ==============================================================================

class ChurnDataModule(pl.LightningDataModule):
    """
    Esta es la clase más importante. Organiza los datos para el entrenamiento.
    Imagina que es el profesor que reparte los exámenes.
    """
    def __init__(self, frame: pd.DataFrame, target: str = TARGET_COLUMN,
                 batch_size: int = 64, num_workers: int = 0):
        super().__init__()
        self.frame = frame.reset_index(drop=True).copy()
        self.target = target
        self.batch_size = batch_size # Cuántos exámenes le damos a la IA de golpe (64).
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available() # Optimización para tarjetas gráficas.
        self.preprocessor = None
        self._is_setup = False # Un candado para que no se prepare dos veces.

    def setup(self, stage=None):
        # Si ya está preparado, nos vamos.
        if self._is_setup:
            return

        # ¡ELIMINAMOS COMPLAIN! Hacemos caso a la decisión que tomamos arriba.
        if DROP_COMPLAIN and "Complain" in self.frame.columns:
            self.frame = self.frame.drop(columns="Complain")

        # Creamos una lista de números [0, 1, 2, ..., 9999] para saber a qué fila nos referimos.
        idx = np.arange(len(self.frame))
        # Sacamos solo la columna que queremos predecir (Exited) para hacer el split.
        y = self.frame[self.target].to_numpy()

        # 1. EL PRIMER EXAMEN (Split):
        # Separamos el 80% de los datos para que la IA estudie (train),
        # y dejamos un 20% apartado (tmp) para hacer exámenes sorpresa después.
        # stratify=y significa: asegúrate de que en ambos grupos haya el mismo % de gente que se va.
        train_idx, tmp_idx = train_test_split(
            idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )
        
        # 2. EL EXAMEN FINAL:
        # Ese 20% que apartamos, lo partimos a la mitad: un 10% para validar mientras estudia,
        # y otro 10% para el examen final definitivo (test).
        val_idx, test_idx = train_test_split(
            tmp_idx, test_size=0.50, random_state=RANDOM_STATE, stratify=y[tmp_idx]
        )

        # 3. Creamos las tablas definitivas para cada grupo.
        train_df = self.frame.iloc[train_idx] # Datos de estudio
        val_df = self.frame.iloc[val_idx]     # Datos de práctica
        test_df = self.frame.iloc[test_idx]   # Datos del examen final

        # 4. ¡APRENDEMOS A LIMPIAR! 
        # El preprocesador (la cadena de montaje) SOLO se le enseña con train.
        # Es como darle las respuestas de estudio al profesor, pero no las del examen final.
        # Si usáramos los datos del examen para aprender, la IA haría trampa (leakage).
        self.preprocessor = build_preprocessor(train_df.drop(columns=self.target))
        self.preprocessor.fit(train_df.drop(columns=self.target))

        # 5. Pasamos los datos por la cadena de montaje.
        self.train_dataset = self._make(train_df)
        self.val_dataset = self._make(val_df)
        self.test_dataset = self._make(test_df)

        # 6. Guardamos datos importantes para el equipo:
        # ¿Cuántas columnas tiene la tabla final? (La respuesta es 19)
        self.input_dim = self.train_dataset.tensors[0].shape[1]
        
        # ¿Cuánta gente se queda (6370) y cuánta se va (1630)? Útil para ajustar el modelo.
        self.class_counts = np.bincount(y[train_idx].astype(int), minlength=2)
        
        # Guardamos los índices usados por si los tests quieren comprobarlos.
        self.split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
        
        # Cerramos el candado.
        self._is_setup = True

    def _make(self, frame: pd.DataFrame) -> TensorDataset:
        """
        Convierte las tablas limpias en "Tensores" (los bloques de construcción de PyTorch).
        """
        # Aplicamos la limpieza aprendida a las columnas (X).
        X = self.preprocessor.transform(frame.drop(columns=self.target))
        # Separamos la columna que queremos predecir (y) y la convertimos a float32.
        y = frame[self.target].to_numpy(dtype=np.float32).reshape(-1, 1)
        
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    def _loader(self, dataset, shuffle=False):
        """
        El "camión" que lleva los datos a la IA. 
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle, # En train=True (barajamos para que aprenda mejor), en val/test=False.
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    # Estas funciones son las que usará el compañero 3 para entrenar.
    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset)

    def test_dataloader(self):
        return self._loader(self.test_dataset)


# ==============================================================================
# --- 5. EJECUCIÓN DIRECTA: El botón de encendido ---
# ==============================================================================

# Todo lo que está dentro de este "if" SOLO se ejecuta cuando le damos a "Play" 
# o escribimos `python -m src.preprocessing` en la terminal.
if __name__ == "__main__":
    print("Cargando y limpiando CSV crudo...")
    df_clean = load_clean()
    print(f"OK. Shape: {df_clean.shape}")
    print(f"Columnas: {df_clean.columns.tolist()}")
    print(f"Tasa de churn: {df_clean[TARGET_COLUMN].mean():.3%}")
    print(f"Guardado en: {DEFAULT_CLEAN_PATH.resolve()}")