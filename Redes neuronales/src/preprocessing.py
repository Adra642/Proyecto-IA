# src/preprocessing.py
"""
Preprocesamiento del dataset de churn bancario.

Este módulo es la fuente única de verdad para toda la fase de limpieza y
transformación de datos. Cualquier persona del equipo (P3 para el modelo,
P4 para el informe) debe importar desde aquí y NO duplicar código.

¿Por qué un módulo y no código en el notebook?
---------------------------------------------
- Reproducibilidad: cualquiera puede regenerar el dataset limpio desde el
  CSV crudo con `python -m src.preprocessing`.
- Trazabilidad: las decisiones de limpieza viven en git, no en la cabeza
  de nadie ni en un CSV generado a mano.
- No bloqueo: P3 puede trabajar con el DataModule sin esperar a que otro
  le mande un CSV por correo.

Flujo completo (de arriba a abajo en este archivo):
---------------------------------------------------
    CSV crudo
    → clean_raw()          [drop de columnas de identificadores]
    → build_preprocessor() [ColumnTransformer: numéricas + categóricas]
    → ChurnDataModule      [split 80/10/10 + TensorDataset + DataLoader]

Contrato con el resto del equipo:
---------------------------------
- Entrada esperada: CSV con 18 columnas (las originales del dataset).
- Salida del CSV limpio: 15 columnas, target `Exited` en {0, 1}, sin nulos.
- Salida del DataModule: lotes (X, y) con X en float32 shape (N, input_dim)
  e y en float32 shape (N, 1) → compatibles con nn.BCEWithLogitsLoss.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# `pathlib` se usa en lugar de `os.path` porque es más legible y portable
# entre Windows / Mac / Linux. Todo el equipo trabaja en el mismo Windows,
# pero P4 podría renderizar el Quarto en otro SO.
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# `pytorch_lightning` puede no estar instalado en todos los entornos. Si no
# lo está, definimos un stub mínimo para que el módulo se pueda importar
# (por ejemplo, desde los tests que no tocan la parte de PL). Así nadie se
# bloquea si aún no ha hecho `pip install -r requerimientos.txt`.
try:
    import pytorch_lightning as pl
    HAS_PL = True
except ImportError:
    HAS_PL = False
    class pl:
        class LightningDataModule:
            pass

# sklearn: piezas que usaremos en el ColumnTransformer.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler


# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------
# Todos los valores "mágicos" del pipeline se centralizan aquí para que
# cambiarlos sea una sola línea y no una cacería por todo el código.

# Semilla fija → splits reproducibles entre ejecuciones y entre personas.
# Si P3 ve métricas distintas a las nuestras, casi siempre es porque se
# cambió esta semilla sin avisar.
RANDOM_STATE = 42

# Decisión pendiente con Persona 1 sobre si `Complain` es fuga de información.
# Si su crosstab con `Exited` es prácticamente determinista, se cambia a True
# y el DataModule eliminará la columna antes del split.
# ⚠️ No cambiar sin documentarlo en el Quarto: afecta al input_dim que P3 usa.
DROP_COMPLAIN = False

# Columnas que se eliminan por ser identificadores o metadatos administrativos.
# Motivo (para el informe):
#   - RowNumber  → índice del archivo, no describe al cliente, mete ruido de orden.
#   - CustomerId → identificador único, cardinalidad = n.º filas, memoriza clientes.
#   - Surname    → alta cardinalidad + valores corruptos + proxy de origen geográfico.
# Si mañana alguien decide eliminar otra columna, se añade aquí y se actualiza
# EXPECTED_COLUMNS_AFTER_CLEAN en consecuencia.
ID_COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]

# Contrato con el resto del equipo: el CSV limpio debe tener 15 columnas.
# Si este número cambia, hay que avisar a P3 (input_dim cambia) y a P4 (informe).
EXPECTED_COLUMNS_AFTER_CLEAN = 15
TARGET_COLUMN = "Exited"

# Rutas por defecto relativas a la raíz del proyecto (donde está el README).
# Cualquier persona que ejecute el módulo desde la raíz encontrará el CSV.
DEFAULT_RAW_PATH = Path("data/Customer-Churn-Records.csv")
DEFAULT_CLEAN_PATH = Path("data/processed/df_limpio.csv")


# ---------------------------------------------------------------------------
# 1. Limpieza mínima del CSV crudo
# ---------------------------------------------------------------------------
# Esta sección SOLO elimina columnas y valida contratos de calidad.
# No hace transformaciones estadísticas (eso es cosa del ColumnTransformer).
# La separación es deliberada: la limpieza es una decisión de negocio
# ("estas columnas no son válidas"), la transformación es una decisión
# técnica ("así se escala mejor el modelo").

def clean_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza mínima acordada con Persona 1.

    Qué hace:
        - Elimina `RowNumber`, `CustomerId`, `Surname` si están presentes.
        - Valida que `Exited` sea binaria, que no haya nulos y que el
          número de columnas sea el esperado.

    Qué NO hace (a propósito):
        - No toca outliers → eso lo hace el IQRClipper dentro del pipeline.
        - No imputa nulos → eso lo hace SimpleImputer dentro del pipeline.
        - No codifica categóricas → eso lo hace OneHotEncoder dentro del pipeline.

    Parameters
    ----------
    df_raw : pd.DataFrame
        DataFrame tal cual viene del CSV crudo (18 columnas).

    Returns
    -------
    pd.DataFrame
        DataFrame limpio (15 columnas) listo para el pipeline.

    Raises
    ------
    AssertionError
        Si alguna de las validaciones de calidad no se cumple. Los asserts
        actúan como contrato: si alguien cambia el CSV crudo y rompe algo,
        peta aquí y no silenciosamente dentro del entrenamiento.
    """
    # `[c for c in ... if c in df_raw.columns]` es defensivo: si alguna de las
    # 3 columnas ya fue eliminada antes (por ejemplo, por otro script), no
    # rompemos. Esto hace la función idempotente.
    drop = [c for c in ID_COLUMNS_TO_DROP if c in df_raw.columns]
    df = df_raw.drop(columns=drop).copy()

    # --- Validaciones de calidad (contrato) ---
    # Si alguna falla, es mejor saberlo AHORA que dentro de 30 minutos de
    # entrenamiento con métricas raras.
    assert TARGET_COLUMN in df.columns, f"Falta la columna target '{TARGET_COLUMN}'"
    assert df[TARGET_COLUMN].isin([0, 1]).all(), "Exited debe ser binaria (0/1)"
    assert df.isnull().sum().sum() == 0, "No debe haber valores nulos"
    assert df.shape[1] == EXPECTED_COLUMNS_AFTER_CLEAN, (
        f"Se esperaban {EXPECTED_COLUMNS_AFTER_CLEAN} columnas, hay {df.shape[1]}"
    )

    return df


def load_raw(path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """
    Carga el CSV crudo desde disco.

    Existe como función separada (en lugar de un read_csv inline) para que
    los tests puedan hacer mock o pasar una ruta temporal sin tocar el CSV real.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el CSV en {path.resolve()}")
    return pd.read_csv(path)


def load_clean(
    raw_path: Path = DEFAULT_RAW_PATH,
    cache_path: Path = DEFAULT_CLEAN_PATH,
    use_cache: bool = False,
) -> pd.DataFrame:
    """
    Carga el dataset limpio (15 columnas).

    Comportamiento por defecto:
        Regenera desde el CSV crudo (fuente de verdad) y guarda una caché
        en `data/processed/df_limpio.csv`. Esto significa que si alguien
        modificó el CSV limpio a mano, la próxima llamada lo va a pisar.
        Es intencional: la fuente de verdad es el código, no el CSV.

    Con use_cache=True:
        Si existe la caché, la lee directamente sin regenerar. Útil para
        acelerar tests que se ejecutan muchas veces seguidas. No usar en
        producción ni en el pipeline de entrenamiento.

    Parameters
    ----------
    raw_path : Path
        Ruta al CSV crudo original.
    cache_path : Path
        Ruta donde se guarda la caché del CSV limpio.
    use_cache : bool
        Si True y la caché existe, la lee sin regenerar.

    Returns
    -------
    pd.DataFrame
        Dataset limpio listo para `build_preprocessor()`.
    """
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path)

    df = clean_raw(load_raw(raw_path))
    # `mkdir(parents=True, exist_ok=True)` → crea `data/processed/` si no
    # existe, sin fallar si ya está creada. Así no dependemos de que alguien
    # la haya creado a mano.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 2. Transformadores personalizados
# ---------------------------------------------------------------------------
# sklearn no trae un "recorta valores extremos usando cuantiles". Lo
# implementamos aquí como una clase compatible con el API de sklearn
# (BaseEstimator + TransformerMixin) para poder meterlo dentro de un
# Pipeline sin que se rompa nada.

class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Recorta (winsoriza) cada columna a los límites IQR aprendidos en fit.

    ¿Por qué?
        Los valores extremos de `Balance`, `EstimatedSalary` y `Age` son
        clientes reales, no errores. Eliminarlos sesga el dataset. Pero
        dejarlos intactos hace que el RobustScaler amplifique su influencia.
        Solución: recortarlos al rango [Q1 - 1.5·IQR, Q3 + 1.5·IQR].

    ¿Por qué es una clase y no una función?
        Porque necesita recordar los límites aprendidos en train. Una
        función `clip(x, a, b)` no sabe qué son `a` y `b` para cada columna.

    Contrato importante:
        - El `fit` se llama SOLO con datos de train (nunca con val/test).
        - El `transform` se puede llamar con train, val o test.
    """

    def __init__(self, factor: float = 1.5):
        # factor=1.5 es el estándar de Tukey. Cambiarlo hace el clipping
        # más o menos agresivo.
        self.factor = factor

    def fit(self, X, y=None):
        # `np.asarray(..., dtype=np.float64)` → trabajamos en float64 aquí
        # aunque el tensor final sea float32, para no perder precisión al
        # calcular cuantiles.
        v = np.asarray(X, dtype=np.float64)
        q1 = np.nanquantile(v, 0.25, axis=0)
        q3 = np.nanquantile(v, 0.75, axis=0)
        iqr = q3 - q1
        # Guardamos los límites como atributos con trailing underscore,
        # que es la convención de sklearn para "parámetros aprendidos".
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        v = np.asarray(X, dtype=np.float64)
        # np.clip aplica el mismo rango a todas las columnas de v.
        # La forma de self.lower_ / self.upper_ es (n_features,), así que
        # cada columna tiene su propio rango.
        return np.clip(v, self.lower_, self.upper_)


def log1p_nonnegative(values):
    """
    Aplica log1p a valores no negativos.

    ¿Por qué log1p y no log?
        log(0) = -inf. log1p(0) = 0. `Balance` puede valer 0 en este
        dataset (clientes sin saldo), así que necesitamos log1p.

    ¿Por qué solo a `Balance` y `EstimatedSalary`?
        Porque son las dos variables con colas muy largas (asimetría alta).
        Aplicar log1p a variables que ya están acotadas (Age, Tenure,
        Satisfaction Score) no aporta y complica la interpretación.
    """
    return np.log1p(np.clip(values, a_min=0, a_max=None))


# ---------------------------------------------------------------------------
# 3. Construcción del ColumnTransformer
# ---------------------------------------------------------------------------

def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """
    Construye el ColumnTransformer SIN ajustarlo.

    ¿Por qué no lo ajusta aquí?
        Porque el fit debe hacerse SOLO con train. Si lo ajustásemos aquí
        usando el frame completo, tendríamos fuga de información (las
        medianas, cuantiles y categorías ya sabrían de val/test).

    ¿Qué hace el ColumnTransformer?
        Aplica transformaciones distintas a grupos de columnas distintos,
        en paralelo, y concatena los resultados. El resultado es una matriz
        numérica que la red neuronal puede consumir.

    Grupos:
        financial   → Balance, EstimatedSalary (imputar + log1p + IQR + escala)
        numeric     → resto de numéricas         (imputar + IQR + escala)
        categorical → Geography, Gender, CardType (imputar + one-hot)

    Parameters
    ----------
    feature_frame : pd.DataFrame
        DataFrame SIN la columna target. Solo features.

    Returns
    -------
    ColumnTransformer
        Transformador sin ajustar. Llamar `.fit(X_train)` después.
    """
    # Identificamos grupos por tipo de dato automáticamente. Si mañana
    # alguien añade una columna numérica nueva, entra en `numeric` sin
    # tocar este código. Si añade una categórica, entra en `categorical`.
    categorical = feature_frame.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = feature_frame.select_dtypes(include=np.number).columns.tolist()
    # `[c for c in [...] if c in numeric]` → defensivo por si alguna de las
    # dos financieras no estuviera en el frame (no debería pasar, pero por
    # si acaso alguien experimenta con subconjuntos).
    financial = [c for c in ["Balance", "EstimatedSalary"] if c in numeric]
    other_numeric = [c for c in numeric if c not in financial]

    # Pipeline específico para las dos variables con cola larga. El orden
    # importa: imputar → log1p → winsorizar → escalar.
    #   - Imputar antes de log1p: si hay NaN, log1p(NaN) = NaN y luego el
    #     imputador no lo puede arreglar.
    #   - log1p antes de winsorizar: comprime la cola, así el IQR se calcula
    #     sobre una distribución menos sesgada.
    #   - Winsorizar antes de escalar: RobustScaler ya usa medianas/IQR,
    #     pero el clipping evita que los extremos dominen la escala.
    financial_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(log1p_nonnegative, feature_names_out="one-to-one")),
        ("winsorize", IQRClipper()),
        ("scale", RobustScaler()),
    ])

    # Pipeline para el resto de numéricas. Sin log1p porque no lo necesitan.
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorize", IQRClipper()),
        ("scale", RobustScaler()),
    ])

    # Pipeline para categóricas. `handle_unknown="ignore"` es CLAVE:
    # si en producción aparece una categoría que no estaba en train
    # (por ejemplo, un cuarto país), sklearn genera un vector de ceros
    # en lugar de petar. Se pierde algo de señal, pero no rompe el modelo.
    # `sparse_output=False` + `dtype=np.float32` para que el resultado sea
    # directamente un array denso en float32 listo para torch.
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])

    # `remainder="drop"` → cualquier columna que no aparezca en los tres
    # grupos se elimina silenciosamente. Es lo que queremos: si alguien
    # añade una columna nueva y no la mete en ningún grupo, no contamina.
    # `verbose_feature_names_out=False` → los nombres de las features de
    # salida no llevan prefijo del grupo (más limpio para debug).
    return ColumnTransformer(
        [
            ("financial", financial_pipe, financial),
            ("numeric", numeric_pipe, other_numeric),
            ("categorical", cat_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# 4. DataModule de PyTorch Lightning
# ---------------------------------------------------------------------------
# Encapsula split + preprocesador + DataLoaders. P3 solo necesita:
#     dm = ChurnDataModule(df)
#     dm.setup()
#     dm.train_dataloader()  # ya con shuffle=True
#     dm.val_dataloader()
#     dm.test_dataloader()
#     dm.input_dim           # para nn.Linear(dm.input_dim, ...)

class ChurnDataModule(pl.LightningDataModule):
    """
    DataModule con split 80/10/10 estratificado.

    Garantías importantes (auditadas y testeadas):
        - El preprocesador se ajusta SOLO con train.
        - El split es estratificado en las dos etapas (train/tmp y val/test).
        - Los tensores son float32: X con shape (N, input_dim), y con
          shape (N, 1). Compatible con nn.BCEWithLogitsLoss.
        - Si DROP_COMPLAIN=True, la columna se elimina ANTES del split,
          así la estratificación no se ve afectada.

    Atributos útiles después de `setup()`:
        - input_dim   : tamaño de entrada de la red (lo usa P3).
        - class_counts: [n_no_churn, n_churn] en train.
        - split_indices: dict con los índices usados por split (para tests).
    """

    def __init__(self, frame: pd.DataFrame, target: str = TARGET_COLUMN,
                 batch_size: int = 64, num_workers: int = 0):
        super().__init__()
        # `reset_index(drop=True)` es importante: si el frame que llega
        # tiene un índice raro (por ejemplo, tras un drop de filas), los
        # `iloc` sobre split_indices podrían dar resultados inesperados.
        self.frame = frame.reset_index(drop=True).copy()
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        # `pin_memory=True` acelera la transferencia CPU→GPU. Solo tiene
        # sentido si hay GPU; lo comprobamos una vez aquí y no en cada batch.
        self.pin_memory = torch.cuda.is_available()
        self.preprocessor = None
        # Guardia para que Lightning no vuelva a llamar a setup() y re-ajuste
        # el preprocesador. En Lightning, setup() se llama una vez por etapa
        # (fit, validate, test); con esta bandera solo hace el trabajo una vez.
        self._is_setup = False

    def setup(self, stage=None):
        """
        Prepara los tres splits. Idempotente: la segunda llamada no hace nada.
        """
        if self._is_setup:
            return

        # DROP_COMPLAIN es una decisión de negocio pendiente con P1.
        # Si se activa, quitamos la columna antes del split para que la
        # estratificación no la vea. Documentar en el Quarto si se activa.
        if DROP_COMPLAIN and "Complain" in self.frame.columns:
            self.frame = self.frame.drop(columns="Complain")

        idx = np.arange(len(self.frame))
        y = self.frame[self.target].to_numpy()

        # Split 80/10/10 en dos etapas. Estratificado en AMBAS para que la
        # tasa de churn sea ~igual en los tres conjuntos (test lo verifica).
        # Primera etapa: 80% train, 20% temporal (que se partirá en val+test).
        train_idx, tmp_idx = train_test_split(
            idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )
        # Segunda etapa: 50% val, 50% test (que es el 10% + 10% del total).
        # `stratify=y[tmp_idx]` es importante: estratificamos sobre las
        # etiquetas del subconjunto temporal, no sobre las originales.
        val_idx, test_idx = train_test_split(
            tmp_idx, test_size=0.50, random_state=RANDOM_STATE, stratify=y[tmp_idx]
        )

        train_df = self.frame.iloc[train_idx]
        val_df = self.frame.iloc[val_idx]
        test_df = self.frame.iloc[test_idx]

        # ⚠️ PUNTO CRÍTICO DE NO-LEAKAGE ⚠️
        # El preprocesador se ajusta SOLO con train_df. Si esto cambia,
        # los tests de no-leakage fallarán y las métricas de val/test
        # estarán contaminadas.
        self.preprocessor = build_preprocessor(train_df.drop(columns=self.target))
        self.preprocessor.fit(train_df.drop(columns=self.target))

        # Construimos los tres TensorDataset. Internamente cada uno llama a
        # `self.preprocessor.transform(...)` (solo transform, nunca fit).
        self.train_dataset = self._make(train_df)
        self.val_dataset = self._make(val_df)
        self.test_dataset = self._make(test_df)

        # Metadatos útiles para P3 y para los tests.
        self.input_dim = self.train_dataset.tensors[0].shape[1]
        self.class_counts = np.bincount(y[train_idx].astype(int), minlength=2)
        self.split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
        self._is_setup = True

    def _make(self, frame: pd.DataFrame) -> TensorDataset:
        """
        Convierte un DataFrame en TensorDataset.
        - X: transformado por el pipeline → float32.
        - y: float32 con shape (N, 1) → compatible con BCEWithLogitsLoss.
        """
        X = self.preprocessor.transform(frame.drop(columns=self.target))
        # `.reshape(-1, 1)` convierte (N,) en (N, 1). BCEWithLogitsLoss
        # espera esta forma (una probabilidad por muestra).
        y = frame[self.target].to_numpy(dtype=np.float32).reshape(-1, 1)
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    def _loader(self, dataset, shuffle=False):
        """
        Crea un DataLoader. `shuffle=True` solo para train (nunca para
        val/test: el orden no debe cambiar entre evaluaciones).
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # `persistent_workers=True` mantiene los workers vivos entre
            # epochs, ahorrando el coste de recrearlos. Solo tiene sentido
            # si num_workers > 0; si es 0, persistent_workers debe ser False.
            persistent_workers=self.num_workers > 0,
        )

    # Métodos que Lightning busca por nombre. Si P3 usa el Trainer de PL,
    # estos son los que se llaman automáticamente.
    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset)

    def test_dataloader(self):
        return self._loader(self.test_dataset)


# ---------------------------------------------------------------------------
# 5. Ejecución directa para generar df_limpio.csv
# ---------------------------------------------------------------------------
# `python -m src.preprocessing` ejecuta esto. Genera el CSV limpio en
# data/processed/. Útil para:
#   - Regenerar la caché tras un cambio de criterio de limpieza.
#   - Que P4 pueda inspeccionar el CSV sin abrir un notebook.
#   - Que P1 tenga su entregable sin depender de nadie.
if __name__ == "__main__":
    print("Cargando y limpiando CSV crudo...")
    df_clean = load_clean()
    print(f"OK. Shape: {df_clean.shape}")
    print(f"Columnas: {df_clean.columns.tolist()}")
    print(f"Tasa de churn: {df_clean[TARGET_COLUMN].mean():.3%}")
    print(f"Guardado en: {DEFAULT_CLEAN_PATH.resolve()}")