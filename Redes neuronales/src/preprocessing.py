# src/preprocessing.py
"""Carga, limpieza y preparación reproducible de los datos de churn."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.lightning_config import configure_warnings

configure_warnings()

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler


# Semilla fija para que la partición sea reproducible.
RANDOM_STATE = 42

# `Complain` contiene información posterior al churn y produciría fuga de datos.
DROP_COMPLAIN = True

ID_COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]
EXPECTED_COLUMNS_AFTER_CLEAN = 15
TARGET_COLUMN = "Exited"
DEFAULT_RAW_PATH = Path("data/raw/Customer-Churn-Records.csv")
DEFAULT_CLEAN_PATH = Path("data/processed/df_limpio.csv")


# ==============================================================================
# --- 1. LIMPIEZA MÍNIMA: Quitar la basura y comprobar que todo está bien ---
# ==============================================================================

def clean_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Elimina identificadores y valida el contrato del conjunto de datos."""
    drop = [c for c in ID_COLUMNS_TO_DROP if c in df_raw.columns]
    df = df_raw.drop(columns=drop).copy()
    assert TARGET_COLUMN in df.columns, f"Falta la columna target '{TARGET_COLUMN}'"
    assert df[TARGET_COLUMN].isin([0, 1]).all(), "Exited debe ser binaria (0/1)"
    assert df.isnull().sum().sum() == 0, "No debe haber valores nulos"
    assert df.shape[1] == EXPECTED_COLUMNS_AFTER_CLEAN, (
        f"Se esperaban {EXPECTED_COLUMNS_AFTER_CLEAN} columnas, hay {df.shape[1]}"
    )
    
    return df


def load_raw(path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Lee el CSV original y ofrece un error explícito si no está disponible."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el CSV en {path.resolve()}")
    return pd.read_csv(path)


def load_clean(raw_path: Path = DEFAULT_RAW_PATH,
               cache_path: Path = DEFAULT_CLEAN_PATH,
               use_cache: bool = False) -> pd.DataFrame:
    """Limpia el CSV, guarda una copia reproducible y devuelve el resultado."""
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path)
    df = clean_raw(load_raw(raw_path))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


# ==============================================================================
# --- 2. TRANSFORMADORES: Poner los datos en el "idioma" que entiende la IA ---
# ==============================================================================

class IQRClipper(BaseEstimator, TransformerMixin):
    """Limita valores numéricos a los límites del rango intercuartílico."""
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        v = np.asarray(X, dtype=np.float64)
        q1 = np.nanquantile(v, 0.25, axis=0) # El 25% de los datos
        q3 = np.nanquantile(v, 0.75, axis=0) # El 75% de los datos
        iqr = q3 - q1                        # La distancia entre ellos
        self.lower_ = q1 - self.factor * iqr # El límite de abajo
        self.upper_ = q3 + self.factor * iqr # El límite de arriba
        return self

    def transform(self, X):
        v = np.asarray(X, dtype=np.float64)
        return np.clip(v, self.lower_, self.upper_)


def log1p_nonnegative(values):
    """Aplica ``log1p`` después de truncar valores negativos no válidos."""
    return np.log1p(np.clip(values, a_min=0, a_max=None))


# ==============================================================================
# --- 3. PIPELINE DE TRANSFORMACIÓN: La "cadena de montaje" ---
# ==============================================================================

def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """Construye el preprocesador sin ajustar estadísticas sobre validación o test."""
    categorical = feature_frame.select_dtypes(include=[str, "category"]).columns.tolist()
    numeric = feature_frame.select_dtypes(include=np.number).columns.tolist()
    financial = [c for c in ["Balance", "EstimatedSalary"] if c in numeric]
    other_numeric = [c for c in numeric if c not in financial]

    financial_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(log1p_nonnegative, feature_names_out="one-to-one")),
        ("winsorize", IQRClipper()),
        ("scale", RobustScaler()),
    ])

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorize", IQRClipper()),
        ("scale", RobustScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])
    return ColumnTransformer(
        [
            ("financial", financial_pipe, financial),
            ("numeric", numeric_pipe, other_numeric),
            ("categorical", cat_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ==============================================================================
# --- 4. DATAMODULE: El organizador del examen ---
# ==============================================================================

class ChurnDataModule(pl.LightningDataModule):
    """Divide, transforma y sirve los datos a PyTorch Lightning."""
    def __init__(self, frame: pd.DataFrame, target: str = TARGET_COLUMN,
                 batch_size: int = 64, num_workers: int = 0):
        super().__init__()
        self.frame = frame.reset_index(drop=True).copy()
        self.target = target
        if batch_size < 1:
            raise ValueError("batch_size debe ser positivo")
        self.batch_size = batch_size
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available()
        self.preprocessor = None
        self._is_setup = False

    def setup(self, stage=None):
        if self._is_setup:
            return
        if DROP_COMPLAIN and "Complain" in self.frame.columns:
            self.frame = self.frame.drop(columns="Complain")
        idx = np.arange(len(self.frame))
        y = self.frame[self.target].to_numpy()
        train_idx, tmp_idx = train_test_split(
            idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )
        
        val_idx, test_idx = train_test_split(
            tmp_idx, test_size=0.50, random_state=RANDOM_STATE, stratify=y[tmp_idx]
        )

        train_df = self.frame.iloc[train_idx]
        val_df = self.frame.iloc[val_idx]
        test_df = self.frame.iloc[test_idx]
        self.preprocessor = build_preprocessor(train_df.drop(columns=self.target))
        self.preprocessor.fit(train_df.drop(columns=self.target))
        self.train_dataset = self._make(train_df)
        self.val_dataset = self._make(val_df)
        self.test_dataset = self._make(test_df)

        self.input_dim = self.train_dataset.tensors[0].shape[1]
        self.class_counts = np.bincount(y[train_idx].astype(int), minlength=2)
        self.split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
        self._is_setup = True

    def _make(self, frame: pd.DataFrame) -> TensorDataset:
        """Transforma un subconjunto en un ``TensorDataset``."""
        X = self.preprocessor.transform(frame.drop(columns=self.target))
        y = frame[self.target].to_numpy(dtype=np.float32).reshape(-1, 1)
        
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    def _loader(self, dataset, shuffle=False):
        """Crea un cargador con la misma configuración en todas las particiones."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

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