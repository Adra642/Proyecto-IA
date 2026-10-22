"""Búsqueda persistente de hiperparámetros del modelo con Optuna.

La búsqueda se puede ejecutar desde Python con :func:`run_study` o desde la
raíz del proyecto con ``python -m src.optuna_search``.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Sequence

import optuna
import pytorch_lightning as pl
import torch
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping

from src.callbacks import F1ThresholdCallback
from src.lightning_config import configure_warnings
from src.model import ChurnLightningModule
from src.preprocessing import ChurnDataModule, load_clean

configure_warnings()
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

DEFAULT_STUDY_NAME = "churn_nn_qmd"
DEFAULT_DATABASE_PATH = Path("artifacts") / "optuna" / "optuna.db"
DEFAULT_SEEDS = (42, 123, 2026)


class BestValidationScore(pl.Callback):
    """Conserva el mejor F1 de validación observado durante un entrenamiento."""

    def __init__(self) -> None:
        super().__init__()
        self.best_score: float | None = None

    def on_validation_end(self, trainer, pl_module) -> None:
        value = trainer.callback_metrics.get("val_f1_at_threshold")
        if value is not None:
            score = float(value.detach().cpu())
            if self.best_score is None or score > self.best_score:
                self.best_score = score


def _suggest_params(trial: optuna.Trial) -> dict:
    return {
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "loss_name": trial.suggest_categorical("loss_name", ["focal", "weighted_bce"]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "num_blocks": trial.suggest_int("num_blocks", 1, 5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.05, 0.5),
        "focal_gamma": trial.suggest_float("focal_gamma", 0.5, 4.0),
        "focal_alpha": trial.suggest_float("focal_alpha", 0.5, 0.95),
    }


def _run_seed(
    params: dict,
    trial: optuna.Trial,
    frame,
    seed: int,
    max_epochs: int,
    patience: int,
) -> float:
    pl.seed_everything(seed, workers=True)
    data = ChurnDataModule(
        frame,
        batch_size=int(params["batch_size"]),
        num_workers=0,
    )
    data.setup()
    model = ChurnLightningModule(
        input_dim=data.input_dim,
        hidden_dim=int(params["hidden_dim"]),
        num_blocks=int(params["num_blocks"]),
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        dropout=float(params["dropout"]),
        loss_name=params["loss_name"],
        focal_gamma=float(params["focal_gamma"]),
        focal_alpha=float(params["focal_alpha"]),
        pos_weight=float(data.class_counts[0] / data.class_counts[1]),
    )
    best_score = BestValidationScore()
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        deterministic=True,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[
            F1ThresholdCallback(),
            best_score,
            PyTorchLightningPruningCallback(
                trial,
                monitor="val_f1_at_threshold",
            ),
            EarlyStopping(
                monitor="val_f1_at_threshold",
                mode="max",
                patience=patience,
            ),
        ],
    )
    try:
        trainer.fit(model, datamodule=data)
        if best_score.best_score is None:
            raise RuntimeError(
                f"El trial {trial.number} no produjo F1 de validación "
                f"con la semilla {seed}."
            )
        return best_score.best_score
    finally:
        del trainer, model, data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def objective(
    trial: optuna.Trial,
    frame,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_epochs: int = 20,
    patience: int = 5,
) -> float:
    """Evalúa una configuración con varias semillas y devuelve su F1 medio."""
    params = _suggest_params(trial)
    seed_scores = [
        _run_seed(params, trial, frame, seed, max_epochs, patience)
        for seed in seeds
    ]
    trial.set_user_attr("seed_scores", seed_scores)
    return float(sum(seed_scores) / len(seed_scores))


def run_study(
    frame,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    n_trials: int = 30,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_epochs: int = 20,
    patience: int = 5,
    study_name: str = DEFAULT_STUDY_NAME,
) -> optuna.Study:
    """Ejecuta o reanuda una búsqueda Optuna persistida en SQLite."""
    if n_trials < 1:
        raise ValueError("n_trials debe ser positivo")
    if not seeds:
        raise ValueError("seeds no puede estar vacío")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs y patience deben ser positivos")

    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{database_path.resolve()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
        ),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            frame,
            seeds=seeds,
            max_epochs=max_epochs,
            patience=patience,
        ),
        n_trials=n_trials,
    )
    return study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    args = parser.parse_args()

    study = run_study(
        load_clean(),
        database_path=args.database,
        n_trials=args.trials,
        study_name=args.study_name,
    )
    print(f"Mejor F1 de validación: {study.best_value:.4f}")
    print(f"Mejores parámetros: {study.best_params}")
    print(f"Base de datos SQLite: {args.database}")


if __name__ == "__main__":
    main()
