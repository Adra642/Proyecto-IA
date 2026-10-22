"""Búsqueda de hiperparámetros del modelo Lightning con Optuna.

Ejecutar desde la raíz del proyecto con ``python -m src.optuna_search``.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import optuna
import torch
from src.lightning_config import configure_warnings

configure_warnings()
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

import pytorch_lightning as pl
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback

from src.callbacks import F1ThresholdCallback
from src.model import ChurnLightningModule
from src.preprocessing import ChurnDataModule, load_clean


def objective(trial: optuna.Trial, frame) -> float:
    data = ChurnDataModule(
        frame,
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        num_workers=0,
    )
    data.setup()
    loss_name = trial.suggest_categorical("loss_name", ["focal", "weighted_bce"])
    model = ChurnLightningModule(
        input_dim=data.input_dim,
        hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        num_blocks=trial.suggest_int("num_blocks", 1, 5),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        dropout=trial.suggest_float("dropout", 0.05, 0.5),
        loss_name=loss_name,
        focal_gamma=trial.suggest_float("focal_gamma", 0.5, 4.0),
        focal_alpha=trial.suggest_float("focal_alpha", 0.5, 0.95),
        pos_weight=float(data.class_counts[0] / data.class_counts[1]),
    )
    trainer = pl.Trainer(
        max_epochs=60,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        callbacks=[
            F1ThresholdCallback(),
            PyTorchLightningPruningCallback(trial, monitor="val_f1_at_threshold"),
        ],
    )
    try:
        trainer.fit(model, datamodule=data)
        return float(trainer.callback_metrics["val_f1_at_threshold"])
    finally:
        del trainer, model, data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--results-csv",
        default="resultados/tables/optuna_trials.csv",
    )
    args = parser.parse_args()
    frame = load_clean()
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, frame), n_trials=args.trials)
    output_dir = Path(args.results_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(args.results_csv, index=False)
    print(f"Mejor F1 de validación: {study.best_value:.4f}")
    print(f"Mejores parámetros: {study.best_params}")
    print(f"CSV de trials: {args.results_csv}")


if __name__ == "__main__":
    main()
