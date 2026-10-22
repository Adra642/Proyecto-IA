"""Entrenamiento reproducible del modelo de churn con PyTorch Lightning."""

from __future__ import annotations

import argparse
import pandas as pd

import torch
from src.lightning_config import configure_warnings

configure_warnings()

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.callbacks import F1ThresholdCallback
from src.model import ChurnLightningModule
from src.preprocessing import ChurnDataModule, load_clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-optuna-best", action="store_true")
    parser.add_argument(
        "--optuna-csv",
        default="resultados/tables/optuna_trials.csv",
    )
    args = parser.parse_args()

    best_params = {}
    if args.use_optuna_best:
        trials = pd.read_csv(args.optuna_csv)
        completed = trials[trials["state"] == "COMPLETE"].dropna(subset=["value"])
        if completed.empty:
            raise ValueError(f"No hay trials de Optuna completados en {args.optuna_csv}")
        best_trial = completed.loc[completed["value"].idxmax()]
        best_params = {
            key.removeprefix("params_"): value
            for key, value in best_trial.items()
            if key.startswith("params_") and pd.notna(value)
        }
        args.batch_size = int(best_params["batch_size"])
        print(
            f"Trial de Optuna usado: {int(best_trial['number'])}; "
            f"F1={float(best_trial['value']):.4f}"
        )

    data = ChurnDataModule(
        load_clean(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data.setup()
    pos_weight = float(data.class_counts[0] / data.class_counts[1])
    model = ChurnLightningModule(
        input_dim=data.input_dim,
        pos_weight=pos_weight,
        hidden_dim=int(best_params.get("hidden_dim", 128)),
        num_blocks=int(best_params.get("num_blocks", 3)),
        learning_rate=float(best_params.get("learning_rate", 1e-3)),
        weight_decay=float(best_params.get("weight_decay", 1e-4)),
        dropout=float(best_params.get("dropout", 0.25)),
        loss_name=best_params.get("loss_name", "focal"),
        focal_gamma=float(best_params.get("focal_gamma", 2.0)),
        focal_alpha=float(best_params.get("focal_alpha", 0.75)),
    )
    checkpoint = ModelCheckpoint(
        monitor="val_f1_at_threshold",
        mode="max",
        save_top_k=1,
        filename="churn-{epoch:02d}-{val_f1_at_threshold:.3f}",
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        deterministic=True,
        callbacks=[
            F1ThresholdCallback(),
            checkpoint,
            EarlyStopping(monitor="val_f1_at_threshold", mode="max", patience=12),
        ],
    )
    trainer.fit(model, datamodule=data)
    best_model = ChurnLightningModule.load_from_checkpoint(checkpoint.best_model_path)
    trainer.validate(best_model, datamodule=data, verbose=False)
    trainer.test(best_model, datamodule=data)
    print(f"Mejor checkpoint: {checkpoint.best_model_path}")
    print(f"Umbral seleccionado: {best_model.threshold:.6f}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
