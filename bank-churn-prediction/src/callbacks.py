"""Callbacks de entrenamiento del clasificador de churn."""

import torch
from sklearn.metrics import precision_recall_curve
from src.lightning_config import configure_warnings

configure_warnings()

import pytorch_lightning as pl


class F1ThresholdCallback(pl.Callback):
    """Selecciona el umbral de validación que maximiza F1."""

    def __init__(self, min_threshold: float = 0.0, max_threshold: float = 1.0):
        if not 0 <= min_threshold < max_threshold <= 1:
            raise ValueError("los límites deben cumplir 0 <= mínimo < máximo <= 1")
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def on_validation_epoch_end(self, trainer, pl_module):
        if not pl_module._val_logits:
            return

        probabilities = torch.cat(
            [torch.sigmoid(logits) for logits in pl_module._val_logits]
        ).numpy().ravel()
        targets = torch.cat(pl_module._val_targets).numpy().astype(int).ravel()
        precision, recall, thresholds = precision_recall_curve(targets, probabilities)
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        valid = (thresholds >= self.min_threshold) & (thresholds <= self.max_threshold)
        if not valid.any():
            valid = thresholds >= 0
        best_index = int(torch.as_tensor(f1[valid]).argmax())
        valid_thresholds = thresholds[valid]
        pl_module.threshold = float(valid_thresholds[best_index])
        pl_module.log("val_optimal_threshold", pl_module.threshold, prog_bar=True)
        pl_module.log("val_f1_at_threshold", float(f1[valid][best_index]), prog_bar=True)
