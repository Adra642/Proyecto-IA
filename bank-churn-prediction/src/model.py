"""Modelo Lightning para clasificar el abandono bancario desbalanceado."""

import torch
import torch.nn as nn
from src.lightning_config import configure_warnings

configure_warnings()

import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class FocalLoss(nn.Module):
    """Pérdida focal binaria aplicada directamente a logits."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma debe ser no negativo")
        if not 0 < alpha <= 1:
            raise ValueError("alpha debe estar en el intervalo (0, 1]")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().reshape_as(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t).pow(self.gamma) * bce).mean()


class TabularResidualBlock(nn.Module):
    """Bloque lineal con normalización, regularización y conexión residual."""

    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.activation = nn.Mish()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class ChurnLightningModule(pl.LightningModule):
    """Clasificador residual que recibe las variables ya codificadas."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.25,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_name: str = "focal",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        pos_weight: float | None = None,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks debe ser como mínimo 1")
        self.save_hyperparameters()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
        )
        self.blocks = nn.Sequential(
            *(TabularResidualBlock(hidden_dim, dropout) for _ in range(num_blocks))
        )
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.loss_fn = self._build_loss()
        self.threshold = 0.5
        self._val_logits: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

        metrics = {
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
            "auroc": BinaryAUROC(),
        }
        self.val_metrics = nn.ModuleDict(metrics)
        self.test_threshold_metrics = nn.ModuleDict(
            {
                "accuracy": BinaryAccuracy(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "f1": BinaryF1Score(),
            }
        )
        self.test_auroc = BinaryAUROC()

    def _build_loss(self) -> nn.Module:
        if self.hparams.loss_name == "focal":
            return FocalLoss(self.hparams.focal_gamma, self.hparams.focal_alpha)
        if self.hparams.loss_name == "weighted_bce":
            weight = None
            if self.hparams.pos_weight is not None:
                weight = torch.tensor(float(self.hparams.pos_weight))
            return nn.BCEWithLogitsLoss(pos_weight=weight)
        raise ValueError("loss_name debe ser 'focal' o 'weighted_bce'")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Devuelve un logit por cada fila de entrada."""
        return self.output_layer(self.blocks(self.input_layer(inputs)))

    def _shared_step(self, batch):
        inputs, targets = batch
        logits = self(inputs)
        return self.loss_fn(logits, targets), logits, targets.reshape_as(logits)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_logits.clear()
        self._val_targets.clear()

    def validation_step(self, batch, batch_idx):
        loss, logits, targets = self._shared_step(batch)
        probabilities = torch.sigmoid(logits.detach())
        self._val_logits.append(logits.detach().cpu())
        self._val_targets.append(targets.detach().cpu())
        for metric in self.val_metrics.values():
            metric.update(probabilities, targets.int())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return
        probabilities = torch.cat([torch.sigmoid(x) for x in self._val_logits])
        targets = torch.cat(self._val_targets).int()
        tuned_predictions = (probabilities >= self.threshold).int()
        tuned_f1 = (2 * (tuned_predictions * targets).sum()).float() / (
            tuned_predictions.sum() + targets.sum() + 1e-8
        )
        self.log("val_f1_current_threshold", tuned_f1, prog_bar=True)
        for name, metric in self.val_metrics.items():
            self.log(f"val_{name}", metric.compute(), prog_bar=name in {"f1", "auroc"})
            metric.reset()

    def test_step(self, batch, batch_idx):
        loss, logits, targets = self._shared_step(batch)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= self.threshold).int()
        for metric in self.test_threshold_metrics.values():
            metric.update(predictions, targets.int())
        self.test_auroc.update(probabilities, targets.int())
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_threshold", self.threshold)
        return loss

    def on_test_epoch_end(self):
        for name, metric in self.test_threshold_metrics.items():
            self.log(f"test_{name}", metric.compute())
            metric.reset()
        self.log("test_auroc", self.test_auroc.compute())
        self.test_auroc.reset()

    def configure_optimizers(self):
        """Configura AdamW con la tasa de aprendizaje y regularización dadas."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )