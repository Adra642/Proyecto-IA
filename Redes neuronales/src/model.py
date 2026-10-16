# src/model.py
"""
Modelo de red neuronal para clasificación binaria de churn bancario.

Este módulo define una única clase `ChurnNet` que se puede configurar
con distintas arquitecturas (número de capas ocultas, neuronas, dropout).
Así, los 4 experimentos se pueden hacer con la misma clase, cambiando
solo los argumentos del constructor.

Contrato con el resto del proyecto:
-----------------------------------
- Entrada: tensores de shape (batch_size, input_dim).
  `input_dim` se obtiene de `ChurnDataModule.input_dim`.
- Salida: tensores de shape (batch_size, 1) con logits (sin sigmoide).
  Compatible con `nn.BCEWithLogitsLoss`.
"""

import torch
import torch.nn as nn


class ChurnNet(nn.Module):
    """
    Red neuronal feed-forward para clasificación binaria.

    Parameters
    ----------
    input_dim : int
        Número de features de entrada (lo da `ChurnDataModule.input_dim`).
    hidden_sizes : list[int]
        Neuronas de cada capa oculta. Ejemplos:
            - [32]           → 1 capa oculta
            - [64, 32]       → 2 capas ocultas
            - [128, 64, 32]  → 3 capas ocultas
    dropout : float
        Probabilidad de dropout entre capas (0.0 = sin dropout).
        En este proyecto no lo usamos, pero se deja preparado.

    Notas
    -----
    - La capa de salida tiene 1 neurona porque es clasificación binaria.
    - NO se aplica sigmoide en forward: BCEWithLogitsLoss ya la incluye.
    """

    def __init__(self, input_dim: int, hidden_sizes: list = None, dropout: float = 0.0):
        super().__init__()

        # Lista por defecto si no se pasa ninguna
        if hidden_sizes is None:
            hidden_sizes = [32]

        # Construimos las capas dinámicamente
        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Capa de salida: 1 neurona, sin activación
        layers.append(nn.Linear(prev_size, 1))

        # `*layers` desempaca la lista y los pasa como argumentos a Sequential
        self.modelo = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Shape (batch_size, 1). Logits sin sigmoide.
        """
        return self.modelo(x)