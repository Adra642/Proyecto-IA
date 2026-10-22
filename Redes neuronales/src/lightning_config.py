"""Configuración de compatibilidad cargada antes de PyTorch Lightning."""

import warnings


def configure_warnings() -> None:
    """Oculta únicamente la advertencia conocida de compatibilidad de Lightning."""
    warnings.filterwarnings(
        "ignore",
        message=r".*is deprecated, use isinstance\(treespec, TreeSpec\).*",
        category=DeprecationWarning,
        module=r"pytorch_lightning\.utilities\._pytree",
    )
