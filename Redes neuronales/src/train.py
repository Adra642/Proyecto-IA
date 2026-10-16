# src/train.py
"""
Entrenamiento y experimentación con la red neuronal de churn.

Ejecuta 4 experimentos cambiando arquitectura, batch size y learning rate,
y guarda todos los artefactos (gráficos, métricas, modelos) en `resultados/`.

Uso:
    python -m src.train
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.preprocessing import ChurnDataModule, load_clean, RANDOM_STATE
from src.model import ChurnNet


# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50

RESULTS_FIGURES = Path("resultados/figures")
RESULTS_TABLES = Path("resultados/tables")
RESULTS_MODELS = Path("resultados/models")

# Los 4 experimentos. Cada uno cambia algo distinto para demostrar
# manejo de hiperparámetros tal y como pide el enunciado.
EXPERIMENTS = [
    {"name": "A", "hidden_sizes": [32],          "batch_size": 32, "lr": 0.001},
    {"name": "B", "hidden_sizes": [64, 32],      "batch_size": 32, "lr": 0.001},
    {"name": "C", "hidden_sizes": [128, 64, 32], "batch_size": 32, "lr": 0.001},
    {"name": "D", "hidden_sizes": [128, 64, 32], "batch_size": 64, "lr": 0.0005},
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def set_seed(seed: int = RANDOM_STATE) -> None:
    """Fija la semilla para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    """Una época de entrenamiento. Devuelve (loss media, accuracy)."""
    model.train()
    losses, preds, targets = [], [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend((torch.sigmoid(logits) > 0.5).float().cpu().numpy())
        targets.extend(y_batch.cpu().numpy())

    return float(np.mean(losses)), accuracy_score(targets, preds)


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evalúa el modelo. Devuelve (loss, accuracy, preds, targets)."""
    model.eval()
    losses, preds, targets = [], [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        losses.append(loss.item())
        preds.extend((torch.sigmoid(logits) > 0.5).float().cpu().numpy())
        targets.extend(y_batch.cpu().numpy())

    return (
        float(np.mean(losses)),
        accuracy_score(targets, preds),
        np.array(preds).flatten(),
        np.array(targets).flatten(),
    )


def train_model(model, train_loader, val_loader, epochs, lr):
    """Entrena el modelo y devuelve el historial de pérdidas y accuracy."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>3}/{epochs} | "
                f"Train Loss {tr_loss:.4f} | Val Loss {vl_loss:.4f} | "
                f"Train Acc {tr_acc:.4f} | Val Acc {vl_acc:.4f}"
            )

    return history


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------
def plot_loss_curves(history, name, path):
    """Curva de pérdida train/val para un modelo."""
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title(f"Curva de pérdida - Modelo {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_confusion_matrix(cm, name, path):
    """Matriz de confusión para un modelo."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusión - Modelo {name}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def run_all_experiments():
    """Ejecuta los 4 experimentos y guarda todos los resultados."""
    set_seed()

    # Crear carpetas si no existen
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_MODELS.mkdir(parents=True, exist_ok=True)

    print(f"Dispositivo: {DEVICE}")
    print(f"Épocas por experimento: {EPOCHS}\n")

    # Cargar datos limpios UNA SOLA VEZ
    df = load_clean()

    results = []

    for exp in EXPERIMENTS:
        name = exp["name"]
        print("=" * 60)
        print(f"EXPERIMENTO {name}")
        print(f"  hidden_sizes  = {exp['hidden_sizes']}")
        print(f"  batch_size    = {exp['batch_size']}")
        print(f"  learning_rate = {exp['lr']}")
        print("=" * 60)

        # Reproducibilidad dentro del experimento
        set_seed()

        # DataModule específico con el batch_size del experimento
        dm = ChurnDataModule(df, batch_size=exp["batch_size"])
        dm.setup()

        # Modelo
        model = ChurnNet(
            input_dim=dm.input_dim,
            hidden_sizes=exp["hidden_sizes"],
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  input_dim = {dm.input_dim} | parámetros = {n_params}")

        # Entrenamiento
        history = train_model(
            model,
            dm.train_dataloader(),
            dm.val_dataloader(),
            epochs=EPOCHS,
            lr=exp["lr"],
        )

        # Evaluación final en test
        criterion = nn.BCEWithLogitsLoss()
        test_loss, test_acc, preds, targets = evaluate(
            model, dm.test_dataloader(), criterion
        )

        # Métricas
        metrics = {
            "Modelo": name,
            "Arquitectura": str(exp["hidden_sizes"]),
            "Parámetros": n_params,
            "Batch": exp["batch_size"],
            "LR": exp["lr"],
            "Test Loss": round(test_loss, 4),
            "Accuracy": round(accuracy_score(targets, preds), 4),
            "Precision": round(precision_score(targets, preds), 4),
            "Recall": round(recall_score(targets, preds), 4),
            "F1": round(f1_score(targets, preds), 4),
        }
        results.append(metrics)

        print(
            f"\n  TEST → Loss {test_loss:.4f} | "
            f"Acc {metrics['Accuracy']} | Prec {metrics['Precision']} | "
            f"Rec {metrics['Recall']} | F1 {metrics['F1']}"
        )
        print(classification_report(targets, preds, target_names=["No Churn", "Churn"]))

        # Gráficos
        plot_loss_curves(history, name, RESULTS_FIGURES / f"loss_modelo_{name}.png")
        cm = confusion_matrix(targets, preds)
        plot_confusion_matrix(cm, name, RESULTS_FIGURES / f"confusion_matrix_{name}.png")

        # Guardar modelo
        torch.save(model.state_dict(), RESULTS_MODELS / f"modelo_{name}.pth")

        print(f"  ✅ Gráficos y modelo guardados.\n")

    # Tabla comparativa
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_TABLES / "comparacion_modelos.csv", index=False)

    print("=" * 60)
    print("TABLA COMPARATIVA FINAL")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print(f"\n✅ Guardada en {RESULTS_TABLES / 'comparacion_modelos.csv'}")

    # Gráfico comparativo de barras
    plt.figure(figsize=(10, 5))
    df_results.set_index("Modelo")[["Accuracy", "Precision", "Recall", "F1"]].plot(
        kind="bar", ax=plt.gca()
    )
    plt.title("Comparación de métricas por modelo")
    plt.ylabel("Valor")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES / "comparacion_modelos.png")
    plt.close()
    print(f"✅ Gráfico comparativo guardado en {RESULTS_FIGURES / 'comparacion_modelos.png'}")

    return df_results


if __name__ == "__main__":
    run_all_experiments()