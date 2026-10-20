# src/train.py
"""
Entrenamiento y experimentación con la red neuronal de churn.

Rejilla de 10 experimentos con ABLACIÓN (una variable por vez), 3 semillas
por experimento, early stopping, weight decay, dropout y pos_weight.

Uso:
    python -m src.train
"""

from __future__ import annotations

import copy
import json
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
    roc_auc_score,
    confusion_matrix,
)

from src.preprocessing import ChurnDataModule, load_clean
from src.model import ChurnNet


# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS   = 300       # tope duro (early stopping decidirá antes)
PATIENCE     = 8         # épocas sin mejora antes de parar
WEIGHT_DECAY = 1e-4      # regularización L2
SEEDS        = [0, 1, 2] # 3 semillas → media ± std
POS_WEIGHT   = None      # se rellena en run_all_experiments

RESULTS_FIGURES = Path("resultados/figures")
RESULTS_TABLES  = Path("resultados/tables")
RESULTS_MODELS  = Path("resultados/models")
RESULTS_JSON    = Path("resultados/tables/resultados_por_semilla.json")


# ---------------------------------------------------------------------------
# Rejilla de experimentos (ABLACIÓN: una variable por vez)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # ============ PROFUNDIDAD CRECIENTE ============
    {"name": "A",  "hidden_sizes": [32],              "batch_size": 32, "lr": 1e-3, "dropout": 0.0},
    {"name": "B",  "hidden_sizes": [64, 32],          "batch_size": 32, "lr": 1e-3, "dropout": 0.0},
    {"name": "C",  "hidden_sizes": [128, 64, 32],     "batch_size": 32, "lr": 1e-3, "dropout": 0.0},
    {"name": "E",  "hidden_sizes": [128, 64, 32, 16], "batch_size": 32, "lr": 1e-3, "dropout": 0.3},

    # ============ ANCHURA (misma profundidad 3, distinta anchura) ============
    {"name": "F1", "hidden_sizes": [32, 32, 32],      "batch_size": 32, "lr": 1e-3, "dropout": 0.0},
    {"name": "F2", "hidden_sizes": [64, 64, 64],      "batch_size": 32, "lr": 1e-3, "dropout": 0.0},
    {"name": "F3", "hidden_sizes": [128, 128, 128],   "batch_size": 32, "lr": 1e-3, "dropout": 0.3},

    # ============ BATCH AISLADO (B con batch=64) ============
    {"name": "G",  "hidden_sizes": [64, 32],          "batch_size": 64, "lr": 1e-3, "dropout": 0.0},

    # ============ LR AISLADO (B con lr más bajo) ============
    {"name": "H1", "hidden_sizes": [64, 32],          "batch_size": 32, "lr": 5e-4, "dropout": 0.0},
    {"name": "H2", "hidden_sizes": [64, 32],          "batch_size": 32, "lr": 1e-4, "dropout": 0.0},
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Fija TODAS las fuentes de aleatoriedad (Python, NumPy, PyTorch, CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    losses, preds, targets = [], [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE); y_batch = y_batch.to(DEVICE)
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
def evaluate(model, loader, criterion, threshold: float = 0.5):
    """Evalúa y devuelve (loss, acc, preds, targets, probs)."""
    model.eval()
    losses, probs, targets = [], [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE); y_batch = y_batch.to(DEVICE)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        losses.append(loss.item())
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        targets.extend(y_batch.cpu().numpy())

    probs   = np.array(probs).flatten()
    targets = np.array(targets).flatten()
    preds   = (probs > threshold).astype(int)
    return float(np.mean(losses)), accuracy_score(targets, preds), preds, targets, probs


def train_model(model, train_loader, val_loader, lr: float):
    """
    Entrena con EARLY STOPPING y devuelve:
      (history, stopped_epoch, best_val_loss, best_train_loss)
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY,
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state    = copy.deepcopy(model.state_dict())
    best_epoch    = 0          # índice 0-based de la mejor época
    epochs_no_improve = 0
    stopped_epoch = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_loss < best_val_loss - 1e-4:
            best_val_loss = vl_loss
            best_epoch    = epoch - 1   # 0-based
            best_state    = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{MAX_EPOCHS} | "
                  f"Train {tr_loss:.4f} | Val {vl_loss:.4f} | "
                  f"TrAcc {tr_acc:.4f} | VlAcc {vl_acc:.4f}")

        if epochs_no_improve >= PATIENCE:
            print(f"  Early stopping en época {epoch} "
                  f"(sin mejora en {PATIENCE}). Best val_loss={best_val_loss:.4f}")
            stopped_epoch = epoch
            break

    model.load_state_dict(best_state)
    best_train_loss = history["train_loss"][best_epoch]
    return history, stopped_epoch, best_val_loss, best_train_loss


# ---------------------------------------------------------------------------
# Helpers de guardado y agregación
# ---------------------------------------------------------------------------
def _save_artifacts(name, history, preds, targets, model):
    # Curva de pérdida
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"],   label="Val")
    plt.xlabel("Época"); plt.ylabel("Loss")
    plt.title(f"Curva de pérdida — Modelo {name}")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_FIGURES / f"loss_modelo_{name}.png"); plt.close()

    # Matriz de confusión
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicción"); plt.ylabel("Real")
    plt.title(f"Matriz de confusión — Modelo {name}")
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES / f"confusion_matrix_{name}.png"); plt.close()

    torch.save(model.state_dict(), RESULTS_MODELS / f"modelo_{name}.pth")


def _aggregate(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Métricas que se agregan con media ± std entre semillas
    metric_cols = [
        "Train_Loss_final", "Val_Loss_final", "Test_Loss",
        "Accuracy", "Precision", "Recall", "F1", "AUC",
    ]
    agg = df_raw.groupby("Modelo")[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]

    meta = df_raw.groupby("Modelo")[[
        "Arquitectura", "Capas_ocultas", "Neuronas_totales",
        "Parámetros", "Batch", "LR", "Dropout", "Epocas_reales",
    ]].first().reset_index()

    cols_order = [
        "Modelo", "Arquitectura", "Capas_ocultas", "Neuronas_totales",
        "Parámetros", "Batch", "LR", "Dropout", "Epocas_reales",
        "Train_Loss_final_mean", "Train_Loss_final_std",
        "Val_Loss_final_mean",   "Val_Loss_final_std",
        "Test_Loss_mean",        "Test_Loss_std",
        "Accuracy_mean",  "Accuracy_std",
        "Precision_mean", "Precision_std",
        "Recall_mean",    "Recall_std",
        "F1_mean",        "F1_std",
        "AUC_mean",       "AUC_std",
    ]
    return meta.merge(agg, on="Modelo")[cols_order].round(4)


def _plot_comparison(df_agg: pd.DataFrame):
    # Compara Accuracy, Precision, Recall, F1, AUC (sin losses)
    metrics = ["Accuracy_mean", "Precision_mean", "Recall_mean", "F1_mean", "AUC_mean"]
    plot_df = df_agg.set_index("Modelo")[metrics]
    plot_df.columns = [c.replace("_mean", "") for c in plot_df.columns]

    yerr = df_agg.set_index("Modelo")[[
        "Accuracy_std", "Precision_std", "Recall_std", "F1_std", "AUC_std"]]
    yerr.columns = [c.replace("_std", "") for c in yerr.columns]

    ax = plot_df.plot(kind="bar", figsize=(13, 5), yerr=yerr, capsize=3)
    ax.set_title("Comparación de métricas por modelo (media ± std, 3 semillas)")
    ax.set_ylabel("Valor"); ax.set_ylim(0, 1.0)
    ax.set_xticklabels(plot_df.index, rotation=45, ha="right")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES / "comparacion_modelos.png"); plt.close()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def run_all_experiments() -> pd.DataFrame:
    for p in (RESULTS_FIGURES, RESULTS_TABLES, RESULTS_MODELS):
        p.mkdir(parents=True, exist_ok=True)

    print(f"Dispositivo: {DEVICE}")
    print(f"Semillas: {SEEDS} | Max épocas: {MAX_EPOCHS} | Patience: {PATIENCE}")
    print(f"Weight decay: {WEIGHT_DECAY}\n")

    df = load_clean()

    # pos_weight calculado UNA VEZ con class_counts del train
    global POS_WEIGHT
    dm_probe = ChurnDataModule(df, batch_size=32); dm_probe.setup()
    n_neg, n_pos = dm_probe.class_counts
    POS_WEIGHT = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)
    print(f"pos_weight = {POS_WEIGHT.item():.3f}  (n_neg={n_neg}, n_pos={n_pos})\n")

    raw_results = []

    for exp in EXPERIMENTS:
        print("=" * 62)
        print(f"EXPERIMENTO {exp['name']} | hidden={exp['hidden_sizes']} "
              f"| batch={exp['batch_size']} | lr={exp['lr']} | dropout={exp['dropout']}")
        print("=" * 62)

        for seed in SEEDS:
            set_seed(seed)
            dm = ChurnDataModule(df, batch_size=exp["batch_size"]); dm.setup()

            model = ChurnNet(
                input_dim=dm.input_dim,
                hidden_sizes=exp["hidden_sizes"],
                dropout=exp["dropout"],
            ).to(DEVICE)
            n_params = sum(p.numel() for p in model.parameters())

            print(f"\n  seed={seed} | parámetros={n_params}")
            history, stopped, best_val, best_train = train_model(
                model, dm.train_dataloader(), dm.val_dataloader(), lr=exp["lr"])

            criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
            test_loss, _, preds, targets, probs = evaluate(
                model, dm.test_dataloader(), criterion, threshold=0.5)

            metrics = {
                "Modelo": exp["name"], "Seed": seed,
                "Arquitectura": str(exp["hidden_sizes"]),
                "Capas_ocultas": len(exp["hidden_sizes"]),
                "Neuronas_totales": sum(exp["hidden_sizes"]),
                "Parámetros": n_params,
                "Batch": exp["batch_size"], "LR": exp["lr"], "Dropout": exp["dropout"],
                "Epocas_reales": stopped,
                "Train_Loss_final": round(best_train, 4),
                "Val_Loss_final":   round(best_val, 4),
                "Test_Loss":        round(test_loss, 4),
                "Accuracy":  round(accuracy_score(targets, preds), 4),
                "Precision": round(precision_score(targets, preds), 4),
                "Recall":    round(recall_score(targets, preds), 4),
                "F1":        round(f1_score(targets, preds), 4),
                "AUC":       round(roc_auc_score(targets, probs), 4),
            }
            raw_results.append(metrics)
            print(f"  seed={seed} → Acc {metrics['Accuracy']:.4f} | "
                  f"Prec {metrics['Precision']:.4f} | "
                  f"Rec {metrics['Recall']:.4f} | F1 {metrics['F1']:.4f} | "
                  f"AUC {metrics['AUC']:.4f} | épocas={stopped}")

            if seed == SEEDS[0]:
                _save_artifacts(exp["name"], history, preds, targets, model)

    # CSV crudo (30 filas = 10 exp × 3 semillas)
    df_raw = pd.DataFrame(raw_results)
    df_raw.to_csv(RESULTS_TABLES / "resultados_por_semilla.csv", index=False)
    RESULTS_JSON.write_text(json.dumps(raw_results, indent=2, default=str))

    # CSV agregado (10 filas, con _mean y _std)
    df_agg = _aggregate(df_raw)
    df_agg.to_csv(RESULTS_TABLES / "comparacion_modelos.csv", index=False)

    print("\n" + "=" * 62)
    print("TABLA AGREGADA (media ± std entre semillas)")
    print("=" * 62)
    print(df_agg.to_string(index=False))

    _plot_comparison(df_agg)
    print(f"\n✅ Tabla: {RESULTS_TABLES / 'comparacion_modelos.csv'}")
    print(f"✅ Gráfico: {RESULTS_FIGURES / 'comparacion_modelos.png'}")
    print(f"✅ Crudos: {RESULTS_TABLES / 'resultados_por_semilla.csv'}")
    return df_agg


if __name__ == "__main__":
    run_all_experiments()