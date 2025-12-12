"""
experiments/tictactoe_train_mlp.py

Allena un MLP per approssimare l'esito perfetto del Tris (Tic-Tac-Toe),
utilizzando il dataset generato da tictactoe_env_and_oracle.py.

Input:  9 valori {-1, 0, +1} (board flatten)
Target: z ∈ {-1, 0, +1} (risultato perfetto dal punto di vista di PLAYER1)

Esegui dalla root del progetto con:
    python -m experiments.tictactoe_train_mlp
"""

import csv
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 0. Utility: seed per riproducibilità
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 1. Caricamento dataset CSV
# ============================================================

def load_tictactoe_dataset_csv(
    filepath: str = "results/tictactoe_dataset_player1.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carica il dataset salvato da tictactoe_env_and_oracle.py.

    CSV con colonne:
        x0, x1, ..., x8, z

    Ritorna:
        X: array (N, 9)
        y: array (N,)
    """
    X: List[List[float]] = []
    y: List[float] = []

    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_vec = [float(row[f"x{i}"]) for i in range(9)]
            z_val = float(row["z"])
            X.append(x_vec)
            y.append(z_val)

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    print(f"Caricati {X_arr.shape[0]} esempi da {filepath}")
    return X_arr, y_arr


# ============================================================
# 2. MLP per H_true (versione Tris)
# ============================================================

class TicTacToeMLP(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # output in [-1,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 3. Split train/test
# ============================================================

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Divide il dataset in train e test.
    """
    set_seed(seed)
    N = X.shape[0]
    indices = list(range(N))
    random.shuffle(indices)

    split = int(N * (1.0 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y[test_idx], dtype=torch.float32).unsqueeze(1)

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test


# ============================================================
# 4. Metriche: MSE + accuracy sul segno
# ============================================================

def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(((pred - target) ** 2).mean().item())


def compute_sign_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Converte l'output continuo in tre classi:
        -1 (sconfitta), 0 (pareggio), +1 (vittoria)
    e calcola la percentuale di predizioni corrette.

    Strategia:
        se |pred| < 0.33 → 0
        se pred >= 0.33 → +1
        se pred <= -0.33 → -1
    """
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    target_np = target.detach().cpu().numpy().reshape(-1)

    pred_label = []
    for v in pred_np:
        if abs(v) < 0.33:
            pred_label.append(0.0)
        elif v >= 0.33:
            pred_label.append(1.0)
        else:
            pred_label.append(-1.0)
    pred_label = np.asarray(pred_label, dtype=np.float32)

    acc = (pred_label == target_np).mean()
    return float(acc)


# ============================================================
# 5. Training MLP su Tris
# ============================================================

def train_tictactoe_mlp(
    dataset_path: str = "results/tictactoe_dataset_player1.csv",
    seed: int = 0,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    epochs: int = 50,
    test_ratio: float = 0.2,
):
    set_seed(seed)

    # Carica dataset
    X, y = load_tictactoe_dataset_csv(dataset_path)

    # Split
    X_train, y_train, X_test, y_test = train_test_split(
        X, y, test_ratio=test_ratio, seed=seed
    )

    # Modello + ottimizzatore
    model = TicTacToeMLP(input_dim=9, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Training loop semplice (batch intero)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred_train = model(X_train)
        loss_train = loss_fn(pred_train, y_train)
        loss_train.backward()
        optimizer.step()

        # Valutazione su train e test
        model.eval()
        with torch.no_grad():
            pred_train = model(X_train)
            pred_test = model(X_test)

            mse_train = compute_mse(pred_train, y_train)
            mse_test = compute_mse(pred_test, y_test)
            acc_train = compute_sign_accuracy(pred_train, y_train)
            acc_test = compute_sign_accuracy(pred_test, y_test)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"[Epoch {epoch:03d}] "
                f"MSE_train={mse_train:.4f} MSE_test={mse_test:.4f} "
                f"ACC_train={acc_train*100:.1f}% ACC_test={acc_test*100:.1f}%"
            )

    # Ritorna modello e metriche finali
    final_metrics = {
        "mse_train": mse_train,
        "mse_test": mse_test,
        "acc_train": acc_train,
        "acc_test": acc_test,
        "num_examples": int(X.shape[0]),
        "hidden_dim": hidden_dim,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
    }
    return model, final_metrics


# ============================================================
# 6. Salvataggio metriche in CSV (per il report)
# ============================================================

def save_metrics_to_csv(metrics: dict, filepath: str = "results/tictactoe_mlp_metrics.csv"):
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Scrive una sola riga (una run)
    fieldnames = list(metrics.keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Metriche salvate in: {filepath}")


# ============================================================
# 7. Entry point
# ============================================================

if __name__ == "__main__":
    model, metrics = train_tictactoe_mlp(
        dataset_path="results/tictactoe_dataset_player1.csv",  # cambia se hai un nome diverso
        seed=0,
        hidden_dim=64,
        lr=1e-3,
        epochs=50,
        test_ratio=0.2,
    )
    save_metrics_to_csv(metrics, filepath="results/tictactoe_mlp_metrics.csv")
    print("Metriche finali:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

