"""
experiments/generate_connect4_testset.py

Genera un test set offline per Connect Four (6x7) usando MinMax
a profondità limitata con una euristica banale (0 sui nodi non terminali).

Per ogni stato s, calcoliamo:
    z ≈ MinMax_value(s, depth=L_eval, K_eval, H=0)
dal punto di vista di PLAYER1.

Salva un CSV con colonne:
    x0, x1, ..., x41, z
dove x* è la board flattenata (encoding usato dal modello).
"""

from __future__ import annotations
from typing import List, Tuple
import csv
import os
import random

import numpy as np

from game.connect_four import (
    ConnectFourState,
    create_initial_state,
    get_valid_moves,
    apply_move,
    get_game_result,
)
from ai.minimax import minimax_value
from utils.encoding import state_to_np_array


def zero_heuristic(state: ConnectFourState) -> float:
    """
    Euristica banale: 0 sui nodi non terminali.
    MinMax vedrà solo vittorie/sconfitte “vicine” alla foglia esplorata.
    """
    return 0.0


def sample_random_state(max_random_moves: int = 20) -> ConnectFourState:
    """
    Esegue al massimo max_random_moves mosse random a partire dallo
    stato iniziale e ritorna lo stato corrente (terminal o no).
    """
    state = create_initial_state()
    for _ in range(max_random_moves):
        if get_game_result(state) is not None:
            break
        moves = get_valid_moves(state)
        if not moves:
            break
        m = random.choice(moves)
        state = apply_move(state, m)
    return state


def generate_test_states(
    n_states: int = 200,
    max_random_moves: int = 20,
    depth_eval: int = 4,
    width_eval: int | None = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera n_states stati casuali di Connect Four e ne approssima
    il valore con MinMax a profondità limitata.

    Ritorna:
        X: shape (n_states, 42)  board flattenata (encoding standard)
        y: shape (n_states,)     valori z in [-1, +1] circa
    """
    random.seed(seed)
    np.random.seed(seed)

    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    while len(X_list) < n_states:
        s = sample_random_state(max_random_moves=max_random_moves)
        x = state_to_np_array(s)
        v = minimax_value(s, depth=depth_eval, K=width_eval, H=zero_heuristic)
        X_list.append(x.astype(np.float32))
        y_list.append(float(v))

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


def save_testset_to_csv(
    X: np.ndarray,
    y: np.ndarray,
    filepath: str = "results/connect4_testset_L4.csv",
):
    """
    Salva il test set in un CSV con colonne x0..x41, z.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    n, d = X.shape
    fieldnames = [f"x{i}" for i in range(d)] + ["z"]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, z_val in zip(X, y):
            row_dict = {f"x{i}": int(row[i]) for i in range(d)}
            row_dict["z"] = float(z_val)
            writer.writerow(row_dict)

    print(f"Saved Connect4 test set to {filepath} (n={n})")


if __name__ == "__main__":
    X, y = generate_test_states(
        n_states=200,
        max_random_moves=20,
        depth_eval=4,
        width_eval=None,
        seed=0,
    )
    save_testset_to_csv(X, y, filepath="results/connect4_testset_L4.csv")
