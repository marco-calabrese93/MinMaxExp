"""
experiments/eval_offline_connect4.py

Offline evaluation del modello di Connect Four:
- genera un insieme di stati "non banali" con partite random corte
- approssima il valore "vero" con un MinMax più profondo (senza euristica)
- calcola MSE e correlazione Pearson tra modello e MinMax

Esegui dalla root del progetto con:
    python -m experiments.eval_offline_connect4
"""

import numpy as np
import torch

from ai.model import ConnectFourNet, HTrueWrapper  # HTrueWrapper non serve qui ma resta coerente con il resto del progetto
from ai.minimax import minimax_decision
from utils.encoding import state_to_np_array
from game.connect_four import (
    create_initial_state,
    apply_move,
    get_game_result,
    get_valid_moves,
)


def generate_test_states(n: int = 200,
                         max_random_moves: int = 12,
                         seed: int = 0):
    """
    Genera n stati di Connect Four "non banali" partendo dalla posizione iniziale
    e facendo un numero random di mosse casuali (fino a max_random_moves).

    Per ogni stato:
      - non si prosegue se la partita termina prima
      - lo stato finale viene aggiunto alla lista

    Restituisce:
      - lista di stati
      - array numpy con i valori "veri" approssimati via MinMax profondo
    """
    rng = np.random.RandomState(seed)
    states = []
    true_vals = []

    for _ in range(n):
        s = create_initial_state()

        # playout random corto per evitare solo board vuote
        num_moves = rng.randint(1, max_random_moves + 1)
        for _ in range(num_moves):
            moves = get_valid_moves(s)
            if not moves:
                break
            m = int(rng.choice(moves))
            s = apply_move(s, m)
            if get_game_result(s) is not None:
                break

        # valore "vero" approssimato con MinMax più profondo senza euristica
        _, v_true = minimax_decision(
            s,
            L=4,          # profondità più alta del solito training
            K=None,       # esplora tutte le mosse
            H=lambda st: 0.0,  # nessuna euristica, pura ricerca
        )

        states.append(s)
        true_vals.append(v_true)

    return states, np.array(true_vals, dtype=float)


def evaluate_model(model: ConnectFourNet,
                   device: str = "cpu",
                   n_states: int = 200,
                   seed: int = 0):
    """
    Valuta il modello in modo offline su un insieme di stati.

    Restituisce:
      - mse  : mean squared error tra predizioni del modello e valori MinMax
      - corr : correlazione di Pearson (0.0 se non definita)
    """
    model.to(device)
    model.eval()

    # 🔧 FIX: passiamo n=n_states invece di n_states=...
    states, true_vals = generate_test_states(
        n=n_states,
        max_random_moves=12,
        seed=seed
    )

    preds = []
    with torch.no_grad():
        for s in states:
            x_np = state_to_np_array(s)
            x = torch.tensor(
                x_np,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)
            z_pred = model(x).item()
            preds.append(z_pred)

    preds = np.array(preds, dtype=float)

    mse = float(np.mean((preds - true_vals) ** 2))

    # Gestione caso degenerato: se una delle due serie ha varianza ~0
    if np.std(preds) == 0.0 or np.std(true_vals) == 0.0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(preds, true_vals)[0, 1])

    print("Offline evaluation:")
    print("  MSE:", mse)
    print("  correlation:", corr)

    return mse, corr


if __name__ == "__main__":
    # Esempio: valuta un modello (non allenato) o carica un checkpoint
    rows, cols = 6, 7
    input_dim = rows * cols

    model = ConnectFourNet(input_dim=input_dim)

    # Se hai un modello allenato, decommenta e aggiorna il path:
    # state_dict = torch.load("results/model_strategy_B.pth", map_location="cpu")
    # model.load_state_dict(state_dict)

    evaluate_model(model, device="cpu", n_states=200, seed=0)
