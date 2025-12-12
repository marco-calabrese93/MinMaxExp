"""
experiments/eval_vs_random_connect4.py

Valutazione online: il modello gioca contro un avversario random
usando MinMax(L, K, H_model).

Esegui dalla root con:
    python -m experiments.eval_vs_random_connect4
"""

import random
import numpy as np
import torch
import torch.nn as nn

from ai.minimax import minimax_decision
from utils.encoding import state_to_np_array
from game.connect_four import (
    create_initial_state,
    get_valid_moves,
    apply_move,
    get_game_result,
    PLAYER1,
    PLAYER2,
)

# ---------------------------------------------------------------------
# MLP con la STESSA architettura degli script di training (hidden_dim=128)
# ---------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # output ∈ [-1,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------
# Costruzione agente MinMax con euristica data dal modello
# ---------------------------------------------------------------------

def make_minimax_agent(model: MLP,
                       device: str = "cpu",
                       L: int = 2,
                       K: int | None = 4):
    """
    Costruisce un agente che, dato uno stato, sceglie la mossa con MinMax
    usando come H(s) il valore restituito dal modello MLP.
    """

    def H(state):
        x_np = state_to_np_array(state)
        x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return model(x).item()

    def agent(state):
        move, _ = minimax_decision(state, L=L, K=K, H=H)
        return move

    return agent


def random_agent(state):
    """Agente completamente random: sceglie una mossa valida a caso."""
    moves = get_valid_moves(state)
    if not moves:
        return None
    return random.choice(moves)


# ---------------------------------------------------------------------
# Simulazione di una partita
# ---------------------------------------------------------------------

def play_single_game(agent_p1, agent_p2):
    """
    Gioca UNA partita:
      - agent_p1 gioca come PLAYER1
      - agent_p2 gioca come PLAYER2

    Restituisce z ∈ {-1, 0, +1} dal punto di vista di PLAYER1.
    """
    state = create_initial_state()

    while True:
        res = get_game_result(state)
        if res is not None:
            return res  # già dal punto di vista di PLAYER1

        if state.current_player == PLAYER1:
            move = agent_p1(state)
        else:
            move = agent_p2(state)

        if move is None:
            # nessuna mossa valida → consideriamo pareggio
            return 0

        state = apply_move(state, move)


# ---------------------------------------------------------------------
# Valutazione vs random
# ---------------------------------------------------------------------

def evaluate_vs_random(checkpoint_path: str,
                       L: int,
                       K: int | None,
                       n_games: int = 100,
                       device: str = "cpu"):
    """
    Carica un modello da checkpoint_path e lo fa giocare come PLAYER1
    contro un avversario random per n_games partite.

    Stampa e restituisce (wins, draws, losses).
    """
    rows, cols = 6, 7
    input_dim = rows * cols

    # ATTENZIONE: usiamo la stessa architettura MLP usata negli script di training
    model = MLP(input_dim=input_dim, hidden_dim=128)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    agent_p1 = make_minimax_agent(model, device=device, L=L, K=K)
    agent_p2 = random_agent

    wins = draws = losses = 0

    for _ in range(n_games):
        z = play_single_game(agent_p1, agent_p2)
        if z > 0:
            wins += 1
        elif z < 0:
            losses += 1
        else:
            draws += 1

    print(f"Valutazione vs random ({checkpoint_path}):")
    print(f"  L={L}, K={K}, games={n_games}")
    print(f"  W/D/L = {wins}/{draws}/{losses}")

    return wins, draws, losses


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    device = "cpu"

    print("=== Valutazione vs random: Strategy A (baseline) ===")
    try:
        evaluate_vs_random("results/model_strategy_A.pth",
                           L=2, K=4, n_games=50, device=device)
    except FileNotFoundError:
        print("  [SKIP] results/model_strategy_A.pth non trovato")

    print("\n=== Valutazione vs random: Strategy B (curriculum v1) ===")
    try:
        evaluate_vs_random("results/model_strategy_B.pth",
                           L=3, K=6, n_games=50, device=device)
    except FileNotFoundError:
        print("  [SKIP] results/model_strategy_B.pth non trovato")

    print("\n=== Valutazione vs random: Strategy C (curriculum v2) ===")
    try:
        evaluate_vs_random("results/model_strategy_C.pth",
                           L=3, K=4, n_games=50, device=device)
    except FileNotFoundError:
        print("  [SKIP] results/model_strategy_C.pth non trovato")
