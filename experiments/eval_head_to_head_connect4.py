"""
experiments/eval_head_to_head_connect4.py

Confronto head-to-head fra due modelli di Connect Four:
- model_new.pth
- model_old.pth

Entrambi giocano tramite MinMax(L, K, H_model),
alternando chi fa il primo giocatore.

Esegui dalla root con:
    python -m experiments.eval_head_to_head_connect4
"""

import torch
import torch.nn as nn

from ai.minimax import minimax_decision
from utils.encoding import state_to_np_array
from game.connect_four import (
    create_initial_state,
    apply_move,
    get_game_result,
    PLAYER1,
    PLAYER2,
)

# ---------------------------------------------------------------------
# MLP con stessa architettura del training (hidden_dim=128)
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
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_minimax_agent_from_model(model: MLP,
                                  device: str = "cpu",
                                  L: int = 2,
                                  K: int | None = 4):
    def H(state):
        x_np = state_to_np_array(state)
        x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return model(x).item()

    def agent(state):
        move, _ = minimax_decision(state, L=L, K=K, H=H)
        return move

    return agent


def play_single_game_p1_p2(agent_p1, agent_p2):
    """
    Gioca una partita:
      - agent_p1 come PLAYER1
      - agent_p2 come PLAYER2

    Restituisce z ∈ {-1,0,+1} dal punto di vista di PLAYER1.
    """
    state = create_initial_state()

    while True:
        res = get_game_result(state)
        if res is not None:
            return res

        if state.current_player == PLAYER1:
            move = agent_p1(state)
        else:
            move = agent_p2(state)

        if move is None:
            return 0  # consideriamo pareggio

        state = apply_move(state, move)


def head_to_head(path_new: str,
                 path_old: str,
                 L: int,
                 K: int | None,
                 n_games: int = 50,
                 device: str = "cpu"):
    """
    Confronta model_new vs model_old.

    In metà delle partite il modello nuovo è PLAYER1,
    nell'altra metà è PLAYER2.

    Restituisce (wins_new, draws, losses_new).
    """
    rows, cols = 6, 7
    input_dim = rows * cols

    # Carica i due modelli con la stessa architettura MLP
    model_new = MLP(input_dim=input_dim, hidden_dim=128)
    model_new.load_state_dict(torch.load(path_new, map_location=device))
    model_new.to(device)
    model_new.eval()

    model_old = MLP(input_dim=input_dim, hidden_dim=128)
    model_old.load_state_dict(torch.load(path_old, map_location=device))
    model_old.to(device)
    model_old.eval()

    agent_new = make_minimax_agent_from_model(model_new, device=device, L=L, K=K)
    agent_old = make_minimax_agent_from_model(model_old, device=device, L=L, K=K)

    wins_new = draws = losses_new = 0

    for i in range(n_games):
        # alterniamo chi inizia
        if i % 2 == 0:
            # new = PLAYER1, old = PLAYER2
            z = play_single_game_p1_p2(agent_new, agent_old)
            # z è dal punto di vista di PLAYER1 → coincide con "new"
            if z > 0:
                wins_new += 1
            elif z < 0:
                losses_new += 1
            else:
                draws += 1
        else:
            # old = PLAYER1, new = PLAYER2
            z = play_single_game_p1_p2(agent_old, agent_new)
            # ora z è dal punto di vista di "old" (PLAYER1)
            if z > 0:
                # ha vinto old → new perde
                losses_new += 1
            elif z < 0:
                # ha vinto PLAYER2 → new vince
                wins_new += 1
            else:
                draws += 1

    print(f"Head-to-head NEW({path_new}) vs OLD({path_old}):")
    print(f"  L={L}, K={K}, games={n_games}")
    print(f"  NEW W/D/L = {wins_new}/{draws}/{losses_new}")

    return wins_new, draws, losses_new


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    device = "cpu"

    print("=== Head-to-head: Strategy B (new) vs Strategy A (baseline) ===")
    try:
        head_to_head(
            path_new="results/model_strategy_B.pth",
            path_old="results/model_strategy_A.pth",
            L=3,
            K=4,
            n_games=200,
            device=device,
        )
    except FileNotFoundError as e:
        print("  [SKIP] File non trovato:", e)

    print("\n=== Head-to-head: Strategy C (new) vs Strategy B (old) ===")
    try:
        head_to_head(
            path_new="results/model_strategy_C.pth",
            path_old="results/model_strategy_B.pth",
            L=3,
            K=4,
            n_games=200,
            device=device,
        )
    except FileNotFoundError as e:
        print("  [SKIP] File non trovato:", e)
