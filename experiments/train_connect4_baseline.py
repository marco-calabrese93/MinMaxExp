"""
experiments/train_connect4_baseline.py

Training del Predictive MinMax su Connect Four (6x7) con
strategia BASELINE: profondità L e ampiezza K COSTANTI.

Esegui dalla root del progetto con:
    python -m experiments.train_connect4_baseline
"""

from dataclasses import dataclass
from typing import List
import random
import numpy as np
import torch
import csv
import os

import torch.nn as nn
import torch.optim as optim

# ============================================================
# 0. Funzione per fissare il seed (riproducibilità)
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 1. IMPORT DEL GIOCO CONNECT FOUR (6x7 reale)
# ============================================================

from game.connect_four import (
    ConnectFourState,
    PLAYER1,
    PLAYER2,
    EMPTY,
    create_initial_state,
    get_valid_moves,
    apply_move,
    get_game_result,
)


# ============================================================
# 2. Wrapper OO: Connect Four Env
# ============================================================

class ConnectFourEnv:
    def reset(self) -> ConnectFourState:
        return create_initial_state()  # di default 6x7

    def legal_actions(self, state: ConnectFourState) -> List[int]:
        return get_valid_moves(state)

    def next_state(self, state: ConnectFourState, action: int) -> ConnectFourState:
        return apply_move(state, action)

    def is_terminal(self, state: ConnectFourState) -> bool:
        return get_game_result(state) is not None

    def outcome(self, state: ConnectFourState) -> int:
        res = get_game_result(state)
        if res is None:
            raise ValueError("outcome chiamato su stato non terminale")
        return res


# ============================================================
# 3. MLP per H_true
# ============================================================

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


# ============================================================
# 4. Encoding dello stato
# ============================================================

def encode_state(state: ConnectFourState, player: int) -> List[float]:
    """
    Codifica la board come vettore flatten di dimensione rows*cols,
    ribaltato dal punto di vista del giocatore `player`.
    """
    board = np.array(state.board, dtype=float)
    board = board * player  # punto di vista del giocatore corrente
    return board.flatten().tolist()


def get_input_dim(example_state: ConnectFourState) -> int:
    return len(encode_state(example_state, example_state.current_player))


# ============================================================
# 5. MinMax con tagli L,K
# ============================================================

@dataclass
class MinMaxConfig:
    depth_limit: int
    width_limit: int


def evaluate_state_with_mlp(state: ConnectFourState, player: int, model: MLP) -> float:
    model.eval()
    with torch.no_grad():
        x_vec = encode_state(state, player)
        x = torch.tensor(x_vec, dtype=torch.float32).unsqueeze(0)
        return model(x).item()


def ordered_legal_moves(env: ConnectFourEnv, state: ConnectFourState, model: MLP):
    legal = env.legal_actions(state)
    p = state.current_player
    out = []
    for a in legal:
        s2 = env.next_state(state, a)
        score = evaluate_state_with_mlp(s2, p, model)
        out.append((score, a, s2))
    # ordina discendente per usare prima le mosse promettenti
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def minimax_rec(env, state, depth, root_player, config, model):

    if depth == 0 or env.is_terminal(state):
        v_move = evaluate_state_with_mlp(state, state.current_player, model)
        return v_move if state.current_player == root_player else -v_move

    moves = ordered_legal_moves(env, state, model)
    moves = moves[: config.width_limit]

    if not moves:
        return 0.0

    if state.current_player == root_player:
        best = float("-inf")
        for _, _, child in moves:
            val = minimax_rec(env, child, depth - 1, root_player, config, model)
            best = max(best, val)
        return best
    else:
        best = float("inf")
        for _, _, child in moves:
            val = minimax_rec(env, child, depth - 1, root_player, config, model)
            best = min(best, val)
        return best


def choose_action_minmax(env, state, config, model):
    legal = env.legal_actions(state)
    if not legal:
        return None

    root = state.current_player
    best_action = None
    best_val = float("-inf")

    for a in legal:
        s2 = env.next_state(state, a)
        v = minimax_rec(env, s2, config.depth_limit - 1, root, config, model)
        if v > best_val:
            best_val = v
            best_action = a

    return best_action


# ============================================================
# 6. Self-play
# ============================================================

def self_play_game(env, model, config):
    state = env.reset()
    states = []
    players = []

    while not env.is_terminal(state):
        p = state.current_player
        states.append(state)
        players.append(p)

        action = choose_action_minmax(env, state, config, model)

        if action is None:  # fallback random
            legal = env.legal_actions(state)
            if not legal:
                break
            action = random.choice(legal)

        state = env.next_state(state, action)

    z = env.outcome(state)
    return states, players, z


def build_batch_from_game(states, players, z):
    X = []
    y = []
    for s, p in zip(states, players):
        X.append(encode_state(s, p))
        y.append(z * p)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X, y


# ============================================================
# 7. Valutazione vs random
# ============================================================

def play_vs_random(env, model, config, n_games=20):
    wins = draws = losses = 0

    for _ in range(n_games):
        state = env.reset()

        while not env.is_terminal(state):
            if state.current_player == PLAYER1:
                action = choose_action_minmax(env, state, config, model)
                if action is None:
                    legal = env.legal_actions(state)
                    if not legal:
                        break
                    action = random.choice(legal)
            else:
                legal = env.legal_actions(state)
                if not legal:
                    break
                action = random.choice(legal)

            state = env.next_state(state, action)

        z = env.outcome(state)

        if z > 0:
            wins += 1
        elif z < 0:
            losses += 1
        else:
            draws += 1

    return wins, draws, losses


# ============================================================
# 8. Salvataggio log in CSV
# ============================================================

def save_logs_to_csv(logs, filepath: str):
    """
    Salva la lista di dizionari `logs` in un file CSV.
    Crea automaticamente la cartella se non esiste.
    """
    if not logs:
        return

    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fieldnames = [
        "seed",
        "iter",
        "L",
        "K",
        "batch_size",
        "loss",
        "wins_vs_random",
        "draws_vs_random",
        "losses_vs_random",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)


# ============================================================
# 9. Train loop BASELINE (L,K costanti)
# ============================================================

def train_connect4_baseline(
    T_max=50,
    N_selfplay_per_iter=5,
    hidden_dim=128,
    lr=1e-3,
    seed=0,
    L_const: int = 2,
    K_const: int = 4,
    log_csv_path: str | None = "results/connect4_baseline_seed0_L2K4.csv",
):
    """
    Training del Predictive MinMax con L e K costanti (baseline).
    """
    set_seed(seed)

    env = ConnectFourEnv()

    example_state = env.reset()
    input_dim = get_input_dim(example_state)

    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    logs = []

    for t in range(1, T_max + 1):
        L = L_const
        K = K_const
        config = MinMaxConfig(depth_limit=L, width_limit=K)

        all_X = []
        all_y = []

        for _ in range(N_selfplay_per_iter):
            states, players, z = self_play_game(env, model, config)
            Xb, yb = build_batch_from_game(states, players, z)
            all_X.append(Xb)
            all_y.append(yb)

        X = torch.cat(all_X, dim=0)
        y = torch.cat(all_y, dim=0)

        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        wins = draws = losses_vs = None

        print(f"[Iter {t:03d}] L={L} K={K} batch={len(X)} loss={loss.item():.4f}")

        if t % 10 == 0:
            wins, draws, losses_vs = play_vs_random(env, model, config, n_games=20)
            print(f"    vs random -> W/D/L = {wins}/{draws}/{losses_vs}")

        logs.append({
            "seed": seed,
            "iter": t,
            "L": L,
            "K": K,
            "batch_size": len(X),
            "loss": float(loss.item()),
            "wins_vs_random": wins,
            "draws_vs_random": draws,
            "losses_vs_random": losses_vs,
        })

    if log_csv_path is not None:
        save_logs_to_csv(logs, log_csv_path)
        print(f"Log salvati in: {log_csv_path}")

    return model, logs


# ============================================================
# 10. Entry point
# ============================================================

if __name__ == "__main__":
    model, logs = train_connect4_baseline(
        T_max=100,
        N_selfplay_per_iter=5,
        hidden_dim=128,
        lr=1e-3,
        seed=0,
        L_const=2,
        K_const=4,
        log_csv_path="results/connect4_baseline_seed0_L2K4.csv",
    )

    # Salvataggio del modello allenato come "Strategy A"
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/model_strategy_A.pth")
    print("Modello baseline salvato in results/model_strategy_A.pth")
