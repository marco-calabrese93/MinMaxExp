"""
experiments/train_predictive_minmax_E.py

Strategy E – Curriculum con vincolo K = 2L + 1
Progressione simile a Strategy B, ma con relazione regolare tra L e K.

USO:
    python -m experiments.train_predictive_minmax_E
"""

from dataclasses import dataclass
from typing import List
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv

from game.connect_four import (
    ConnectFourState,
    PLAYER1,
    PLAYER2,
    create_initial_state,
    get_valid_moves,
    apply_move,
    get_game_result,
)

# ------------------------------------------------------------
# 0. Seed fisso per riproducibilità
# ------------------------------------------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------
# 1. Environment wrapper
# ------------------------------------------------------------

class ConnectFourEnv:
    def reset(self) -> ConnectFourState:
        return create_initial_state()

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


# ------------------------------------------------------------
# 2. MLP euristica H_true (stessa architettura delle altre)
# ------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------
# 3. Encoding dello stato
# ------------------------------------------------------------

def encode_state(state: ConnectFourState, player: int) -> list[float]:
    board = np.array(state.board, dtype=float)
    board = board * player
    return board.flatten().tolist()


def get_input_dim(example_state: ConnectFourState) -> int:
    return len(encode_state(example_state, example_state.current_player))


# ------------------------------------------------------------
# 4. MinMax con profondità L e ampiezza K
# ------------------------------------------------------------

@dataclass
class MinMaxConfig:
    depth_limit: int
    width_limit: int  # numero massimo di mosse figlie da esplorare


def evaluate_state_with_mlp(state: ConnectFourState, player: int, model: MLP) -> float:
    model.eval()
    with torch.no_grad():
        x_vec = encode_state(state, player)
        x = torch.tensor(x_vec, dtype=torch.float32).unsqueeze(0)
        return model(x).item()


def ordered_legal_moves(env: ConnectFourEnv,
                        state: ConnectFourState,
                        model: MLP):
    legal = env.legal_actions(state)
    p = state.current_player
    scored = []
    for a in legal:
        s2 = env.next_state(state, a)
        score = evaluate_state_with_mlp(s2, p, model)
        scored.append((score, a, s2))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def minimax_rec(env: ConnectFourEnv,
                state: ConnectFourState,
                depth: int,
                root_player: int,
                config: MinMaxConfig,
                model: MLP) -> float:
    if depth == 0 or env.is_terminal(state):
        v = evaluate_state_with_mlp(state, state.current_player, model)
        return v if state.current_player == root_player else -v

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


def choose_action_minmax(env: ConnectFourEnv,
                         state: ConnectFourState,
                         config: MinMaxConfig,
                         model: MLP) -> int | None:
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


# ------------------------------------------------------------
# 5. Self-play
# ------------------------------------------------------------

def self_play_game(env: ConnectFourEnv,
                   model: MLP,
                   config: MinMaxConfig):
    state = env.reset()
    states = []
    players = []

    while not env.is_terminal(state):
        p = state.current_player
        states.append(state)
        players.append(p)

        action = choose_action_minmax(env, state, config, model)
        if action is None:
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


# ------------------------------------------------------------
# 6. Valutazione vs random
# ------------------------------------------------------------

def play_vs_random(env: ConnectFourEnv,
                   model: MLP,
                   config: MinMaxConfig,
                   n_games: int = 20):
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


# ------------------------------------------------------------
# 7. CSV logging
# ------------------------------------------------------------

def save_logs_to_csv(logs, filepath: str):
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


# ------------------------------------------------------------
# 8. Strategy E: curriculum con K = 2L + 1
# ------------------------------------------------------------

def L_schedule_E(t: int) -> int:
    """
    Curriculum "dolce" simile a Strategy B:
    - 1 <= t <= 15  -> L = 1
    - 16 <= t <= 35 -> L = 2
    - t >= 36       -> L = 3
    """
    if t <= 15:
        return 1
    elif t <= 35:
        return 2
    else:
        return 3


def K_schedule_E(t: int) -> int:
    """
    Vincolo K = 2L + 1.
    """
    L = L_schedule_E(t)
    return 2 * L + 1


# ------------------------------------------------------------
# 9. Train loop Strategy E
# ------------------------------------------------------------

def train_predictive_minmax_E(
    T_max: int = 100,
    N_selfplay_per_iter: int = 5,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    log_csv_path: str | None = "results/connect4_curriculum_seed0_E.csv",
    ckpt_path: str | None = "results/model_strategy_E.pth",
):
    set_seed(seed)

    env = ConnectFourEnv()
    example_state = env.reset()
    input_dim = get_input_dim(example_state)

    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    logs = []

    for t in range(1, T_max + 1):
        L = L_schedule_E(t)
        K = K_schedule_E(t)
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

        print(f"[Strategy E][Iter {t:03d}] L={L} K={K} batch={len(X)} loss={loss.item():.4f}")

        # valutazione periodica vs random ogni 10 iter
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
        print(f"Log Strategy E salvati in: {log_csv_path}")

    if ckpt_path is not None:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint Strategy E salvato in: {ckpt_path}")

    return model, logs


# ------------------------------------------------------------
# 10. Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    model, logs = train_predictive_minmax_E(
        T_max=100,
        N_selfplay_per_iter=5,
        hidden_dim=128,
        lr=1e-3,
        seed=0,
        log_csv_path="results/connect4_curriculum_seed0_E.csv",
        ckpt_path="results/model_strategy_E.pth",
    )
