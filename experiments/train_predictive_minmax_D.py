"""
experiments/train_predictive_minmax_D.py

Strategy D – Predictive MinMax su Connect Four (6x7)
con curriculum L(t), K(t) più aggressivo, ma stesso budget
di training delle altre strategie (T_max, N_selfplay_per_iter, architettura).

USO:
    python -m experiments.train_predictive_minmax_D
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
# 2. MLP euristica H_true (stessa architettura delle altre strategie)
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
    """
    Codifica la board come vettore flatten (rows*cols),
    dal punto di vista del giocatore `player`.
    """
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
    """
    Valutazione euristica H_true(s) usando l'MLP.
    Il valore è sempre dal punto di vista di `player`.
    """
    model.eval()
    with torch.no_grad():
        x_vec = encode_state(state, player)
        x = torch.tensor(x_vec, dtype=torch.float32).unsqueeze(0)
        return model(x).item()


def ordered_legal_moves(env: ConnectFourEnv,
                        state: ConnectFourState,
                        model: MLP):
    """
    Restituisce le mosse legali ordinate in base alla valutazione MLP
    (dal più promettente al meno promettente).
    Ogni elemento è una tupla (score, action, next_state).
    """
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
    """
    MinMax ricorsivo con:
      - profondità massima = config.depth_limit
      - ampiezza massima = config.width_limit (numero massimo di figli)
      - valutazione H_true (MLP) sui nodi foglia.
    """
    # Caso base: profondità 0 o stato terminale
    if depth == 0 or env.is_terminal(state):
        v = evaluate_state_with_mlp(state, state.current_player, model)
        # Riporta il valore dal punto di vista di root_player
        return v if state.current_player == root_player else -v

    moves = ordered_legal_moves(env, state, model)
    moves = moves[: config.width_limit]  # taglio sulle mosse esplorate

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
    """
    Restituisce la mossa scelta da Predictive MinMax(s, H_true, L, K).
    """
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
    """
    Gioca una partita di self-play in cui entrambi i giocatori
    usano Predictive MinMax con la stessa H_true, L, K.
    Restituisce:
      - lista degli stati
      - lista dei giocatori che hanno mosso
      - esito finale z ∈ {-1, 0, +1} dal punto di vista di PLAYER1
    """
    state = env.reset()
    states = []
    players = []

    while not env.is_terminal(state):
        p = state.current_player
        states.append(state)
        players.append(p)

        action = choose_action_minmax(env, state, config, model)

        # fallback random nel caso estremo
        if action is None:
            legal = env.legal_actions(state)
            if not legal:
                break
            action = random.choice(legal)

        state = env.next_state(state, action)

    z = env.outcome(state)
    return states, players, z


def build_batch_from_game(states, players, z):
    """
    Costruisce il batch (X, y) per l'update di H_true:
      - X: encoding degli stati dal punto di vista di chi ha mosso
      - y: z * player (valore del risultato dal punto di vista del giocatore che ha mosso)
    """
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
    """
    Gioca n partite contro un avversario random.
    L'agente MinMax+MLP è sempre PLAYER1.
    """
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


# ------------------------------------------------------------
# 8. Strategy D: curriculum L_D(t), K_D(t)
#    (stesso T_max delle altre, cambia solo come usiamo il budget)
# ------------------------------------------------------------

def L_schedule_D(t: int) -> int:
    """
    Strategy D: profondità L(t).
    - prime 10 iter: L=1 (fase di "riscaldamento")
    - iter 11–30:   L=2
    - iter 31–50:   L=3
    """
    if t <= 10:
        return 1
    elif t <= 30:
        return 2
    else:
        return 3


def K_schedule_D(t: int) -> int:
    """
    Strategy D: ampiezza K(t).
    - prime 10 iter: K=3  (poche mosse, rapido)
    - iter 11–30:   K=5
    - iter 31–50:   K=7  (più ampiezza verso la fine)
    """
    if t <= 10:
        return 3
    elif t <= 30:
        return 5
    else:
        return 7


# ------------------------------------------------------------
# 9. Train loop Strategy D
# ------------------------------------------------------------

def train_predictive_minmax_D(
    T_max: int = 50,
    N_selfplay_per_iter: int = 5,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    log_csv_path: str | None = "results/connect4_curriculum_seed0_D.csv",
    ckpt_path: str | None = "results/model_strategy_D.pth",
):
    """
    Training del Predictive MinMax con Strategy D:
    curriculum L_D(t), K_D(t) più aggressivo, ma stesso T_max e N_selfplay.
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
        L = L_schedule_D(t)
        K = K_schedule_D(t)
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

        print(f"[Strategy D][Iter {t:03d}] L={L} K={K} batch={len(X)} loss={loss.item():.4f}")

        # valutazione periodica vs random (ogni 10 iter per coerenza con le altre)
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

    # Salva CSV
    if log_csv_path is not None:
        save_logs_to_csv(logs, log_csv_path)
        print(f"Log Strategy D salvati in: {log_csv_path}")

    # Salva checkpoint
    if ckpt_path is not None:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint Strategy D salvato in: {ckpt_path}")

    return model, logs


# ------------------------------------------------------------
# 10. Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    model, logs = train_predictive_minmax_D(
        T_max=100,
        N_selfplay_per_iter=5,
        hidden_dim=128,
        lr=1e-3,
        seed=0,
        log_csv_path="results/connect4_curriculum_seed0_D.csv",
        ckpt_path="results/model_strategy_D.pth",
    )
