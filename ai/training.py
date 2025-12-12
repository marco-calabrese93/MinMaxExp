from typing import List, Tuple
import random
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from game.connect_four import (
    ConnectFourState,
    create_initial_state,
    get_game_result,
    PLAYER1,
    PLAYER2,
    get_valid_moves,
    apply_move,
)
from ai.minimax import minimax_decision
from ai.model import HTrueWrapper, ConnectFourNet
from utils.encoding import state_to_np_array


# ============================================================
# Replay Buffer (semplice ma efficace)
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=50000):
        from collections import deque
        self.buffer = deque(maxlen=capacity)

    def add_episode(self, states, z):
        """Aggiunge tutti gli stati della partita, ognuno con target z."""
        for s in states:
            self.buffer.append((s, z))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, zs = zip(*batch)
        return list(states), list(zs)

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Self-play game
# ============================================================

def self_play_game(H_wrapper: HTrueWrapper,
                   L: int,
                   K: int) -> Tuple[List[ConnectFourState], int]:

    states: List[ConnectFourState] = []
    state = create_initial_state()

    while True:
        states.append(state)
        result = get_game_result(state)
        if result is not None:
            z = result
            return states, z

        move, _ = minimax_decision(state, L=L, K=K, H=H_wrapper)
        state = apply_move(state, move)


# ============================================================
# Train on batch
# ============================================================

def train_on_batch(model: ConnectFourNet,
                   optimizer: optim.Optimizer,
                   batch_states: List[ConnectFourState],
                   batch_z: List[int],
                   device: str = "cpu") -> float:

    model.train()
    xs = []
    ys = []

    for s, z in zip(batch_states, batch_z):
        xs.append(state_to_np_array(s))
        ys.append(float(z))

    x_np = np.stack(xs, axis=0)
    y_np = np.array(ys, dtype=np.float32).reshape(-1, 1)

    x = torch.from_numpy(x_np).float().to(device)
    y = torch.from_numpy(y_np).float().to(device)

    optimizer.zero_grad()
    y_pred = model(x)
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    return float(loss.item())


# ============================================================
# Training loop con logging & replay buffer
# ============================================================

def training_loop(num_iterations: int,
                  L_schedule,
                  K_schedule,
                  device: str = "cpu",
                  use_replay=True,
                  replay_capacity=50000,
                  batch_size=64,
                  log_csv_path="results/training_log.csv"):

    rows, cols = 6, 7
    input_dim = rows * cols

    model = ConnectFourNet(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    H_wrapper = HTrueWrapper(model, device=device)

    # Replay Buffer opzionale
    replay = ReplayBuffer(capacity=replay_capacity) if use_replay else None

    # Logging CSV
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "L", "K", "z", "num_states", "loss"])

    history = []

    for t in range(num_iterations):
        L = L_schedule(t)
        K = K_schedule(t)

        # Self-play
        states, z = self_play_game(H_wrapper, L=L, K=K)

        # Training (con replay opzionale)
        if replay is not None:
            replay.add_episode(states, z)
            batch_states, batch_z = replay.sample(batch_size)
            loss = train_on_batch(model, optimizer, batch_states, batch_z, device=device)
        else:
            loss = train_on_batch(model, optimizer, states, [z] * len(states), device=device)

        # Log CSV
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([t, L, K, z, len(states), loss])

        # Log interno
        info = {
            "iteration": t,
            "L": L,
            "K": K,
            "z": z,
            "num_states": len(states),
            "loss": loss,
        }
        history.append(info)

    return model, history
