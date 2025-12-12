import numpy as np
from ai.minimax import minimax_decision
from ai.model import ConnectFourNet, HTrueWrapper
from game.connect_four import (
    create_initial_state,
    apply_move,
    get_game_result,
)
from utils.encoding import state_to_np_array
import torch

def play_game_vs_random(H_wrapper, L=2, K=5):
    """Il modello gioca contro un giocatore random."""
    from game.connect_four import get_valid_moves
    state = create_initial_state()

    turn = 0

    while True:
        result = get_game_result(state)
        if result is not None:
            return result

        moves = get_valid_moves(state)
        if not moves:
            return 0  # draw-safe

        if turn % 2 == 0:
            # AI turn
            move, _ = minimax_decision(state, L=L, K=K, H=H_wrapper)
        else:
            # random opponent
            move = np.random.choice(moves)

        state = apply_move(state, move)
        turn += 1


def evaluate_strength(model, n_games=50):
    H_wrapper = HTrueWrapper(model)

    wins = draws = losses = 0

    for _ in range(n_games):
        z = play_game_vs_random(H_wrapper)
        if z == 1:
            wins += 1
        elif z == -1:
            losses += 1
        else:
            draws += 1

    print("Evaluation vs random:")
    print("  Wins:", wins)
    print("  Draws:", draws)
    print("  Losses:", losses)

    return wins, draws, losses


if __name__ == "__main__":
    model = ConnectFourNet(input_dim=42)
    model.load_state_dict(torch.load("results/latest_model.pth"))
    evaluate_strength(model)
