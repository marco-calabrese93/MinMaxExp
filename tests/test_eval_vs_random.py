from experiments.eval_vs_random import evaluate_strength
from ai.model import ConnectFourNet

def test_eval_vs_random_runs():
    model = ConnectFourNet(input_dim=42)
    wins, draws, losses = evaluate_strength(model, n_games=5)
    assert wins + draws + losses == 5
