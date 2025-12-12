from experiments.eval_offline_connect4 import evaluate_model
from ai.model import ConnectFourNet
import torch

def test_offline_eval_runs():
    model = ConnectFourNet(input_dim=42)
    mse, corr = evaluate_model(model, n_states=10)
    assert mse >= 0
    assert -1 <= corr <= 1
