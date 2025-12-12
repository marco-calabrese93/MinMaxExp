from ai.model import ConnectFourNet
from experiments.eval_offline_connect4 import evaluate_model
import torch


def main():
    rows, cols = 6, 7
    input_dim = rows * cols

    model = ConnectFourNet(input_dim=input_dim)
    # Modello allenato con train_predictive_minmax.py
    state_dict = torch.load("results/model_strategy_B.pth", map_location="cpu")
    model.load_state_dict(state_dict)

    mse, corr = evaluate_model(
        model,
        device="cpu",
        n_states=200,
        use_csv=True,
        csv_path="results/connect4_testset_L4.csv",
    )

    print("[Strategy B - curriculum v1]")
    print("MSE :", mse)
    print("corr:", corr)


if __name__ == "__main__":
    main()
