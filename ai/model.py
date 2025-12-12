import torch
import torch.nn as nn
from game.connect_four import ConnectFourState
from utils.encoding import state_to_np_array


class ConnectFourNet(nn.Module):
    """
    Semplice MLP per stimare H_true(s).
    Input: stato flattenato (rows * cols)
    Output: valore scalare in [-1, +1] (approssimazione dell'esito).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # restringe l'output a circa [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HTrueWrapper:
    """
    Wrapper comodo per usare il modello come funzione H_true(state) -> float.
    """
    def __init__(self, model: ConnectFourNet, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def __call__(self, state: ConnectFourState) -> float:
        self.model.eval()
        with torch.no_grad():
            x_np = state_to_np_array(state)
            x = torch.from_numpy(x_np).float().to(self.device)
            x = x.unsqueeze(0)  # batch size 1
            y = self.model(x)
            return float(y.item())
