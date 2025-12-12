import torch

from ai.training import (
    self_play_game,
    train_on_batch,
    training_loop,
)
from ai.model import ConnectFourNet, HTrueWrapper
from game.connect_four import (
    create_initial_state,
    get_game_result,
)
from utils.encoding import state_to_np_array


def test_self_play_game_terminates():
    # Modello piccolo e casuale
    rows, cols = 6, 7
    input_dim = rows * cols
    model = ConnectFourNet(input_dim=input_dim)
    H_wrapper = HTrueWrapper(model, device="cpu")

    states, z = self_play_game(
        H_wrapper,
        L=1,
        K=3,   # esplora poche mosse per velocizzare il test
    )

    # Deve aver generato almeno uno stato
    assert len(states) > 0
    # Esito deve essere in {-1, 0, +1}
    assert z in (-1, 0, 1)

    # Controllo che l'ultimo stato sia davvero terminale e coerente con z
    last_state = states[-1]
    result = get_game_result(last_state)
    assert result == z


def test_train_on_batch_updates_model_parameters():
    rows, cols = 6, 7
    input_dim = rows * cols
    model = ConnectFourNet(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    H_wrapper = HTrueWrapper(model, device="cpu")

    # Generiamo una partita di self-play
    states, z = self_play_game(H_wrapper, L=1, K=2)

    # Salviamo una copia dei parametri prima del training
    before = {name: p.clone().detach() for name, p in model.state_dict().items()}

    # Eseguiamo un passo di training
    loss = train_on_batch(model, optimizer, states, z, device="cpu")

    # Almeno un parametro deve essere cambiato
    after = model.state_dict()
    changed = any(
        not torch.allclose(before[name], after[name])
        for name in before.keys()
    )

    assert changed, "Model parameters did not change after training step"
    assert loss >= 0.0


def test_training_loop_runs_one_iteration():
    def L_schedule(t: int) -> int:
        return 1  # profondità bassa per i test

    def K_schedule(t: int) -> int:
        return 3  # poche mosse esplorate

    model, history = training_loop(
        num_iterations=1,
        L_schedule=L_schedule,
        K_schedule=K_schedule,
        device="cpu",
    )

    # Una sola iterazione loggata
    assert len(history) == 1
    info = history[0]
    # Controllo che le chiavi principali siano presenti
    for key in ["iteration", "L", "K", "z", "num_states", "loss"]:
        assert key in info
