import os
import csv

from ai.training import training_loop


def test_training_loop_creates_log_csv(tmp_path):
    """
    Verifica che il training_loop:
    - giri senza errori per poche iterazioni
    - crei un file CSV di log con header + righe dati
    """

    # Funzioni L(t) e K(t) molto leggere per velocizzare il test
    def L_schedule(t: int) -> int:
        return 1  # profondità bassa

    def K_schedule(t: int) -> int:
        return 2  # poche mosse per minimax

    # Mettiamo il log in una directory temporanea creata da pytest
    log_csv_path = tmp_path / "training_log_test.csv"

    # Eseguiamo un training brevissimo
    model, history = training_loop(
        num_iterations=2,
        L_schedule=L_schedule,
        K_schedule=K_schedule,
        device="cpu",
        use_replay=False,           # per renderlo più deterministico e semplice
        log_csv_path=str(log_csv_path),
    )

    # 1) Il file CSV deve esistere
    assert log_csv_path.exists()

    # 2) Deve avere header + 2 righe dati
    with open(log_csv_path, "r", newline="") as f:
        reader = list(csv.reader(f))

    # Prima riga = header
    header = reader[0]
    assert header == ["iteration", "L", "K", "z", "num_states", "loss"]

    # Due righe successive = due iterazioni
    assert len(reader) == 1 + 2

    # 3) history in memoria deve avere 2 elementi con le chiavi giuste
    assert len(history) == 2
    for info in history:
        for key in ["iteration", "L", "K", "z", "num_states", "loss"]:
            assert key in info
