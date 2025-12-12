from ai.training import ReplayBuffer
from game.connect_four import create_initial_state


def test_replay_buffer_add_and_len():
    buf = ReplayBuffer(capacity=10)

    s1 = create_initial_state()
    s2 = create_initial_state()

    # Aggiungo due "episodi" con 2 e 3 stati
    buf.add_episode([s1, s2], z=1)
    buf.add_episode([s1, s2, s1], z=-1)

    # 2 + 3 = 5 elementi totali
    assert len(buf) == 5


def test_replay_buffer_respects_capacity():
    buf = ReplayBuffer(capacity=4)

    s = create_initial_state()

    # Aggiungo più episodi finché supero la capacità
    for _ in range(5):
        buf.add_episode([s, s], z=1)  # ogni episodio aggiunge 2 stati

    # La lunghezza non deve superare la capacity
    assert len(buf) == 4


def test_replay_buffer_sample_returns_valid_batch():
    buf = ReplayBuffer(capacity=10)
    s = create_initial_state()

    buf.add_episode([s, s, s, s], z=1)

    states, zs = buf.sample(batch_size=3)

    # batch size corretto (o min(len(buf), batch_size))
    assert len(states) == len(zs)
    assert len(states) <= 3
    assert len(states) > 0

    # tipi sensati
    for st in states:
        # controllo soft: lo stato deve avere attributi tipici di ConnectFourState
        assert hasattr(st, "board")
        assert hasattr(st, "current_player")

    for z in zs:
        assert z in (-1, 0, 1)
