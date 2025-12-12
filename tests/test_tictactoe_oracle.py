from game.tictactoe import (
    TicTacToeState,
    create_initial_state,
    apply_move,
    get_game_result,
    get_valid_moves,
    PLAYER1,
    PLAYER2,
    EMPTY,
)
from experiments.tictactoe_oracle import (
    oracle_value_from_p1,
    oracle_best_move,
    oracle_initial_value,
)


def test_oracle_initial_state_is_draw():
    """
    Con gioco perfetto, TicTacToe è un pareggio.
    L'oracolo deve restituire 0 sulla posizione iniziale.
    """
    v0 = oracle_initial_value()
    assert v0 == 0


def test_oracle_finds_immediate_win_for_p1():
    """
    Se P1 ha già due simboli in fila e può vincere al prossimo colpo,
    l'oracolo deve scegliere la mossa vincente e dare valore +1.
    """
    state = create_initial_state()
    # Costruiamo una posizione:
    # [ X, X, . ]
    # [ ., ., . ]
    # [ ., ., . ]
    state.board[0][0] = PLAYER1
    state.board[0][1] = PLAYER1
    state.current_player = PLAYER1

    best_move, best_val = oracle_best_move(state)

    # La mossa vincente è la cella (0,2) cioè indice 2 se codifichi 0..8 in row-major
    # Ma non facciamo assunzioni sull'indice specifico: verifichiamo che dopo la mossa
    # lo stato sia vinto e il valore dell'oracolo sia +1.
    next_state = apply_move(state, best_move)
    result = get_game_result(next_state)

    assert result == PLAYER1
    assert best_val == 1


def test_oracle_detects_forced_loss_for_p1():
    """
    Costruiamo una posizione dove, qualunque cosa faccia P1, P2 ha una vittoria forzata.
    L'oracolo deve restituire -1 come valore dalla prospettiva di P1.
    """

    # Esempio semplice: diamo a P2 una doppia minaccia
    # Board (X = P1, O = P2):
    # [ O, O, EMPTY ]
    # [ X, X, EMPTY ]
    # [ EMPTY, EMPTY, EMPTY ]
    #
    # e tocca a P2, così P2 vince immediatamente.
    #
    # Poi guardiamo lo stato dal punto di vista di P1 DOPO una mossa di P1 in cui
    # P1 non può bloccare entrambe le minacce (per esercizi complessi puoi creare
    # configurazioni più articolate).
    state = TicTacToeState(
        board=[
            [PLAYER2, PLAYER2, EMPTY],
            [PLAYER1, PLAYER1, EMPTY],
            [EMPTY,   EMPTY,   EMPTY],
        ],
        current_player=PLAYER2,
    )

    # P2 gioca la mossa perfetta e vince subito
    best_move_p2, best_val_p2 = oracle_best_move(state)
    s_after = apply_move(state, best_move_p2)
    assert get_game_result(s_after) == PLAYER2
    # Dal punto di vista di P1 il valore finale è -1
    assert oracle_value_from_p1(state) == -1


def test_oracle_is_consistent_with_terminal_states():
    """
    Se lo stato è già terminale, oracle_value_from_p1 deve restituire l'esito corretto.
    """
    # Stato di vittoria per P1
    s = create_initial_state()
    s.board[0] = [PLAYER1, PLAYER1, PLAYER1]
    s.current_player = PLAYER2  # non importa chi tocca
    z = oracle_value_from_p1(s)
    assert z == 1

    # Stato di vittoria per P2
    s2 = create_initial_state()
    s2.board[0] = [PLAYER2, PLAYER2, PLAYER2]
    s2.current_player = PLAYER1
    z2 = oracle_value_from_p1(s2)
    assert z2 == -1

    # Stato di pareggio
    s3 = create_initial_state()
    s3.board = [
        [PLAYER1, PLAYER1, PLAYER2],
        [PLAYER2, PLAYER2, PLAYER1],
        [PLAYER1, PLAYER2, PLAYER1],
    ]
    z3 = oracle_value_from_p1(s3)
    assert z3 == 0
