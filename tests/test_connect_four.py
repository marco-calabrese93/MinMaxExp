import numpy as np

from game.connect_four import (
    PLAYER1, PLAYER2, EMPTY,
    create_initial_state,
    get_valid_moves,
    apply_move,
    get_game_result,
    ConnectFourState,
)
from utils.encoding import state_to_np_array


def test_initial_state():
    s = create_initial_state()

    # Il giocatore iniziale deve essere PLAYER1
    assert s.current_player == PLAYER1

    # La prima riga deve essere vuota
    assert all(s.board[0][c] == EMPTY for c in range(s.cols))

    # All'inizio tutte le 7 colonne sono mosse valide
    assert get_valid_moves(s) == [0, 1, 2, 3, 4, 5, 6]


def test_apply_move():
    s = create_initial_state()
    s2 = apply_move(s, 3)

    # Dopo la prima mossa, la pedina di PLAYER1 deve essere in fondo alla colonna 3
    assert s2.board[s.rows - 1][3] == PLAYER1

    # Il turno passa a PLAYER2
    assert s2.current_player == PLAYER2


def test_vertical_win():
    s = create_initial_state()
    # Simuliamo 4 mosse verticali nella stessa colonna per PLAYER1
    s = apply_move(s, 0)   # P1
    s = apply_move(s, 1)   # P2
    s = apply_move(s, 0)   # P1
    s = apply_move(s, 1)   # P2
    s = apply_move(s, 0)   # P1
    s = apply_move(s, 1)   # P2
    s = apply_move(s, 0)   # P1 → vince P1 in verticale

    assert get_game_result(s) == PLAYER1


def test_horizontal_win():
    s = create_initial_state()
    s = apply_move(s, 0) # P1
    s = apply_move(s, 0) # P2
    s = apply_move(s, 1) # P1
    s = apply_move(s, 1) # P2
    s = apply_move(s, 2) # P1
    s = apply_move(s, 2) # P2
    s = apply_move(s, 3) # P1 → orizzontale 0-3 sulla bottom row

    assert get_game_result(s) == PLAYER1


def test_diagonal_win():
    s = create_initial_state()
    # Costruiamo una diagonale tipo "\" per PLAYER1
    s = apply_move(s, 0)  # P1
    s = apply_move(s, 1)  # P2
    s = apply_move(s, 1)  # P1
    s = apply_move(s, 2)  # P2
    s = apply_move(s, 2)  # P1
    s = apply_move(s, 3)  # P2
    s = apply_move(s, 2)  # P1
    s = apply_move(s, 3)  # P2
    s = apply_move(s, 3)  # P1
    s = apply_move(s, 4)  # P2
    s = apply_move(s, 3)  # P1 → diagonale completata

    assert get_game_result(s) == PLAYER1


def test_encoding_dimension():
    s = create_initial_state()
    x = state_to_np_array(s)

    # Board 6x7 → vettore 42
    assert x.shape == (6 * 7,)

