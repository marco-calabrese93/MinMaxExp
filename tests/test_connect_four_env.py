import numpy as np

from game.connect_four import (
    ConnectFourState,
    create_initial_state,
    apply_move,
    get_valid_moves,
    get_game_result,
    PLAYER1,
    PLAYER2,
    EMPTY,
)
from utils.encoding import state_to_np_array


def test_connect_four_initial_state():
    state = create_initial_state()
    board = state.board
    player = state.current_player

    # Board 6x7 tutta vuota
    assert len(board) == 6
    assert len(board[0]) == 7
    assert all(cell == EMPTY for row in board for cell in row)
    assert player == PLAYER1


def test_connect_four_valid_moves_initial():
    state = create_initial_state()
    moves = get_valid_moves(state)

    # Tutte le 7 colonne sono giocabili all'inizio
    assert moves == list(range(7))


def test_connect_four_apply_move_bottom_gravity():
    state = create_initial_state()
    col = 3
    next_state = apply_move(state, col)

    # Il gettone deve cadere in bottom row (riga 5)
    assert next_state.board[5][col] == PLAYER1
    # Turno passa a PLAYER2
    assert next_state.current_player == PLAYER2


def test_connect_four_apply_move_stacks_tokens():
    state = create_initial_state()

    s1 = apply_move(state, 0)  # P1 in (5,0)
    s2 = apply_move(s1, 0)     # P2 in (4,0)

    assert s1.board[5][0] == PLAYER1
    assert s2.board[4][0] == PLAYER2


def test_connect_four_horizontal_win_p1():
    state = create_initial_state()
    # Quattro in riga sulla bottom row
    row = 5
    state.board[row][0] = PLAYER1
    state.board[row][1] = PLAYER1
    state.board[row][2] = PLAYER1
    state.board[row][3] = PLAYER1
    assert get_game_result(state) == PLAYER1


def test_connect_four_vertical_win_p1():
    state = create_initial_state()
    col = 2
    # Quattro in colonna
    state.board[5][col] = PLAYER1
    state.board[4][col] = PLAYER1
    state.board[3][col] = PLAYER1
    state.board[2][col] = PLAYER1
    assert get_game_result(state) == PLAYER1


def test_connect_four_diagonal_win_p1():
    state = create_initial_state()
    # Diagonale bottom-left → top-right
    state.board[5][0] = PLAYER1
    state.board[4][1] = PLAYER1
    state.board[3][2] = PLAYER1
    state.board[2][3] = PLAYER1
    assert get_game_result(state) == PLAYER1


def test_connect_four_encoding_dimension():
    state = create_initial_state()
    x = state_to_np_array(state)
    assert x.shape == (6 * 7,)
