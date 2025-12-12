import numpy as np

from game.tictactoe import (
    TicTacToeState,
    create_initial_state,
    apply_move,
    get_valid_moves,
    get_game_result,
    PLAYER1,
    PLAYER2,
    EMPTY,
)
from utils.encoding import state_to_np_array


def test_tictactoe_initial_state():
    state = create_initial_state()
    board = state.board
    player = state.current_player

    # Board 3x3 tutta vuota
    assert len(board) == 3
    assert len(board[0]) == 3
    assert all(cell == EMPTY for row in board for cell in row)
    # Deve iniziare PLAYER1
    assert player == PLAYER1


def test_tictactoe_valid_moves_initial():
    state = create_initial_state()
    moves = get_valid_moves(state)

    # 9 celle libere → 9 mosse (0..8)
    assert len(moves) == 9
    assert set(moves) == set(range(9))


def test_tictactoe_apply_move_changes_board_and_player():
    state = create_initial_state()
    move = 0  # cella (0,0)
    next_state = apply_move(state, move)

    # Board modificata nella cella corretta
    assert next_state.board[0][0] == PLAYER1
    # Turno deve passare a PLAYER2
    assert next_state.current_player == PLAYER2


def test_tictactoe_horizontal_win_p1():
    state = create_initial_state()
    # Riga 0 = [1,1,1]
    state.board[0] = [PLAYER1, PLAYER1, PLAYER1]
    assert get_game_result(state) == PLAYER1


def test_tictactoe_vertical_win_p1():
    state = create_initial_state()
    # Colonna 1 = [1,1,1]
    state.board[0][1] = PLAYER1
    state.board[1][1] = PLAYER1
    state.board[2][1] = PLAYER1
    assert get_game_result(state) == PLAYER1


def test_tictactoe_diagonal_win_p1():
    state = create_initial_state()
    # Diagonale principale = [1,1,1]
    state.board[0][0] = PLAYER1
    state.board[1][1] = PLAYER1
    state.board[2][2] = PLAYER1
    assert get_game_result(state) == PLAYER1


def test_tictactoe_draw_no_three_in_a_row():
    state = create_initial_state()
    # Board piena senza 3 uguali allineati
    state.board = [
        [ 1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
    ]
    assert get_game_result(state) == 0  # pareggio


def test_tictactoe_encoding_dimension():
    state = create_initial_state()
    x = state_to_np_array(state)
    assert x.shape == (9,)
