from ai.minimax import (
    minimax_decision,
    evaluate_state_from_p1_perspective,
)
from game.connect_four import (
    create_initial_state,
    apply_move,
    get_game_result,
    PLAYER1,
    PLAYER2,
)
from typing import Callable


def constant_heuristic(state) -> float:
    # Euristica fittizia: ritorna sempre 0
    return 0.0


def test_minimax_finds_immediate_win_for_p1():
    state = create_initial_state()
    # Costruiamo una posizione dove PLAYER1 ha 3 in fila e può vincere giocando colonna 3.
    row = 5
    state.board[row][0] = PLAYER1
    state.board[row][1] = PLAYER1
    state.board[row][2] = PLAYER1
    # colonna 3 vuota
    state.current_player = PLAYER1

    best_move, best_value = minimax_decision(
        state,
        L=1,           # basta 1 ply: valuta solo mosse immediate
        K=None,        # nessun limite di branching
        H=constant_heuristic,
    )

    assert best_move == 3  # deve giocare nella colonna che chiude la riga
    # Dopo apply_move su quella mossa, il risultato deve essere vittoria per P1
    next_state = apply_move(state, best_move)
    assert get_game_result(next_state) == PLAYER1


def test_minimax_blocks_immediate_threat_from_p2():
    state = create_initial_state()
    row = 5
    # PLAYER2 ha tre in fila e minaccia di vincere in colonna 3
    state.board[row][0] = PLAYER2
    state.board[row][1] = PLAYER2
    state.board[row][2] = PLAYER2
    state.current_player = PLAYER1  # tocca a PLAYER1 difendersi

    best_move, best_value = minimax_decision(
        state,
        L=2,           # serve guardare 2 ply: mossa P1, risposta P2
        K=None,
        H=constant_heuristic,
    )

    # MinMax deve giocare in colonna 3 per bloccare la vittoria immediata di P2
    assert best_move == 3


def test_minimax_works_with_limited_branching():
    state = create_initial_state()
    # Board iniziale: molte mosse valide, ma limitiamo K
    best_move, best_value = minimax_decision(
        state,
        L=1,
        K=1,                  # guarda solo 1 mossa figlia
        H=constant_heuristic,
    )

    # La mossa scelta dev'essere comunque una mossa valida
    assert best_move in range(7)


def test_evaluate_state_from_p1_perspective_sign_flip():
    from game.connect_four import ConnectFourState

    # Stato fittizio per testare solo il sign flip
    dummy_board = [[0] * 7 for _ in range(6)]
    state_p1 = ConnectFourState(board=dummy_board, current_player=PLAYER1)
    state_p2 = ConnectFourState(board=dummy_board, current_player=PLAYER2)

    def dummy_H(s):
        # restituisce sempre +0.5 dal punto di vista del giocatore corrente
        return 0.5

    val_p1 = evaluate_state_from_p1_perspective(state_p1, dummy_H)
    val_p2 = evaluate_state_from_p1_perspective(state_p2, dummy_H)

    assert val_p1 == 0.5         # se tocca a P1, valore positivo
    assert val_p2 == -0.5        # se tocca a P2, deve essere negato per prospettiva P1
