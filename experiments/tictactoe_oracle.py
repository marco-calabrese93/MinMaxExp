"""
TicTacToe perfect MinMax oracle.

Usa l'ambiente definito in game.tictactoe:
- TicTacToeState
- create_initial_state
- get_valid_moves
- apply_move
- get_game_result
- PLAYER1, PLAYER2
"""

from typing import Tuple, Optional

from game.tictactoe import (
    TicTacToeState,
    create_initial_state,
    get_valid_moves,
    apply_move,
    get_game_result,
    PLAYER1,
    PLAYER2,
)


def oracle_value_from_p1(state: TicTacToeState) -> int:
    """
    Valore PERFETTO dello stato 'state' dal punto di vista di PLAYER1,
    assumendo che entrambi giochino in modo ottimale.

    Restituisce:
        +1 se con gioco perfetto PLAYER1 vince,
         0 se con gioco perfetto è patta,
        -1 se con gioco perfetto PLAYER1 perde.
    """
    result = get_game_result(state)
    if result is not None:
        # get_game_result restituisce già il valore dal punto di vista di PLAYER1
        # (PLAYER1 win = +1, PLAYER2 win = -1, draw = 0)
        return result

    # Non terminale: tocca a state.current_player
    current = state.current_player

    moves = get_valid_moves(state)
    # Per sicurezza, se non ci sono mosse ma lo stato non è segnato terminale,
    # trattiamo come pareggio (non dovrebbe succedere in un TTT ben implementato)
    if not moves:
        return 0

    if current == PLAYER1:
        # PLAYER1 massimizza
        best_val = -2  # meno di -1
        for m in moves:
            next_state = apply_move(state, m)
            v = oracle_value_from_p1(next_state)
            if v > best_val:
                best_val = v
            # pruning opzionale: se troviamo una vittoria, è il max possibile
            if best_val == 1:
                break
        return best_val
    else:
        # PLAYER2 minimizza il valore di PLAYER1
        best_val = +2  # più di +1
        for m in moves:
            next_state = apply_move(state, m)
            v = oracle_value_from_p1(next_state)
            if v < best_val:
                best_val = v
            # pruning opzionale: se troviamo una sconfitta per P1, è il min possibile
            if best_val == -1:
                break
        return best_val


def oracle_best_move(state: TicTacToeState) -> Tuple[int, int]:
    """
    Restituisce la mossa PERFETTA per il giocatore di turno
    e il valore perfetto dal punto di vista di PLAYER1 dopo tale mossa.

    Ritorna:
        (best_move, best_value_from_p1)
    """
    result = get_game_result(state)
    if result is not None:
        raise ValueError("oracle_best_move chiamato su stato terminale")

    moves = get_valid_moves(state)
    if not moves:
        raise ValueError("Nessuna mossa valida disponibile")

    current = state.current_player

    if current == PLAYER1:
        # PLAYER1 sceglie la mossa che massimizza il valore per P1
        best_move = None
        best_value = -2
        for m in moves:
            next_state = apply_move(state, m)
            v = oracle_value_from_p1(next_state)
            if v > best_value:
                best_value = v
                best_move = m
        return best_move, best_value
    else:
        # PLAYER2 sceglie la mossa che MINIMIZZA il valore per P1
        best_move = None
        best_value = +2
        for m in moves:
            next_state = apply_move(state, m)
            v = oracle_value_from_p1(next_state)
            if v < best_value:
                best_value = v
                best_move = m
        return best_move, best_value


def oracle_initial_value() -> int:
    """
    Valore perfetto della posizione iniziale dal punto di vista di PLAYER1.

    Per TicTacToe sappiamo teoricamente che è 0 (patta con gioco perfetto),
    ma qui lo calcoliamo via MinMax completo.
    """
    s0 = create_initial_state()
    return oracle_value_from_p1(s0)


if __name__ == "__main__":
    # Piccolo sanity check manuale
    v0 = oracle_initial_value()
    print("Oracle value of initial state (should be 0 for draw):", v0)
