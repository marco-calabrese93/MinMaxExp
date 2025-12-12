from typing import Callable, Tuple, Optional, List
from game.connect_four import (
    ConnectFourState,
    get_valid_moves,
    apply_move,
    get_game_result,
    PLAYER1,
    PLAYER2,
)

# Tipo: funzione di valutazione H_true(state) -> float
HeuristicFn = Callable[[ConnectFourState], float]


def evaluate_state_from_p1_perspective(state: ConnectFourState,
                                       H: HeuristicFn) -> float:
    """
    Valuta lo stato dal punto di vista del Giocatore 1.
    - Se lo stato è terminale, ritorna z direttamente.
    - Altrimenti usa H(state), assumendo che H predica l'esito
      dal punto di vista del giocatore corrente.
    """
    result = get_game_result(state)
    if result is not None:
        # z è già in {-1,0,+1} dal punto di vista di PLAYER1
        return float(result)

    # Stato non terminale: usiamo H
    # Supponiamo che H restituisca un valore dal punto di vista del player corrente.
    h_val = H(state)
    if state.current_player == PLAYER1:
        return h_val
    else:
        # Se il turno è di PLAYER2, invertiamo il segno per avere la prospettiva di P1
        return -h_val


def minimax_value(state: ConnectFourState,
                  depth: int,
                  K: Optional[int],
                  H: HeuristicFn) -> float:
    """
    Restituisce il valore MinMax dello stato dal punto di vista di PLAYER1,
    con profondità limitata e ampiezza (branching) limitata a K mosse.
    """
    result = get_game_result(state)
    if result is not None:
        # Terminale: risultato esatto
        return float(result)

    if depth == 0:
        # Limite di profondità: valutazione euristica
        return evaluate_state_from_p1_perspective(state, H)

    valid_moves = get_valid_moves(state)
    if not valid_moves:
        # Nessuna mossa valida, trattiamo come pareggio
        return 0.0

    # Applica limite K all'ampiezza (se specificato)
    if K is not None and len(valid_moves) > K:
        valid_moves = valid_moves[:K]  # semplice: prendo le prime K

    # MinMax:
    # - se tocca a PLAYER1: massimizza
    # - se tocca a PLAYER2: minimizza (equivalente a massimizzare -valore avversario)
    if state.current_player == PLAYER1:
        best_value = float("-inf")
        for move in valid_moves:
            child = apply_move(state, move)
            val = minimax_value(child, depth - 1, K, H)
            if val > best_value:
                best_value = val
        return best_value
    else:
        best_value = float("inf")
        for move in valid_moves:
            child = apply_move(state, move)
            val = minimax_value(child, depth - 1, K, H)
            if val < best_value:
                best_value = val
        return best_value


def minimax_decision(state: ConnectFourState,
                     L: int,
                     K: Optional[int],
                     H: HeuristicFn) -> Tuple[int, float]:
    """
    Sceglie la mossa migliore dallo stato corrente usando MinMax
    con profondità L e ampiezza K, usando H come valutatore di foglia.

    Ritorna (best_move, best_value).
    """
    valid_moves = get_valid_moves(state)
    if not valid_moves:
        raise ValueError("Nessuna mossa valida disponibile.")

    best_move = valid_moves[0]
    # Dal punto di vista di PLAYER1
    if state.current_player == PLAYER1:
        best_value = float("-inf")
        for move in valid_moves:
            child = apply_move(state, move)
            val = minimax_value(child, L - 1, K, H)
            if val > best_value:
                best_value = val
                best_move = move
    else:
        # Se tocca a PLAYER2, scegliamo la mossa che minimizza il valore per PLAYER1
        best_value = float("inf")
        for move in valid_moves:
            child = apply_move(state, move)
            val = minimax_value(child, L - 1, K, H)
            if val < best_value:
                best_value = val
                best_move = move

    return best_move, best_value
