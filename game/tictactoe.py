from dataclasses import dataclass
from typing import List, Optional
import copy

# Costanti uguali a quelle di connect_four.py
PLAYER1 = 1    # Giocatore 1 (X)
PLAYER2 = -1   # Giocatore 2 (O)
EMPTY   = 0

DEFAULT_SIZE = 3  # Tris 3x3


@dataclass
class TicTacToeState:
    board: List[List[int]]
    current_player: int  # PLAYER1 o PLAYER2

    @property
    def rows(self) -> int:
        return len(self.board)

    @property
    def cols(self) -> int:
        return len(self.board[0])


def create_initial_state(size: int = DEFAULT_SIZE) -> TicTacToeState:
    """
    Crea lo stato iniziale di un Tris size x size (di default 3x3).
    """
    board = [[EMPTY for _ in range(size)] for _ in range(size)]
    return TicTacToeState(board=board, current_player=PLAYER1)


def get_valid_moves(state: TicTacToeState) -> List[int]:
    """
    Restituisce la lista delle mosse valide.

    Qui codifichiamo una mossa come un intero da 0 a (rows*cols - 1),
    dove:
        r = idx // cols
        c = idx % cols

    Una mossa è valida se la cella corrispondente è vuota.
    """
    valid = []
    for r in range(state.rows):
        for c in range(state.cols):
            if state.board[r][c] == EMPTY:
                idx = r * state.cols + c
                valid.append(idx)
    return valid


def index_to_rc(state: TicTacToeState, move: int) -> (int, int):
    """
    Converte un indice di mossa (0..rows*cols-1) in coordinate (r, c).
    """
    r = move // state.cols
    c = move % state.cols
    return r, c


def apply_move(state: TicTacToeState, move: int) -> TicTacToeState:
    """
    Applica la mossa `move` (indice 0..8) e restituisce un NUOVO stato.

    Lancia ValueError se la mossa non è valida.
    """
    if move not in get_valid_moves(state):
        raise ValueError(f"Mossa non valida: {move}")

    new_state = copy.deepcopy(state)
    r, c = index_to_rc(state, move)
    new_state.board[r][c] = state.current_player

    # Cambia giocatore
    if state.current_player == PLAYER1:
        new_state.current_player = PLAYER2
    else:
        new_state.current_player = PLAYER1

    return new_state


def check_winner_for_player(state: TicTacToeState, player: int) -> bool:
    """
    Controlla se `player` ha vinto (tre in fila su riga, colonna o diagonale).
    """
    n = state.rows
    b = state.board

    # Righe
    for r in range(n):
        if all(b[r][c] == player for c in range(n)):
            return True

    # Colonne
    for c in range(n):
        if all(b[r][c] == player for r in range(n)):
            return True

    # Diagonale principale "\"
    if all(b[i][i] == player for i in range(n)):
        return True

    # Diagonale secondaria "/"
    if all(b[i][n - 1 - i] == player for i in range(n)):
        return True

    return False


def get_game_result(state: TicTacToeState) -> Optional[int]:
    """
    Ritorna:
      +1 se vince PLAYER1,
      -1 se vince PLAYER2,
       0 se pareggio,
      None se la partita non è terminata.
    """
    if check_winner_for_player(state, PLAYER1):
        return PLAYER1
    if check_winner_for_player(state, PLAYER2):
        return PLAYER2

    # C'è ancora almeno una cella vuota?
    for r in range(state.rows):
        for c in range(state.cols):
            if state.board[r][c] == EMPTY:
                return None  # partita in corso

    # Nessun vincitore e nessuna cella vuota → pareggio
    return 0


def print_board(state: TicTacToeState) -> None:
    """
    Stampa la board in formato leggibile.
    """
    symbols = {
        EMPTY: ".",
        PLAYER1: "X",
        PLAYER2: "O",
    }
    for r in range(state.rows):
        row_str = " ".join(symbols[state.board[r][c]] for c in range(state.cols))
        print(row_str)
    print()


if __name__ == "__main__":
    # Debug: partita tra 2 umani in console (facoltativo)
    state = create_initial_state()
    print("Inizio partita Tic-Tac-Toe!")
    print_board(state)

    while True:
        result = get_game_result(state)
        if result is not None:
            if result == PLAYER1:
                print("Ha vinto il Giocatore 1 (X)!")
            elif result == PLAYER2:
                print("Ha vinto il Giocatore 2 (O)!")
            else:
                print("Pareggio!")
            break

        print(f"Tocca al giocatore {'1 (X)' if state.current_player == PLAYER1 else '2 (O)'}")
        valid = get_valid_moves(state)
        print(f"Mosse valide (0..8): {valid}")

        try:
            move = int(input("Scegli una cella (0..8): "))
        except ValueError:
            print("Inserisci un numero valido.\n")
            continue

        if move not in valid:
            print("Mossa non valida.\n")
            continue

        state = apply_move(state, move)
        print_board(state)
