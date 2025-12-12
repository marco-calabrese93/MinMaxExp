from dataclasses import dataclass
from typing import List, Optional
import copy

PLAYER1 = 1   # Giocatore 1 (X)
PLAYER2 = -1  # Giocatore 2 (O)
EMPTY   = 0

DEFAULT_ROWS = 6
DEFAULT_COLS = 7


@dataclass
class ConnectFourState:
    board: List[List[int]]
    current_player: int  # PLAYER1 o PLAYER2

    @property
    def rows(self) -> int:
        return len(self.board)

    @property
    def cols(self) -> int:
        return len(self.board[0])


def create_initial_state(rows: int = DEFAULT_ROWS,
                         cols: int = DEFAULT_COLS) -> ConnectFourState:
    board = [[EMPTY for _ in range(cols)] for _ in range(rows)]
    return ConnectFourState(board=board, current_player=PLAYER1)


def get_valid_moves(state: ConnectFourState) -> List[int]:
    """Restituisce la lista delle colonne in cui è possibile giocare."""
    valid = []
    for c in range(state.cols):
        if state.board[0][c] == EMPTY:
            valid.append(c)
    return valid


def apply_move(state: ConnectFourState, col: int) -> ConnectFourState:
    """Ritorna un NUOVO stato dopo aver giocato nella colonna `col`."""
    if col not in get_valid_moves(state):
        raise ValueError(f"Mossa non valida: colonna {col}")

    new_state = copy.deepcopy(state)
    player = state.current_player

    # Scendo dal basso verso l'alto per trovare la prima cella vuota
    for r in range(state.rows - 1, -1, -1):
        if new_state.board[r][col] == EMPTY:
            new_state.board[r][col] = player
            break

    # Cambio giocatore
    new_state.current_player = PLAYER1 if player == PLAYER2 else PLAYER2
    return new_state


def check_winner_for_player(state: ConnectFourState, player: int) -> bool:
    """Controlla se `player` ha 4 in fila."""
    rows, cols = state.rows, state.cols
    b = state.board

    # Orizzontale
    for r in range(rows):
        for c in range(cols - 3):
            if (b[r][c] == player and b[r][c+1] == player and
                b[r][c+2] == player and b[r][c+3] == player):
                return True

    # Verticale
    for c in range(cols):
        for r in range(rows - 3):
            if (b[r][c] == player and b[r+1][c] == player and
                b[r+2][c] == player and b[r+3][c] == player):
                return True

    # Diagonale "\"
    for r in range(rows - 3):
        for c in range(cols - 3):
            if (b[r][c] == player and b[r+1][c+1] == player and
                b[r+2][c+2] == player and b[r+3][c+3] == player):
                return True

    # Diagonale "/"
    for r in range(3, rows):
        for c in range(cols - 3):
            if (b[r][c] == player and b[r-1][c+1] == player and
                b[r-2][c+2] == player and b[r-3][c+3] == player):
                return True

    return False


def get_game_result(state: ConnectFourState) -> Optional[int]:
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

    # Board piena?
    if all(state.board[0][c] != EMPTY for c in range(state.cols)):
        return 0  # pareggio

    return None  # partita in corso


def print_board(state: ConnectFourState) -> None:
    """Stampa la board in formato leggibile."""
    symbols = {
        EMPTY: ".",
        PLAYER1: "X",
        PLAYER2: "O",
    }
    for r in range(state.rows):
        row_str = " ".join(symbols[state.board[r][c]] for c in range(state.cols))
        print(row_str)
    print("-" * (2 * state.cols - 1))
    print(" ".join(str(c) for c in range(state.cols)))
    print()


if __name__ == "__main__":
    # Debug: partita tra 2 umani in console
    state = create_initial_state()
    print("Inizio partita Connect Four!")
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
        print(f"Mosse valide: {valid}")

        try:
            col = int(input("Scegli una colonna: "))
        except ValueError:
            print("Inserisci un numero valido.\n")
            continue

        if col not in valid:
            print("Mossa non valida.\n")
            continue

        state = apply_move(state, col)
        print_board(state)
