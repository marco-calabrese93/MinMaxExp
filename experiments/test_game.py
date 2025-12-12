import os
import sys

# Aggiunge automaticamente la root del progetto al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from game.connect_four import (
    create_initial_state,
    print_board,
    get_valid_moves,
    apply_move,
    get_game_result,
    PLAYER1,
    PLAYER2,
)

# inizializza board
state = create_initial_state()
print("Board iniziale:")
print_board(state)

# fai alcune mosse a mano
moves = [3, 3, 2, 2, 1, 1, 0]  # esempio: sequenza qualsiasi
for col in moves:
    print(f"\nGioco in colonna {col}")
    state = apply_move(state, col)
    print_board(state)

    result = get_game_result(state)
    if result is not None:
        if result == PLAYER1:
            print("Ha vinto PLAYER1")
        elif result == PLAYER2:
            print("Ha vinto PLAYER2")
        else:
            print("Pareggio")
        break
