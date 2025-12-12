import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from game.connect_four import (
    create_initial_state,
    print_board,
)
from ai.minimax import minimax_decision


def stupid_H(state) -> float:
    # Valutatore fittizio: sempre 0
    return 0.0


if __name__ == "__main__":
    state = create_initial_state()
    print("Stato iniziale:")
    print_board(state)

    move, value = minimax_decision(state, L=2, K=None, H=stupid_H)
    print(f"Mossa scelta da MinMax(L=2) con H=0: colonna {move}, valore {value}")
