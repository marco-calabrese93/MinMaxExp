from typing import List
import numpy as np
from game.connect_four import ConnectFourState


def state_to_np_array(state: ConnectFourState) -> np.ndarray:
    """
    Converte lo stato in un vettore NumPy di forma (rows * cols,).
    Valori in {-1, 0, +1}.
    """
    rows, cols = state.rows, state.cols
    arr = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            arr[r, c] = state.board[r][c]
    return arr.flatten()
