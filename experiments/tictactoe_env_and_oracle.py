"""
experiments/tictactoe_env_and_oracle.py

- TicTacToeEnv: wrapper OO per il gioco del Tris (3x3)
- minmax_value: MinMax completo (senza cut-off), oracolo perfetto (con cache)
- generate_random_dataset: genera un dataset (X, y) di stati + valore perfetto
- save_dataset_to_csv: salva il dataset in CSV per addestrare un MLP

Esegui dalla root del progetto con:
    python -m experiments.tictactoe_env_and_oracle
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
import csv
import os

from game.tictactoe import (
    TicTacToeState,
    PLAYER1,
    PLAYER2,
    EMPTY,
    create_initial_state,
    get_valid_moves,
    apply_move,
    get_game_result,
)


# ============================================================
# 1. Wrapper ambiente: TicTacToeEnv
# ============================================================

class TicTacToeEnv:
    """
    Wrapper OO, con la stessa API di ConnectFourEnv:

        env.reset()          -> stato iniziale
        env.legal_actions(s) -> lista di mosse valide
        env.next_state(s,a)  -> nuovo stato
        env.is_terminal(s)   -> True/False
        env.outcome(s)       -> z ∈ {+1, 0, -1} (risultato finale)
    """

    def reset(self) -> TicTacToeState:
        return create_initial_state()

    def legal_actions(self, state: TicTacToeState) -> List[int]:
        return get_valid_moves(state)

    def next_state(self, state: TicTacToeState, action: int) -> TicTacToeState:
        return apply_move(state, action)

    def is_terminal(self, state: TicTacToeState) -> bool:
        return get_game_result(state) is not None

    def outcome(self, state: TicTacToeState) -> int:
        res = get_game_result(state)
        if res is None:
            raise ValueError("outcome chiamato su stato non terminale")
        return res


# ============================================================
# 2. Oracolo perfetto: MinMax completo con cache
# ============================================================

# Cache: (current_player, board_flat) -> valore in {-1,0,+1}
_minmax_cache: Dict[Tuple[int, Tuple[int, ...]], int] = {}


def _serialize_state(state: TicTacToeState) -> Tuple[int, Tuple[int, ...]]:
    """
    Trasforma lo stato in una chiave hashable per la cache:
    (current_player, board_flattened)
    """
    flat = []
    for r in range(state.rows):
        for c in range(state.cols):
            flat.append(state.board[r][c])
    return (state.current_player, tuple(flat))


def minmax_value(state: TicTacToeState, maximizing_player: int = PLAYER1) -> int:
    """
    Calcola il valore perfetto dello stato usando MinMax completo (senza cut-off),
    con memoization per evitare ricalcoli inutili.

    Restituisce:
        +1 se, con gioco perfetto, maximizing_player può forzare la vittoria;
        -1 se l'avversario può forzare la vittoria;
         0 se, con gioco perfetto, il risultato è pareggio.

    Nel nostro uso tipico:
        z = minmax_value(state, maximizing_player=PLAYER1)
    """

    # Controllo terminale
    result = get_game_result(state)
    if result is not None:
        if result == 0:
            return 0
        return 1 if result == maximizing_player else -1

    # Chiave per cache (solo board + giocatore di turno).
    # Il maximizing_player lo teniamo fisso nella chiamata iniziale: in TicTacToe
    # la simmetria è tale che è sufficiente.
    key = _serialize_state(state)

    if key in _minmax_cache:
        return _minmax_cache[key]

    moves = get_valid_moves(state)
    if not moves:
        # Nessuna mossa (dovrebbe essere terminale, ma per sicurezza)
        _minmax_cache[key] = 0
        return 0

    # Turno del maximizing_player → massimizziamo
    if state.current_player == maximizing_player:
        best_val = -2
        for m in moves:
            child = apply_move(state, m)
            v = minmax_value(child, maximizing_player)
            if v > best_val:
                best_val = v
                if best_val == 1:
                    break  # non si può fare meglio
        _minmax_cache[key] = best_val
        return best_val

    # Turno dell'avversario → minimizziamo
    else:
        best_val = 2
        for m in moves:
            child = apply_move(state, m)
            v = minmax_value(child, maximizing_player)
            if v < best_val:
                best_val = v
                if best_val == -1:
                    break  # peggio di così non si può
        _minmax_cache[key] = best_val
        return best_val


# ============================================================
# 3. Encoding dello stato (per dataset MLP)
# ============================================================

def encode_state_player1_view(state: TicTacToeState) -> List[int]:
    """
    Codifica lo stato come vettore di 9 interi [-1,0,1],
    dal punto di vista di PLAYER1 (board così com'è).
    """
    flat = []
    for r in range(state.rows):
        for c in range(state.cols):
            flat.append(state.board[r][c])
    return flat


# ============================================================
# 4. Generazione dataset supervisionato
# ============================================================

def generate_random_dataset(
    num_games: int = 100,
    sample_prob: float = 0.7,
    seed: int = 0,
) -> Tuple[List[List[int]], List[int]]:
    """
    Genera un dataset supervisionato (X, y) di stati di Tris + valore perfetto.

    Parametri ridotti (num_games=100) per una prima prova veloce.
    Puoi aumentare a 500/1000 una volta verificato che gira bene.
    """
    random.seed(seed)

    env = TicTacToeEnv()
    X: List[List[int]] = []
    y: List[int] = []

    for _ in range(num_games):
        state = env.reset()

        while not env.is_terminal(state):
            # Decidi se campionare questo stato
            if random.random() < sample_prob:
                x_vec = encode_state_player1_view(state)
                z = minmax_value(state, maximizing_player=PLAYER1)
                X.append(x_vec)
                y.append(z)

            # Mossa casuale per proseguire la partita
            moves = env.legal_actions(state)
            if not moves:
                break
            m = random.choice(moves)
            state = env.next_state(state, m)

    return X, y


def save_dataset_to_csv(
    X: List[List[int]],
    y: List[int],
    filepath: str = "results/tictactoe_dataset_player1.csv",
):
    """
    Salva il dataset (X, y) in un CSV con colonne x0..x8, z.
    Crea la cartella se non esiste.
    """
    if len(X) != len(y):
        raise ValueError("X e y devono avere la stessa lunghezza")

    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fieldnames = [f"x{i}" for i in range(9)] + ["z"]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for x_vec, z in zip(X, y):
            row = {f"x{i}": int(x_vec[i]) for i in range(9)}
            row["z"] = int(z)
            writer.writerow(row)

    print(f"Dataset salvato in: {filepath} (num esempi: {len(X)})")


# ============================================================
# 5. Test veloci e main
# ============================================================

def _quick_sanity_checks():
    """
    Piccoli test per verificare che l'oracolo si comporti come atteso.
    """
    env = TicTacToeEnv()

    # Stato iniziale: con gioco perfetto il risultato è sempre pareggio (0)
    s0 = env.reset()
    v0 = minmax_value(s0, maximizing_player=PLAYER1)
    print(f"Valore stato iniziale (dovrebbe essere 0): {v0}")


if __name__ == "__main__":
    print("Eseguo sanity check sull'oracolo MinMax per TicTacToe...")
    _quick_sanity_checks()

    print("Genero dataset casuale di stati di TicTacToe...")
    X, y = generate_random_dataset(num_games=500, sample_prob=0.7, seed=0)
    save_dataset_to_csv(X, y, filepath="results/tictactoe_dataset_player1.csv")
