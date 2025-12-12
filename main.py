from ai.training import training_loop


def L_schedule(t: int) -> int:
    """
    Strategia semplice:
    - profondità 1 per le prime 10 iterazioni
    - profondità 2 per le successive 10
    - max 3 dopo.
    """
    if t < 10:
        return 1
    elif t < 20:
        return 2
    else:
        return 3


def K_schedule(t: int) -> int:
    """
    Strategia semplice per K:
    - esplora fino a 7 mosse (tutto) all'inizio
    - poi riduce leggermente (per esempio, a 5).
    """
    if t < 20:
        return 7
    else:
        return 5


if __name__ == "__main__":
    # Attenzione: servirà parecchio tempo se num_iterations è grande.
    model, history = training_loop(
        num_iterations=5,
        L_schedule=L_schedule,
        K_schedule=K_schedule,
        device="cpu",
    )

    # Se vuoi, puoi salvare il modello e le metriche qui.
