from typing import List, Dict
import matplotlib.pyplot as plt


def plot_loss(history: List[Dict]):
    iters = [h["iteration"] for h in history]
    losses = [h["loss"] for h in history]
    plt.figure()
    plt.plot(iters, losses, marker="o")
    plt.xlabel("Iterazione")
    plt.ylabel("Loss MSE")
    plt.title("Andamento loss durante il training")
    plt.grid(True)
    plt.show()
