import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add_episode(self, states, z):
        """
        Aggiunge tutti gli stati di una partita, ognuno con target z.
        """
        for s in states:
            self.buffer.append((s, z))

    def sample(self, batch_size=64):
        """
        Restituisce una lista random di (state, z).
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, zs = zip(*batch)
        return list(states), list(zs)

    def __len__(self):
        return len(self.buffer)
