from .q_table import QTable
from .entities import State
import numpy as np


class EpsilonGreedyPolicy:
    def __init__(self, initial_epsilon: float, decay: float, final_epsilon: float):
        self._epsilon = initial_epsilon
        self._decay = decay
        self._final_epsilon = final_epsilon

    def select_action(self, q_table: QTable, state: State, num_actions: int) -> int:
        if np.random.random() < self._epsilon:
            return np.random.randint(num_actions)
        return q_table.get_best_action(state)

    def decay(self):
        self._epsilon = max(self._final_epsilon, self._epsilon - self._decay)
