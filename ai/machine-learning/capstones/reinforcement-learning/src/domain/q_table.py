import numpy as np
from collections import defaultdict
from .entities import State


class QTable:
    def __init__(self, num_actions: int):
        self._table = defaultdict(lambda: np.zeros(num_actions))

    def get_values(self, state: State) -> np.ndarray:
        return self._table[state]

    def update_value(self, state: State, action: int, new_value: float) -> np.ndarray:
        self._table[state][action] = new_value
        return self._table[state][action]

    def get_best_action(self, state: State) -> int:
        return int(np.argmax(self._table[state]))
