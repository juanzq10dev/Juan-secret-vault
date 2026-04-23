import numpy as np
from .entities import State


class StateDiscretizer:
    """
    Helper class to discretize continuous state values. (Done by Claude Code with sonnet-4.5)
    """

    def __init__(self, bounds: list[tuple[float, float]], bins: tuple[int, ...]):
        self._bounds = bounds
        self._bins = bins

    def discretize(self, continuous_state: np.ndarray) -> State:
        discrete_values = []
        for i, (low, high) in enumerate(self._bounds):
            clipped = np.clip(continuous_state[i], low, high)
            scaled = int(np.round((clipped - low) / (high - low) * (self._bins[i] - 1)))
            discrete_values.append(scaled)
        return State(tuple(discrete_values))
