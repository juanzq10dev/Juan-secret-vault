from dataclasses import dataclass
import numpy as np


@dataclass
class State:
    values: tuple[int, ...]

    def __hash__(self):
        return hash(self.values)


@dataclass
class AgentConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.999
    initial_epsilon: float = 1.0
    epsilon_decay: float = 0.001
    final_epsilon: float = 0.01
    state_bins: tuple[int, ...] = (6, 6, 6, 6)


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
