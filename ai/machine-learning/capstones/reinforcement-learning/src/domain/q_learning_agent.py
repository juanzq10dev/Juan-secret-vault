from .state_discretizer import StateDiscretizer
from .epsilon_greedy_policy import EpsilonGreedyPolicy
from .q_table import QTable
from .entities import AgentConfig, Experience
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        num_actions: int,
        discretizer: StateDiscretizer,
        policy: EpsilonGreedyPolicy,
        config: AgentConfig,
    ):
        self._num_actions = num_actions
        self._discretizer = discretizer
        self._policy = policy
        self._config = config
        self._q_table = QTable(num_actions)
        self._training_errors: list[float] = []

    def select_action(self, observation: np.ndarray) -> int:
        state = self._discretizer.discretize(observation)
        return self._policy.select_action(self._q_table, state, self._num_actions)

    def learn(self, experience: Experience) -> None:
        state = self._discretizer.discretize(experience.state)
        next_state = self._discretizer.discretize(experience.next_state)

        # Calculate TD target
        future_q_value = (
            0.0
            if experience.terminated
            else np.max(self._q_table.get_values(next_state))
        )
        target = experience.reward + self._config.discount_factor * future_q_value

        # Calculate TD error
        current_value = self._q_table.get_values(state)[experience.action]
        td_error = target - current_value

        # Update Q-value
        new_value = current_value + self._config.learning_rate * td_error
        self._q_table.update_value(state, experience.action, new_value)

        self._training_errors.append(td_error)

    def update_policy(self) -> None:
        self._policy.decay()
