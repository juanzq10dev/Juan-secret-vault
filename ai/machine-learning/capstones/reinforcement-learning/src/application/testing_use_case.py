import gymnasium as gym
import numpy as np
from domain.q_learning_agent import QLearningAgent
from .models import TestMetrics


class TestingUseCase:
    def __init__(self, agent: QLearningAgent, environment: gym.Env):
        self._agent = agent
        self._env = environment

    def test(self, num_episodes: int, exploit_only: bool = True) -> TestMetrics:
        """Test agent performance."""
        old_epsilon = self._agent._policy._epsilon

        if exploit_only:
            # Temporarily disable exploration
            self._agent._policy._epsilon = 0.0

        episode_rewards = []
        for episode in range(num_episodes):
            reward = self._run_episode()
            episode_rewards.append(reward)
            print(f"Test Episode {episode + 1}: Reward = {reward}")

        # Restore exploration rate
        if exploit_only:
            self._agent._policy._epsilon = old_epsilon

        return TestMetrics(
            episode_rewards=episode_rewards,
            average_reward=np.mean(episode_rewards),
            std_deviation=np.std(episode_rewards),
            max_reward=np.max(episode_rewards),
            min_reward=np.min(episode_rewards),
        )

    def _run_episode(self) -> float:
        """Execute one test episode."""
        observation, _ = self._env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = self._agent.select_action(observation)
            observation, reward, terminated, truncated, _ = self._env.step(action)
            total_reward += reward
            done = terminated or truncated

        return total_reward
