import gymnasium as gym
import numpy as np
from domain.q_learning_agent import QLearningAgent
from domain.entities import Experience
from .models import TrainingMetrics


class TrainingUseCase:
    def __init__(self, agent: QLearningAgent, environment: gym.Env) -> None:
        self._agent = agent
        self._env = environment

    def train(self, num_episodes: int, log_interval: int) -> TrainingMetrics:
        rewards_per_episode = []

        for episode in range(num_episodes):
            episode_reward = self._run_episode()
            rewards_per_episode.append(episode_reward)
            self._agent.update_policy()

            if (episode + 1) % log_interval == 0:
                self._log_progress(episode + 1, num_episodes, rewards_per_episode)

        final_avg = np.mean(rewards_per_episode[-100:])
        print(f"\nTraining completed! Final average: {final_avg:.2f}")

        return TrainingMetrics(
            rewards_per_episode=rewards_per_episode, final_avg_reward=final_avg
        )

    def _run_episode(self) -> float:
        """Execute one training episode."""
        observation, _ = self._env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = self._agent.select_action(observation)
            next_obs, reward, terminated, truncated, _ = self._env.step(action)

            experience = Experience(
                state=observation,
                action=action,
                reward=reward,
                next_state=next_obs,
                terminated=terminated,
            )
            self._agent.learn(experience)

            observation = next_obs
            total_reward += reward
            done = terminated or truncated

        return total_reward

    def _log_progress(self, episode: int, total: int, rewards: list[float]):
        """Log training progress."""
        avg_reward = np.mean(rewards[-100:])
        epsilon = self._agent._policy._epsilon
        print(
            f"Episode {episode}/{total}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}"
        )
