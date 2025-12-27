from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    """Results from training session."""

    rewards_per_episode: list[float]
    final_avg_reward: float


@dataclass
class TestMetrics:
    """Results from testing session."""

    episode_rewards: list[float]
    average_reward: float
    std_deviation: float
    max_reward: float
    min_reward: float
