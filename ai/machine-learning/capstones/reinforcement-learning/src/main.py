import gymnasium as gym
from domain.entities import AgentConfig
from domain.q_learning_agent import QLearningAgent
from domain.epsilon_greedy_policy import EpsilonGreedyPolicy
from domain.state_discretizer import StateDiscretizer
from application.training_use_case import TrainingUseCase
from application.testing_use_case import TestingUseCase


def main():
    num_episodes = 1000
    config = AgentConfig(
        learning_rate=0.1,
        initial_epsilon=1.0,
        epsilon_decay=1.0 / (num_episodes / 2),
        final_epsilon=0.01,
    )

    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)

    bounds = list(zip(env.observation_space.low, env.observation_space.high))
    # Override unbounded velocities
    bounds[1] = (-4.0, 4.0)  # Cart velocity
    bounds[3] = (-4.0, 4.0)
    discretizer = StateDiscretizer(bounds, config.state_bins)
    policy = EpsilonGreedyPolicy(
        config.initial_epsilon, config.epsilon_decay, config.final_epsilon
    )
    num_actions = env.action_space.n

    agent = QLearningAgent(num_actions, discretizer, policy, config)

    trainer = TrainingUseCase(agent, env)
    trainer.train(num_episodes, log_interval=200)

    tester = TestingUseCase(agent, gym.make("CartPole-v1", render_mode="human"))
    test_metrics = tester.test(num_episodes=5)
    env.close()

    print(f"\nTest Results:")
    print(f"Average Reward: {test_metrics.average_reward:.2f}")
    print(f"Std Dev: {test_metrics.std_deviation:.2f}")
    print(f"Max: {test_metrics.max_reward:.0f}, Min: {test_metrics.min_reward:.0f}")


if __name__ == "__main__":
    main()
