from collections import defaultdict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# This is an example of how to train an agent using gymnasium. For more info:
# - https://gymnasium.farama.org/introduction/basic_usage/
# - https://gymnasium.farama.org/introduction/train_agent/
class CartPoleTrain:
    def __init__(self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.999,
        bins: tuple[int, int, int, int] = (6, 6, 6, 6),
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.bins = bins

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.training_error = []

        # Define bounds for CartPole state space discretization
        # CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        self.state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
        # CartPole has unbounded velocity values, so we set reasonable bounds
        self.state_bounds[1] = (-4.0, 4.0)  # Cart velocity
        self.state_bounds[3] = (-4.0, 4.0)  # Pole angular velocity

    def discretize_state(self, state: np.ndarray) -> tuple[int, int, int, int]:
        """Convert continuous state to discrete state bins"""
        discrete_state = []
        for i, (low, high) in enumerate(self.state_bounds):
            # Clip the state value to bounds
            state_value = np.clip(state[i], low, high)
            # Scale to [0, bins[i]-1]
            scaled = int(np.round((state_value - low) / (high - low) * (self.bins[i] - 1)))
            discrete_state.append(scaled)
        return tuple(discrete_state)

    def get_action(self, obs: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            discrete_obs = self.discretize_state(obs)
            return int(np.argmax(self.q_values[discrete_obs]))
        
    def update(self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray
    ):
        """Update Q-values using the Q-learning update rule"""
        # Discretize continuous states
        discrete_obs = self.discretize_state(obs)
        discrete_next_obs = self.discretize_state(next_obs)

        future_q_value = (not terminated) * np.max(self.q_values[discrete_next_obs])
        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value
        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[discrete_obs][action]
        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[discrete_obs][action] = (
            self.q_values[discrete_obs][action] + self.learning_rate * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def reduce_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def train_model(episodes=1000):
    learning_rate = 0.1        # How fast to learn (higher = faster but less stable)
    n_episodes = episodes       # Number of episodes to train
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.01        # Always keep some exploration

    # Create environment and agent
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = CartPoleTrain(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    rewards_per_episode = []

    for episode in range(n_episodes):
        # Start a new episode
        obs, info = env.reset()
        done = False
        total_reward = 0

        # Play one complete episode
        while not done:
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent.get_action(obs)

            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            # Move to next state
            done = terminated or truncated
            obs = next_obs
            total_reward += reward

        rewards_per_episode.append(total_reward)

        # Reduce exploration rate (agent becomes less random over time)
        agent.reduce_epsilon()

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{n_episodes}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    env.close()

    print("\nTraining completed!")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards_per_episode[-100:]):.2f}")

    return agent, rewards_per_episode

# Test the trained agent
def test_agent(agent, num_episodes=10, render=True):
    """Test agent performance without learning or exploration."""
    # Create a new environment for testing
    env = gym.make("CartPole-v1", render_mode='human' if render else None)
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}: Reward = {episode_reward}")

    # Restore original epsilon
    agent.epsilon = old_epsilon

    average_reward = np.mean(total_rewards)

    print(f"\nTest Results over {num_episodes} episodes:")
    print(f"Average Reward: {average_reward:.2f}")
    print(f"Standard Deviation: {np.std(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.0f}")
    print(f"Min Reward: {np.min(total_rewards):.0f}")

    env.close()
    return total_rewards


def plot_training_progress(rewards):
    """Plot training progress"""
    plt.figure(figsize=(12, 5))

    # Plot 1: Rewards per episode
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.grid(True)

    # Plot 2: Moving average
    plt.subplot(1, 2, 2)
    window_size = 100
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Moving Average Reward (window={window_size})')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('cartpole_training.png')
    print("Training plot saved as 'cartpole_training.png'")
    plt.show()


if __name__ == "__main__":
    # Train the agent
    print("Training CartPole Q-Learning Agent...")
    agent, rewards = train_model(episodes=1000)

    # Plot training progress
    plot_training_progress(rewards)

    # Test the trained agent
    print("\nTesting trained agent...")
    test_agent(agent, num_episodes=5, render=True)
