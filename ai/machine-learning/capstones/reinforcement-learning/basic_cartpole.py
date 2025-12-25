import gymnasium as gym

# Create a CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Reset the environment to start a new episode
observation, info = env.reset()

episode_over = False
total_reward = 0

while not episode_over:
    # Choose a random action
    action = env.action_space.sample()

    # Take the action and get results
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
