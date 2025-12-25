import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQNNetwork(nn.Module):
    """Deep Q-Network for CartPole"""
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """DQN Agent for reinforcement learning"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().numpy())

    def replay(self, batch_size):
        """Train on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state)
            target_f = target_f.clone()
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename):
        """Load the model"""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


def train_cartpole(episodes=1000, render=False):
    """Train the DQN agent on CartPole"""
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    scores = []

    for episode in range(episodes):
        state, _ = env.reset()
        score = 0

        for time_step in range(500):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

            agent.replay(batch_size)

        scores.append(score)
        avg_score = np.mean(scores[-100:])

        print(f"Episode: {episode+1}/{episodes}, Score: {score:.0f}, "
              f"Avg Score (last 100): {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}")

        # Save model if performance is good
        if avg_score >= 475 and episode >= 100:
            agent.save('cartpole_dqn_model.pth')
            print(f"Model saved! Average score: {avg_score:.2f}")
            break

    env.close()

    # Save final model
    agent.save('cartpole_dqn_model.pth')
    print("Training completed!")

    return agent, scores


if __name__ == "__main__":
    print("Training CartPole with DQN...")
    agent, scores = train_cartpole(episodes=1000)
    print(f"Final average score (last 100 episodes): {np.mean(scores[-100:]):.2f}")
