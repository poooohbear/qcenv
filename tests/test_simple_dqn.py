from qcenv.agents import DQNAgent
from qcenv.qnets import QNet
from qcenv.replay_buffers import ReplayBuffer

import gymnasium as gym
import torch
import numpy as np


def test_simple_dqn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    episodes = 200
    sync_interval = 20

    q_function = QNet(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(q_function.parameters(), lr=1e-2)
    replay_buffer = ReplayBuffer(10000, 32)
    agent = DQNAgent(q_function, optimizer, replay_buffer, 0.99, 0.1, device)
    rewards = []
    for episode in range(episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        episode_reward = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            done = terminated or truncated
            done = torch.tensor(done, dtype=torch.int32)
            agent.update(state, action, reward, done, next_state)
            state = next_state
            episode_reward += reward
        rewards.append(episode_reward)
        if episode % sync_interval == 0:
            if episode == 0:
                continue
            agent.sync_target()

    rewards = np.array(rewards)
    assert rewards[-20:].mean() > 50
