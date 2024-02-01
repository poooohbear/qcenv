from qcenv.agents import DQNAgent
from qcenv.replay_buffers import ReplayBuffer
from qcenv.environments import EasyTestEnv
from qcenv.qnets import Conv1dQNet

import torch
import numpy as np


def test_conv1d_qnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = EasyTestEnv()
    episodes = 1000
    sync_interval = 20

    q_function = Conv1dQNet()
    optimizer = torch.optim.Adam(q_function.parameters(), lr=1e-2)
    replay_buffer = ReplayBuffer(1000, 32)
    agent = DQNAgent(q_function, optimizer, replay_buffer, 0.99, 0.1, device)

    for episode in range(episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        episode_reward = 0
        action_length = 0
        while not done:
            action = agent.get_action(state)
            action_length += 1
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            done = terminated or truncated
            done = torch.tensor(done, dtype=torch.float32)
            agent.update(state, action, reward, done, next_state)
            state = next_state
            episode_reward += reward
        if episode % sync_interval == 0:
            if episode == 0:
                continue
            agent.sync_target()

    # evaluation
    ntest = 10
    action_lengths = []
    episode_rewards = []
    for test in range(ntest):
        print(f"test: {test}")
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        episode_reward = 0
        action_length = 0
        while not done:
            action = agent.get_action(state, is_eval=True)
            print(f"action length: {action_length}")
            print(f"action: {action}")
            print()
            action_length += 1
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            # next_state = torch.t(next_state)
            reward = torch.tensor(reward, dtype=torch.float32)
            done = terminated or truncated
            done = torch.tensor(done, dtype=torch.float32)
            state = next_state
            episode_reward += reward
        action_lengths.append(action_length)
        episode_rewards.append(episode_reward)
    action_lengths = np.array(action_lengths)
    mean_action_length = action_lengths.mean(axis=0)
    assert mean_action_length <= 4.0

    episode_rewards = np.array(episode_rewards)
    mean_episode_reward = episode_rewards.mean(axis=0)
    assert mean_episode_reward > 0.5
