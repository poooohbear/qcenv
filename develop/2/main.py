from qcenv.agents import DQNAgent

# from qcenv.utils import QNet
from qcenv.replay_buffers import ReplayBuffer
from easy_test_env import *

import gymnasium as gym
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.functional import F
from datetime import datetime
import os
import sys
from torchinfo import summary


class EasyTestEnvQNet(torch.nn.Module):
    # Denseでもやってみる
    # 他にも確認
    def __init__(self):
        super().__init__()
        seq_length = 5
        one_hot_category_size = 9
        # 入力サイズは(1, seq_length, one_hot_category_size)で、出力サイズは(1, one_hot_category_size)になるようにする
        self.observation_size = (seq_length, one_hot_category_size)
        self.action_size = one_hot_category_size
        middle_size = 27
        kernel_size = (seq_length - 1) // 2 + 1
        # nn.Conv1d→ReLu→Conv1d→Flatten→Fully-connected→softmax
        self.conv1 = torch.nn.Conv1d(
            in_channels=one_hot_category_size,
            out_channels=middle_size,
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=middle_size,
            out_channels=middle_size,
            kernel_size=kernel_size,
            stride=1,
        )
        self.fc1 = torch.nn.Linear(middle_size, one_hot_category_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        return F.softmax(self.fc1(x), dim=1)


def to_one_hot(data_point, n_action=5):
    one_hot = torch.zeros(n_action)
    one_hot[data_point] = 1
    return one_hot


if __name__ == "__main__":
    # qnet = EasyTestEnvQNet()
    # summary(qnet, (1, 9, 5))
    # sys.exit()

    current_time = datetime.now().strftime("%H%M")
    dir_name = f"results/{current_time}"
    os.makedirs(dir_name, exist_ok=True)
    log_file_name = dir_name + "/log.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    # env = gym.make("CartPole-v1")
    env = EasyTestEnv()
    episodes = 20000
    sync_interval = 20
    reward_histories = []
    num_tests = 3

    with open(log_file_name, "w") as f:
        f.write(f"Using {device}\n")
        f.write(f"episodes: {episodes}\n")
        f.write(f"sync_interval: {sync_interval}\n")
        f.write(f"num_tests: {num_tests}\n")

    for test in tqdm(range(num_tests)):
        q_function = EasyTestEnvQNet()
        optimizer = torch.optim.Adam(q_function.parameters(), lr=1e-2)
        replay_buffer = ReplayBuffer(1000, 32)
        agent = DQNAgent(q_function, optimizer, replay_buffer, 0.99, 0.1, device)
        rewards = []
        action_lengths = []
        for episode in tqdm(range(episodes)):
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            # state = torch.t(state)
            done = False
            episode_reward = 0
            action_length = 0
            while not done:
                action = agent.get_action(state)
                action_length += 1
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                # next_state = torch.t(next_state)
                reward = torch.tensor(reward, dtype=torch.float32)
                done = terminated or truncated
                done = torch.tensor(done, dtype=torch.float32)
                agent.update(state, action, reward, done, next_state)
                state = next_state
                episode_reward += reward
            action_lengths.append(action_length)
            rewards.append(episode_reward)
            if episode % sync_interval == 0:
                if episode == 0:
                    continue
                agent.sync_target()
        reward_histories.append(rewards)

    reward_histories = np.array(reward_histories)
    mean_reward_history = reward_histories.mean(axis=0)
    plt.plot(mean_reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(dir_name + "/easy_test_env.png")
    plt.close()

    action_lengths = np.array(action_lengths)
    mean_action_length = action_lengths.mean(axis=0)
    plt.plot(mean_action_length)
    plt.xlabel("Episode")
    plt.ylabel("Action Length")
    plt.savefig(dir_name + "/easy_test_env_action_length.png")
    plt.close()
