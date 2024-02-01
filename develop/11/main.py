from qcenv.agents import DQNAgent

# from qcenv.utils import QNet
from qcenv.replay_buffers import ReplayBuffer
from qcenv.environments import EasyTestEnv
from qcenv.qnets import Conv1dQNet

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


def to_one_hot(data_point, n_action=5):
    one_hot = torch.zeros(n_action)
    one_hot[data_point] = 1
    return one_hot


if __name__ == "__main__":
    current_time = datetime.now().strftime("%m%d%H%M")
    dir_name = f"results/{current_time}"
    os.makedirs(dir_name, exist_ok=True)
    log_file_name = dir_name + "/log.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    env = EasyTestEnv(reward_type="Bhattacharyya")
    episodes = 400
    sync_interval = 20
    reward_histories = []
    all_action_lengths = []
    num_tests = 5

    with open(log_file_name, "w") as f:
        f.write(f"Using {device}\n")
        f.write(f"episodes: {episodes}\n")
        f.write(f"sync_interval: {sync_interval}\n")
        f.write(f"num_tests: {num_tests}\n")

    for test in tqdm(range(num_tests)):
        q_function = Conv1dQNet()
        optimizer = torch.optim.Adam(q_function.parameters(), lr=1e-2)
        replay_buffer = ReplayBuffer(1000, 32)
        agent = DQNAgent(q_function, optimizer, replay_buffer, 0.99, 0.1, device)
        rewards = []
        action_lengths = []
        for episode in tqdm(range(episodes)):
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
        all_action_lengths.append(action_lengths)

    reward_histories = np.array(reward_histories)
    mean_reward_history = reward_histories.mean(axis=0)
    plt.plot(mean_reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(dir_name + "/easy_test_env.png")
    plt.close()

    all_action_lengths = np.array(all_action_lengths)
    mean_action_length = all_action_lengths.mean(axis=0)
    plt.plot(mean_action_length)
    plt.xlabel("Episode")
    plt.ylabel("Action Length")
    plt.savefig(dir_name + "/easy_test_env_action_length.png")
    plt.close()
