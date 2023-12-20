from dqn import DQNAgent
from qnet import QNet
from replay_buffer import ReplayBuffer

import gymnasium as gym
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    env = gym.make("CartPole-v1")
    episodes = 300
    sync_interval = 20
    reward_histories = []
    num_tests = 5

    for test in tqdm(range(num_tests)):
        q_function = QNet(env.observation_space.shape[0], env.action_space.n)
        optimizer = torch.optim.Adam(q_function.parameters(), lr=1e-2)
        replay_buffer = ReplayBuffer(10000, 32)
        agent = DQNAgent(q_function, optimizer, replay_buffer, 0.99, 0.1, device)
        rewards = []
        for episode in tqdm(range(episodes)):
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
        reward_histories.append(rewards)

    reward_histories = np.array(reward_histories)
    mean_reward_history = reward_histories.mean(axis=0)
    plt.plot(mean_reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("cartpole.png")
