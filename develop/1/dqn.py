import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy


class DQNAgent:
    def __init__(
        self,
        q_function: torch.nn.Module,
        optimizer,
        replay_buffer,
        gamma: float,
        epsilon: float,
        device: str,
    ) -> None:
        super().__init__()
        self.qnet = q_function.to(device)
        self.qnet_target = copy.deepcopy(self.qnet)
        self.observation_size = self.qnet.observation_size
        self.action_size = self.qnet.action_size
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device

    def sync_target(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state: np.ndarray):
        state = state.to(self.device)
        with torch.no_grad():
            if torch.rand(1) < self.epsilon:
                action = torch.randint(0, self.qnet.action_size, (1,))
            else:
                state = state[None, :]
                qs = self.qnet(state)
                action = torch.argmax(qs)
            return action.item()

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
    ):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return None
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state = state.to(self.device)
        qs = self.qnet(state)
        q = qs[torch.arange(self.replay_buffer.batch_size), action]
        next_state = next_state.to(self.device)
        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1).values.cpu()
        target = reward + self.gamma * next_q * (1 - done)
        loss = torch.nn.functional.mse_loss(q, target.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
