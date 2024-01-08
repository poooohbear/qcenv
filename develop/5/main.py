from qcenv.agents import DQNAgent

# from qcenv.utils import QNet
from qcenv.replay_buffers import ReplayBuffer

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
        seq_length = 3
        one_hot_category_size = 5
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


"""
[0,1,2,3,4,3,2,1,]が繰り返すデータセット
"""


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # データセットの作成
    data = [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2]
    sequences = []
    labels = []

    for i in range(len(data) - 3):
        sequence = torch.stack([to_one_hot(data[j]) for j in range(i, i + 3)])
        sequences.append(sequence)
        labels.append(data[i + 3])

    sequences = torch.stack(sequences)
    labels = torch.tensor(labels, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(sequences, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # train
    model = EasyTestEnvQNet()

    n_train = 1000
    one_hot_category_size = 5
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 500
    for epoch in range(epochs):
        for sequences, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(sequences.permute(0, 2, 1))
            loss = loss_fn(outputs, labels.long())
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch}, loss: {loss}")

    # test
    model.eval()
    with torch.no_grad():
        for sequences, labels in dataloader:
            outputs = model(sequences.permute(0, 2, 1))
            y_pred = torch.argmax(outputs, dim=1)
            y_true = labels
            print(f"sequences: {sequences}")
            print(f"y_pred: {y_pred}, y_true: {y_true}\n")
