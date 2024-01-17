import torch
import torch.nn.functional as F


class Conv1dQNet(torch.nn.Module):
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
