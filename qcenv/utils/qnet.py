import torch


class QNet(torch.nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.fc1 = torch.nn.Linear(observation_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
