import torch


class QNet(torch.nn.Module):
    def __init__(self, observation_size, action_size, middle_size=18):
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.fc1 = torch.nn.Linear(observation_size, middle_size)
        self.fc2 = torch.nn.Linear(middle_size, middle_size)
        self.fc3 = torch.nn.Linear(middle_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
