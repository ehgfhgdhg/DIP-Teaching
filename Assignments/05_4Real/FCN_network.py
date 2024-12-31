import torch
import torch.nn as nn


class FullyConnectedNetwork(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(inplace=True),
        ).to(device)
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        ).to(device)
        self.fc3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        ).to(device)
        self.fc4 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        ).to(device)
        self.fc5 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        ).to(device)
        self.fc6 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        ).to(device)
        self.fc7 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        ).to(device)
        self.fc8 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        ).to(device)
        self.fc9 = nn.Linear(256, 3).to(device)
        self.fc10 = nn.Linear(256, 4).to(device)

    def forward(self, x, t):
        x = torch.column_stack((x, t * torch.ones(x.size()[0], 1, device=x.device)))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        delta_x = self.fc9(x)
        delta_q = self.fc10(x)
        return delta_x, delta_q
