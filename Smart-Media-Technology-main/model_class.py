"""
This is the NN Model for Hand Spread Detection.
I Put this into its own file to avoid issues with multiple definitions of essentially the same thing.
Used by modelTraining.py and main.py
"""
import torch
import torch.nn as nn

class HandGestureModel(nn.Module):
    def __init__(self):
        super(HandGestureModel, self).__init__()
        self.fc1 = nn.Linear(63, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x