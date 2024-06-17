import torch
import torch.nn as nn

class PosePanningModel(nn.Module):
    def __init__(self):
        super(PosePanningModel, self).__init__()
        self.fc1 = nn.Linear(99, 128)  # 33 landmarks * 3 coordinates (x, y, z) = 99
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

# Example usage
if __name__ == "__main__":
    model = PosePanningModel()
    print(model)
