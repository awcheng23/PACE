import torch as th
import torch.nn as nn

"""
Author: Alice Cheng, Josue N Rivera
Date: 9/1/2023
"""

class Pace(nn.Module):
    def __init__(self, 
                 beat_length:int = 966,
                 widths:int = 31,
                 n_categories:int = 5) -> None:
        super().__init__()
        size = th.tensor([widths, beat_length])
        size = ((size-4)//2-9)//2

        self.conv1 = nn.Conv2d(1, 6, 5, dtype=th.double)
        self.conv2 = nn.Conv2d(6, 16, 10, dtype=th.double)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * size.prod(), 1000, dtype=th.double)
        self.fc2 = nn.Linear(1000, 500, dtype=th.double)
        self.fc3 = nn.Linear(500, 25, dtype=th.double)
        self.fc4 = nn.Linear(25, n_categories, dtype=th.double)
    
    def predict(self, x:th.Tensor):
        x = x.unsqueeze(0) if len(x.shape) == 2 else x
        return th.softmax(x, dim=1)

    def forward(self, x:th.Tensor):
        x = self.pool(th.relu(self.conv1(x)))
        x = self.pool(th.relu(self.conv2(x)))
        x = th.flatten(x, 1) # flatten all dimensions except batch
        x = th.relu(self.fc1(x))
        x = th.relu(self.fc2(x))
        x = th.sigmoid(self.fc3(x))
        return self.fc4(x)
    