import torch as th
import torch.nn as nn
from math import floor

"""
Author: Alice Cheng, Josue N Rivera
Date: 9/1/2023
"""

class Pace(nn.Module):
    def __init__(self, 
                 beat_length:int = 966,
                 widths:int = 31) -> None:
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 10)

        height = ((widths-4)//2-9)//2
        width = ((beat_length-4)//2-9)//2

        self.fc1 = nn.Linear(16 * height * width, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 25)
        self.fc4 = nn.Linear(25, 5)

    def forward(self, x):
        x = self.pool(th.relu(self.conv1(x)))
        x = self.pool(th.relu(self.conv2(x)))
        x = th.flatten(x, 1) # flatten all dimensions except batch
        x = th.relu(self.fc1(x))
        x = th.relu(self.fc2(x))
        x = th.relu(self.fc3(x))
        return th.softmax(self.fc4(x))
    