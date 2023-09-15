from typing import Tuple
import numpy as np
import torch as th
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

"""
Author: Alice Cheng, Josue N Rivera
Date: 8/22/2023
"""

class ArrhythmiaDatabase(Dataset):

    def __init__(self, 
                 path:str = "data/db.npz") -> None:
        super().__init__()

        npzfile = np.load(path)
        scalograms = npzfile["scalograms"]
        labels = npzfile["labels"]
        self.n = len(labels)

        self.scalograms = th.tensor(scalograms)
        self.labels = th.tensor(labels)

    def __len__(self) -> int:
        return self.n 

    def __getitem__(self, 
                    idx: int) -> Tuple[th.Tensor, th.Tensor]:
        return self.segments[idx], self.labels[idx]

if __name__ == "__main__":
    dt = ArrhythmiaDatabase("data/db_31_100.npz")

    nonzero_label_indices = (dt.labels >= 1).nonzero(as_tuple=True)[0]
    for i in range(10, 13):
        segment_1, label_1 = dt[nonzero_label_indices[i]]

        plt.figure()
        plt.title(f'{label_1}')
        plt.imshow(segment_1, cmap='viridis')

    for i in range(0, len(dt), int(len(dt)/3)):
        segment_1, label_1 = dt[i]

        plt.figure()
        plt.title(f'{label_1}')
        plt.imshow(segment_1, cmap='viridis')

    plt.show()    
