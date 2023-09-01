from typing import Tuple
import numpy as np
import torch as th
from torch.utils.data import Dataset

"""
Author: Alice Cheng, Josue N Rivera
Date: 8/22/2023
"""

class ArrhythmiaDatabase(Dataset):

    def __init__(self, 
                 path:str = "../data/db.npz") -> None:
        super().__init__()

        npzfile = np.load(path)
        segments = npzfile.segments
        labels = npzfile.labels
        self.n = len(segments)

        self.segments = th.tensor(segments)
        self.labels = th.tensor(labels)

    def __len__(self) -> int:
        return self.n 

    def __getitem__(self, 
                    idx: int) -> Tuple[th.Tensor, th.Tensor]:
        return self.segments[idx], self.labels[idx]

if __name__ == "__main__":
    dt = ArrhythmiaDatabase()
    temp = dt[1]