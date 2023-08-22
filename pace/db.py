from typing import Tuple
import numpy as np
import torch as th
from torch.utils.data import Dataset

"""
Author: Alice, Josue N Rivera
Date: 8/22/2023
"""

class ArrhythmiaDatabase(Dataset):

    def __init__(self, 
                 path:str = "../data/db.npy") -> None:
        super().__init__()
        
        imgs = []
        ncount = []

        ## TODO: load data from numpy file
        
        self.nclass = len(ncount)
        self.n = sum(ncount)

        self.imgs = th.tensor(np.array(imgs, dtype = np.float32))
        self.labels = th.concat([i*th.ones(ncount[i], dtype=th.long) for i in range(len(ncount))], dim=0)

    def get_nclass(self) -> int:
        return self.nclass

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, 
                    idx: int) -> Tuple[th.Tensor, th.Tensor]:
        return self.imgs[idx], self.labels[idx]

if __name__ == "__main__":
    dt = ArrhythmiaDatabase()
    temp = dt[1]