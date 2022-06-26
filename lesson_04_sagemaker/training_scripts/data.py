import os
import multiprocessing as mp
import pickle
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self, x: torch.FloatTensor, y: torch.LongTensor) -> None:
        super().__init__()
        self.x = torch.FloatTensor(x) / 255.0
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        return self.x[index], self.y[index]


def get_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    with open(os.environ["SM_CHANNEL_TRAIN"] + "/train.npy", "rb") as f:
        train_x, train_y = pickle.load(f)

    with open(os.environ["SM_CHANNEL_VALID"] + "/valid.npy", "rb") as f:
        valid_x, valid_y = pickle.load(f)

    return (
        DataLoader(
            Data(train_x.reshape(len(train_x), -1), train_y),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=mp.cpu_count(),
        ),
        DataLoader(
            Data(valid_x.reshape(len(valid_x), -1), valid_y),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=mp.cpu_count(),
        ),
    )
