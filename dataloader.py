from pandas import DataFrame
from typing import List, Optional, Union, Iterator
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
from multiprocessing import cpu_count
from pathlib import Path
from glob import glob
import numpy as np


class GameDataset(Dataset):
    def __init__(self, datapath: str = './data', is_val: bool = False, seed: int = 7):
        # Load in data
        self.datapath = datapath
        self.data = []
        start = 2010 if not is_val else 2021
        end = 2025 if is_val else 2021
        for season in range(start, end):
            dp = f'{datapath}/t{season}'
            if Path(f'{datapath}/{season}').exists():
                self.data.append(glob(f'{dp}/*.pt'))
        self.data = np.concatenate(self.data)
        np.random.shuffle(self.data)
        # Xt, Xs = train_test_split(self.data, random_state=seed)
        # self.data = Xs if is_val else Xt
        check = torch.load(self.data[0])
        self.data_len = check[0].shape[-1]

    def __getitem__(self, idx):
        return torch.load(self.data[idx])

    def __len__(self):
        return self.data.shape[0]


class GameDataModule(LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            single_example: bool = False,
            device: str = 'cpu',
            datapath: str = './data',
            **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = cpu_count() // 2
        self.pin_memory = pin_memory
        self.single_example = single_example
        self.device = device
        self.datapath = datapath

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = GameDataset(self.datapath)
        self.val_dataset = GameDataset(self.datapath, is_val=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )