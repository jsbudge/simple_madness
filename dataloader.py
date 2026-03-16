from pandas import DataFrame
from typing import List, Optional, Union, Iterator
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
from multiprocessing import cpu_count
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
from utils.dataframe_utils import prepFrame, getMatches, date_weight, normalize
from utils.sklearn_utils import SeasonalSplit, get_legendre_pipeline


class GameDataset(Dataset):
    def __init__(self, datapath: str = './data', is_val: bool = False, season: int = 2023, game_num: int = 0, seed: int = 7):
        # Load in data
        self.datapath = datapath
        raw_data = pd.read_csv(Path(f'{datapath}/kalman_data.csv'))
        raw_data = raw_data.loc[raw_data['season'] == season] if is_val else raw_data.loc[raw_data['season'] != season]
        raw_data = raw_data.loc[raw_data['season'] != 2021]
        self.min_size = raw_data.groupby(['season', 'tid']).count().min().values[0]
        x_cols = [col for col in raw_data.columns if 't_' in col]
        u_cols = [col for col in raw_data.columns if 'o_' in col]
        x = []
        u = []
        y = []
        ids = []
        for idx, row in raw_data.groupby(['season', 'tid']):
            x.append(row[x_cols].values)
            u.append(row[u_cols].values)
            y.append(row[x_cols].values)
            ids.append(idx)
        self.x = x
        self.u = u
        self.y = y
        self.x_cols = x_cols
        self.u_cols = u_cols
        self.ids = ids
        self.data_len = len(x_cols)

    def __getitem__(self, idx):
        return (torch.tensor(self.x[idx]).float()[:self.min_size], torch.tensor(self.u[idx]).float()[:self.min_size],
                torch.tensor(self.y[idx]).float()[:self.min_size], self.ids[idx])

    def __len__(self):
        return len(self.x)

    def full_data(self):
        return self.data, self.labels

    def get_team(self, d):
        loc = np.where([i == d for i in self.ids])[0].item()
        return torch.tensor(self.x[loc]).float(), torch.tensor(self.u[loc]).float(), torch.tensor(self.y[loc]).float()

    def get_state(self, d):
        torch.load(f'{self.datapath}/state_vectors/{d[0]}_{d[1]}.pt', weights_only=True)


class KalmanDataset(Dataset):
    def __init__(self, datapath: str = './data', is_val: bool = False, season: int = 2023, seed: int = 7):
        # Load in data
        self.datapath = datapath
        gids = pd.read_csv(Path(f'{datapath}/GameDataAdv.csv')).set_index(['gid', 'season', 'tid', 'oid'])
        bids = pd.read_csv(Path(f'{datapath}/GameDataBasic.csv')).set_index(['gid', 'season', 'tid', 'oid'])
        avs = normalize(date_weight(gids, bids))
        data = gids.loc[gids.index.get_level_values(1) == season] if is_val else gids.loc[
            gids.index.get_level_values(1) != season]
        data = data.loc[:, 2011:, :, :]
        d0, d1 = getMatches(data, avs)
        home = bids.loc[data.index, ['gloc']].values
        self.labels = torch.tensor((data[['t_mov']].values > 0).astype(float)).float()
        self.d0 = torch.tensor(d0.values).float()
        self.d1 = torch.tensor(d1.values).float()
        self.home = torch.tensor(home).float()

    def __getitem__(self, idx):
        return self.d0[idx], self.d1[idx], self.home[idx], self.labels[idx]

    def __len__(self):
        return self.d0.shape[0]


class GameDataModuleCV(LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            single_example: bool = False,
            device: str = 'cpu',
            datapath: str = './data',
            is_tourney: bool = False,
            season: int = 2023,
            game_num: int = 0,
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
        self.is_tourney = is_tourney
        self.season = season
        self.game_num = game_num

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = GameDataset(self.datapath, season=self.season, game_num=self.game_num)
        self.val_dataset = GameDataset(self.datapath, season=self.season, is_val=True)

    def changeSeason(self, season: int, is_tourney: bool = False) -> None:
        self.season = season
        self.is_tourney = is_tourney
        self.train_dataset = GameDataset(self.datapath, season=self.season, game_num=self.game_num)
        self.val_dataset = GameDataset(self.datapath, season=self.season, is_val=True)

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


class KalmanDataModuleCV(LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            single_example: bool = False,
            device: str = 'cpu',
            datapath: str = './data',
            season: int = 2023,
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
        self.season = season

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = KalmanDataset(self.datapath, season=self.season)
        self.val_dataset = KalmanDataset(self.datapath, season=self.season, is_val=True)

    def changeSeason(self, season: int) -> None:
        self.season = season
        self.setup()

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
