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
from utils.dataframe_utils import prepFrame, getMatches
from utils.sklearn_utils import SeasonalSplit, get_legendre_pipeline


class GameDataset(Dataset):
    def __init__(self, datapath: str = './data', is_val: bool = False, is_tourney: bool = False, season: int = 2023, seed: int = 7):
        # Load in data
        self.datapath = datapath
        self.data = []
        raw_features = pd.read_csv(Path(f'{datapath}/NormalizedEloAverages.csv')).set_index(['season', 'tid'])
        gids = prepFrame(pd.read_csv(Path(f'{datapath}/MNCAATourneyCompactResults.csv'))) if is_tourney else (
            prepFrame(pd.read_csv(Path(f'{datapath}/MRegularSeasonCompactResults.csv'))))
        data = gids.loc[gids.index.get_level_values(1) == season] if is_val else gids.loc[gids.index.get_level_values(1) != season]
        pipe = get_legendre_pipeline(degree=3)
        data = data.loc[:, 2004:, :, :]
        self.labels = torch.tensor(((data['t_score'] - data['o_score']) > 0).values).reshape(-1, 1).float()
        self.data = getMatches(data, raw_features, diff=True)
        self.gids = data
        self.data = torch.tensor(pipe.fit_transform(self.data)).float()
        self.data_len = self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.data.shape[0]

    def full_data(self):
        return self.data, self.labels


class KalmanDataset(Dataset):
    def __init__(self, datapath: str = './data', is_val: bool = False, season: int = 2023, seed: int = 7):
        # Load in data
        self.datapath = datapath
        self.data = []
        raw_features = pd.read_csv(Path(f'{datapath}/Averages.csv')).set_index(['season', 'tid'])
        raw_features = (raw_features - raw_features.mean()) / raw_features.std()
        y_gids = prepFrame(pd.read_csv(Path(f'{datapath}/MNCAATourneyCompactResults.csv')))
        opp_gids = prepFrame(pd.read_csv(Path(f'{datapath}/MRegularSeasonDetailedResults.csv')))
        y_data = y_gids.loc[y_gids.index.get_level_values(1) == season] if is_val else y_gids.loc[y_gids.index.get_level_values(1) != season]
        y_data = y_data.loc[:, 2004:, :, :]
        opp_data = opp_gids.loc[opp_gids.index.get_level_values(1) == season] if is_val else opp_gids.loc[
            opp_gids.index.get_level_values(1) != season]
        opp_data = opp_data.loc[:, 2004:, :, :]
        self.labels = torch.tensor(((y_data['t_score'] - y_data['o_score']) > 0).values).reshape(-1, 1).float()
        self.y_data = [torch.tensor(y.values, dtype=torch.float32) for y in getMatches(y_data, raw_features)]
        self.opp_data = getMatches(opp_data, raw_features)[1]
        # self.opp_data = torch.tensor(np.stack([self.opp_data.loc[:, :, idx[2], :].values for idx, row in self.opp_data.iterrows()]))

        # self.opp_data = torch.tensor(np.stack([grp.values[-4:] for idx, grp in self.opp_data]))
        kmd = opp_data[['t_ast', 't_score', 't_stl', 't_blk']]
        self.kalman_measure_data = torch.tensor(kmd.values / np.array([50, 130, 40, 40]), dtype=torch.float32)
        # self.kalman_measure_data = torch.tensor(np.stack([grp.values[-4:] for idx, grp in kmd]))
        self.gids = (y_gids, opp_gids)
        self.data_len = self.y_data[0].shape[1]

    def __getitem__(self, idx):
        opp = torch.tensor(self.opp_data.loc[:, self.opp_data.iloc[idx].name[1], self.opp_data.iloc[idx].name[2], :].values, dtype=torch.float32)
        return self.y_data[0][idx], opp, self.kalman_measure_data[idx], self.y_data[1][idx], self.labels[idx]

    def __len__(self):
        return self.y_data[0].shape[0]

    def full_data(self):
        return self.data, self.labels


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

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = GameDataset(self.datapath, season=self.season, is_tourney=self.is_tourney)
        self.val_dataset = GameDataset(self.datapath, season=self.season, is_val=True, is_tourney=self.is_tourney)

    def changeSeason(self, season: int, is_tourney: bool = False) -> None:
        self.season = season
        self.is_tourney = is_tourney
        self.train_dataset = GameDataset(self.datapath, season=self.season, is_tourney=self.is_tourney)
        self.val_dataset = GameDataset(self.datapath, season=self.season, is_val=True, is_tourney=self.is_tourney)

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
