import torch

import pandas as pd
import numpy as np

import pytorch_lightning as pl

from ..utils.preprocessing import StandardScalerTorch
from .timeseries_dataset_crops import TimeseriesDatasetCrops
    

from torch.utils.data import DataLoader


class PTB_XL_Datamodule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, filter_for_singlelabel=False, filter_no_labels=True):
        super().__init__()

        self.data_path = data_path
        self.scaler = StandardScalerTorch()
        self.batch_size = batch_size
        self.filter_for_singlelabel = filter_for_singlelabel
        self.filter_no_labels = filter_no_labels

    def prepare_data(self):
        self._load_data()
        if self.filter_for_singlelabel:
            self._filter_for_singlelabel()
        elif self.filter_no_labels:
            self._filter_no_labels()
        self._scale()

    def setup(self, stage=None):
        self.train_dataset = TimeseriesDatasetCrops(self.X_train, self.y_train, output_size=250, chunk_length=0, min_chunk_length=250, random_crop=True, stride=0, transforms=[], time_dim=-1, batch_dim=0)
        self.val_dataset = TimeseriesDatasetCrops(self.X_val, self.y_val, output_size=250, chunk_length=250, min_chunk_length=250, random_crop=True, stride=125, transforms=[], time_dim=-1, batch_dim=0)
        self.test_dataset = TimeseriesDatasetCrops(self.X_test, self.y_test, output_size=250, chunk_length=250, min_chunk_length=250, random_crop=True, stride=125, transforms=[], time_dim=-1, batch_dim=0)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


    def _load_data(self):

        self.X_train = torch.Tensor(np.load(self.data_path / "X_train.npy")).transpose(1, 2)
        self.y_train = torch.Tensor(pd.read_csv(self.data_path / "y_train.csv").iloc[:, 1:].values)
        self.X_val = torch.Tensor(np.load(self.data_path / "X_val.npy")).transpose(1, 2)
        self.y_val = torch.Tensor(pd.read_csv(self.data_path / "y_val.csv").iloc[:, 1:].values)
        self.X_test = torch.Tensor(np.load(self.data_path / "X_test.npy")).transpose(1, 2)
        self.y_test = torch.Tensor(pd.read_csv(self.data_path / "y_test.csv").iloc[:, 1:].values)
        
        
    def _filter_no_labels(self):
        mask_y_val = self.y_val.sum(axis=1) > 0
        mask_y_test = self.y_test.sum(axis=1) > 0
        mask_y_train = self.y_train.sum(axis=1) > 0

        self.X_train = self.X_train[mask_y_train]
        self.y_train = self.y_train[mask_y_train]
        self.X_val = self.X_val[mask_y_val]
        self.y_val = self.y_val[mask_y_val]
        self.X_test = self.X_test[mask_y_test]
        self.y_test = self.y_test[mask_y_test]

    def _filter_for_singlelabel(self):
        mask_y_val = self.y_val.sum(axis=1)==1
        mask_y_test = self.y_test.sum(axis=1)==1
        mask_y_train = self.y_train.sum(axis=1)==1

        self.X_train = self.X_train[mask_y_train]
        self.y_train = self.y_train[mask_y_train]
        self.X_val = self.X_val[mask_y_val]
        self.y_val = self.y_val[mask_y_val]
        self.X_test = self.X_test[mask_y_test]
        self.y_test = self.y_test[mask_y_test]

    def _scale(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
    