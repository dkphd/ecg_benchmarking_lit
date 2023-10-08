import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from fastai.data.core import DataLoaders
from fastai import *

from fastai.metrics import *
from fastai.torch_core import *
from fastai.learner import Learner

import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib


from src.models.timeseries_utils import *

from fastai import *

from fastai.metrics import *
from fastai.torch_core import *
from fastai.callback.tracker import SaveModelCallback
from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.callback.training import GradientClip, Recorder

from fastai.callback.schedule import lr_find
from fastai.callback import *

from pathlib import Path
from functools import partial
from fastai.callback.wandb import WandbCallback

from src.data.timeseries_dataset_crops import TimeseriesDatasetCrops

from src.models.resnet1d import resnet1d_wang


# Load initial model and datasets from disk
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.load('./initial_model.pt').to(dev)
# ds_train, ds_valid = torch.load('./ds_train.pt'), torch.load('./ds_valid.pt')
model = resnet1d_wang(
                num_classes=5,
                input_channels=12,
                kernel_size=5,
                ps_head=0.5,
                lin_ftrs_head=[128],
            )

X_train, X_val, y_train, y_val = np.load("./data/ptbxl_split/train_data.npy"), np.load("./data/ptbxl_split/val_data.npy"), pd.read_csv("./data/ptbxl_split/train_labels.csv"), pd.read_csv("./data/ptbxl_split/val_labels.csv")

print(y_train)

# exit()
X_train, X_val = np.transpose(X_train, (0,2,1)), np.transpose(X_val, (0,2,1))
X_train, X_val, y_train, y_val = torch.tensor(X_train).float(), torch.tensor(X_val).float(), torch.tensor(y_train.iloc[:, 2:].values).float(), torch.tensor(y_val.iloc[:, 2:].values).float()

y_train_mask = y_train.sum(axis=1) > 0
y_val_mask = y_val.sum(axis=1) > 0

X_train, y_train = X_train[y_train_mask], y_train[y_train_mask]
X_val, y_val = X_val[y_val_mask], y_val[y_val_mask]

print(X_train.shape)

# Initialize W&B
wandb.init(project='ecg_benchmarking_lit', name='fastai_run', entity="phd-dk")

ds_train = TimeseriesDatasetCrops(X_train, y_train, output_size=250, chunk_length=0, min_chunk_length=250, random_crop=True, stride=0, transforms=[], time_dim=-1, batch_dim=0)
ds_val = TimeseriesDatasetCrops(X_val, y_val, output_size=250, chunk_length=250, min_chunk_length=250, random_crop=True, stride=125, transforms=[], time_dim=-1, batch_dim=0)

# Create DataLoaders
dls = DataLoaders.from_dsets(ds_train, ds_val, bs=128)

# Create the Learner
learn = Learner(
    dls,
    model,
    loss_func=F.binary_cross_entropy_with_logits,
    metrics=None,  # Replace with your metrics
    cbs=[WandbCallback(log='all')],  # WandB logging
    wd=0.01,
    path="./lolmodels/"
)


# Train the model
learn.fit_one_cycle(50, 0.01)
torch.save(learn.model, 'trained_model.pt')