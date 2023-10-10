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
from fastai.vision.all import *

from src.data.timeseries_dataset_crops import TimeseriesDatasetCrops

from src.models.resnet1d import resnet1d_wang


# Load initial model and datasets from disk
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {dev}")
# model = torch.load('./initial_model.pt').to(dev)
# ds_train, ds_valid = torch.load('./ds_train.pt'), torch.load('./ds_valid.pt')
model = resnet1d_wang(
                num_classes=5,
                input_channels=12,
                kernel_size=5,
                ps_head=0.5,
                lin_ftrs_head=[128],
            )

base_path = "/home/arek/Desktop/ecg_benchmarking_lit/artifacts/ptbxl_split:v0/"
X_train, X_val, y_train, y_val = np.load(base_path + "train_data.npy"), np.load(base_path + "val_data.npy"), pd.read_csv(base_path + "train_labels.csv"), pd.read_csv(base_path + "val_labels.csv")

print(y_train)

# exit()
X_train, X_val = np.transpose(X_train, (0,2,1)), np.transpose(X_val, (0,2,1))
X_train, X_val, y_train, y_val = torch.tensor(X_train).float(), torch.tensor(X_val).float(), torch.tensor(y_train.iloc[:, 2:].values).float(), torch.tensor(y_val.iloc[:, 2:].values).float()

y_train_mask = y_train.sum(axis=1) > 0
y_val_mask = y_val.sum(axis=1) > 0

X_train, y_train = X_train[y_train_mask], y_train[y_train_mask]
X_val, y_val = X_val[y_val_mask], y_val[y_val_mask]

print(X_train.shape)
from pathlib import Path

import torch
import pytorch_lightning as pl
import wandb

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from src.lit_models.ptbxl_model import ECGClassifier
from src.models.resnet1d import resnet1d_wang
from pytorch_lightning.loggers import WandbLogger

from src.data.ptb_xl_multiclass_datamodule import PTB_XL_Datamodule

BATCH_SIZE = 128
EPOCHS = 50
ACCUMULATE_GRADIENT_STEPS = 1
FILTER_FOR_SINGLELABEL = False

# Initialize W&B
wandb.init(project='ecg_benchmarking_lit', name='fastai_run', entity="phd-dk")

run = wandb.init(project="ecg_benchmarking_lit", name="test_run", entity="phd-dk")
artifact = run.use_artifact(f"{'ptbxl_split'}:latest")

datadir = artifact.download()

data_module = PTB_XL_Datamodule(Path(datadir), filter_for_singlelabel=FILTER_FOR_SINGLELABEL, batch_size=BATCH_SIZE)
data_module.prepare_data()
data_module.setup()

#ds_train = TimeseriesDatasetCrops(X_train, y_train, output_size=250, chunk_length=0, min_chunk_length=250, random_crop=True, stride=0, transforms=[], time_dim=-1, batch_dim=0)
#ds_val = TimeseriesDatasetCrops(X_val, y_val, output_size=250, chunk_length=250, min_chunk_length=250, random_crop=True, stride=125, transforms=[], time_dim=-1, batch_dim=0)

# Create DataLoaders

print(len(data_module.val_dataset))

dls = DataLoaders.from_dsets(data_module.train_dataset,data_module.val_dataset, bs=128, device=dev)



print(len(dls.valid) * 128)
from fastai.imports import *

# Create the Learner
learn = Learner(
    dls,
    model,
    loss_func=F.binary_cross_entropy_with_logits,
    metrics=None,  # Replace with your metrics
    cbs=[WandbCallback(log='all')],  # WandB logging
    wd=0.01,
    path="./lolmodels/",
    
)

print(learn.opt_func)


#exit(0)


# Train the model
learn.fit_one_cycle(50, 0.01)
learn.save('trained_model.pt')

dls2 = DataLoaders.from_dsets(data_module.train_dataset,data_module.test_dataset, bs=128, device=dev)

y_pred,y = learn.get_preds(dl=dls2.valid)

print(y_pred.shape, y.shape)

torch.save({
    "y_pred": y_pred,
    "y": y
},"prediction_data_fastai_50.pt")