from pathlib import Path

import torch
import pytorch_lightning as pl
import wandb

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from src.lit_models.ptbxl_model import ECGClassifier
from src.models.resnet1d import resnet1d_wang
from pytorch_lightning.loggers import WandbLogger

from .data.ptb_xl_multiclass_datamodule import PTB_XL_Datamodule

import os
from datetime import datetime

def create_directory_with_timestamp(path, prefix):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = f"{prefix}_{timestamp}"
    full_path = os.path.join(path, dir_name)
    os.makedirs(full_path, exist_ok=True)

    return full_path


BATCH_SIZE = 128
EPOCHS = 50
ACCUMULATE_GRADIENT_STEPS = 1

data_module = PTB_XL_Datamodule(Path("./data"), filter_for_singlelabel=False, batch_size=BATCH_SIZE)

data_module.prepare_data()
data_module.setup()


total_optimizer_steps = int(len(data_module.train_dataset) * EPOCHS / ACCUMULATE_GRADIENT_STEPS)
print(total_optimizer_steps)
print(len(data_module.train_dataset))

# exit()

# Initialize W&B
wandb.init(project='fastai_vs_lightning', name='lightning_run_ours')

model = resnet1d_wang(
                num_classes=5,
                input_channels=12,
                kernel_size=5,
                ps_head=0.5,
                lin_ftrs_head=[128],
            )

model_lit = ECGClassifier(model, 5, torch.nn.BCEWithLogitsLoss(), 0.01, wd=0.01, total_optimizer_steps=total_optimizer_steps)
wandb_logger = WandbLogger()
wandb_logger.watch(model_lit, log='all')

dir_model = create_directory_with_timestamp("./models", "resnet1d_wang")

checkpoint_callback = ModelCheckpoint(dirpath=dir_model, save_top_k=2, monitor="val_loss")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
learning_rate_monitor = LearningRateMonitor(logging_interval='step')

# Create the Learner
trainer = pl.Trainer(max_epochs=50, logger=wandb_logger, callbacks=[checkpoint_callback, early_stop_callback, learning_rate_monitor])

trainer.fit(model_lit, datamodule=data_module)