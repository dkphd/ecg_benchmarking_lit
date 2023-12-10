import wandb

from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from src.lit_models.ptbxl_model import ECGClassifier
from src.models.conv_transformer import conv_transformer


from pytorch_lightning.loggers import WandbLogger

from src.data.ptb_xl_multiclass_datamodule import PTB_XL_Datamodule
from torchmetrics.classification import MulticlassAccuracy

import os
from datetime import datetime


def create_directory_with_timestamp(path, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}_{timestamp}"
    full_path = os.path.join(path, dir_name)
    os.makedirs(full_path, exist_ok=True)

    return full_path


def get_datamodule(run, FILTER_FOR_SINGLELABEL, BATCH_SIZE):
    artifact = run.use_artifact(f"{'ptbxl_split'}:latest")

    datadir = artifact.download()

    data_module = PTB_XL_Datamodule(Path(datadir), filter_for_singlelabel=FILTER_FOR_SINGLELABEL, batch_size=BATCH_SIZE)

    data_module.prepare_data()
    data_module.setup()

    return data_module


def train_model():
    
    
    model_config = dict(
        num_classes = 5,
        k = 12,
        seq_length=250,
    )
    FILTER_FOR_SINGLELABEL = False
    BATCH_SIZE = 64
    loss = torch.nn.BCEWithLogitsLoss() if not FILTER_FOR_SINGLELABEL else torch.nn.CrossEntropyLoss()
    task = "multilabel" if not FILTER_FOR_SINGLELABEL else "multiclass"

    with wandb.init() as run:

        artifact = run.use_artifact(f"{'ptbxl_split'}:latest")
        datadir = artifact.download()

        data_module = PTB_XL_Datamodule(Path(datadir), filter_for_singlelabel=FILTER_FOR_SINGLELABEL, batch_size=BATCH_SIZE)

        data_module.prepare_data()
        data_module.setup()


        model_config.update(run.config)
    
        model = conv_transformer(
        **model_config
        )
    
        model_lit = ECGClassifier(
            model, 5, loss, 0.01, wd=0.01, task=task)
    

        wandb_logger = WandbLogger(log_model="all")
        wandb_logger.watch(model_lit, log="all") 
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=10, verbose=False, mode="min")
        learning_rate_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True)
        # Create the Learner
        trainer = pl.Trainer(
            accumulate_grad_batches=8,
            log_every_n_steps=1,
            max_epochs=50,
            logger=wandb_logger,
            callbacks=[early_stop_callback, learning_rate_monitor],
            enable_progress_bar=False
        )

        trainer.fit(model_lit, datamodule=data_module)

sweep_config = {
    'method': 'bayes',  # Can be grid, random, bayes
    'metric': {
        'name': 'val_loss_epoch',
        'goal': 'minimize'   
    },
    'parameters': {
        'headers': {
            'values': [2,3,4,5,6,7,8,9,10,11,12]
        },
        'dropout_proba': {
            'min': 0.1,
            'max': 0.5
        },
        'depth': {
            'values': [1,2,3,4,5]
        },
        'dd1': {
            'min': 0.1,
            'max': 0.5
        }
        # Add other hyperparameters here
    }
}

sweep_id = wandb.sweep(sweep_config, project="ecg_arek_tests")
print(f"Sweep id: {sweep_id}")
wandb.agent(sweep_id, train_model, count=40)
