#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pathlib import Path

import torch
import pytorch_lightning as pl
import wandb

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from src.lit_models.ptbxl_model import ECGClassifier
from src.models.resnet1d import resnet1d_wang
from src.models.conv_transformer import conv_transformer
from src.models.inception1d import inception1d

from pytorch_lightning.loggers import WandbLogger

from src.data.ptb_xl_multiclass_datamodule import PTB_XL_Datamodule
from torchmetrics.classification import MulticlassAccuracy

import os
from datetime import datetime


# In[4]:


def get_model_registry():
    return {
        "resnet1d_wang": resnet1d_wang,
        "conv_transformer": conv_transformer,
        "inception1d": inception1d
    }


# In[5]:


def create_directory_with_timestamp(path, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}_{timestamp}"
    full_path = os.path.join(path, dir_name)
    os.makedirs(full_path, exist_ok=True)

    return full_path


# In[6]:


def get_datamodule(run, FILTER_FOR_SINGLELABEL, BATCH_SIZE):
    artifact = run.use_artifact(f"{'ptbxl_split'}:latest")

    datadir = artifact.download()

    data_module = PTB_XL_Datamodule(Path(datadir), filter_for_singlelabel=FILTER_FOR_SINGLELABEL, batch_size=BATCH_SIZE)

    data_module.prepare_data()
    data_module.setup()

    return data_module


# In[7]:


def get_model(total_optimizer_steps, model_config, model_name="resnet1d_wang", task='multilabel', loss=torch.nn.BCEWithLogitsLoss()):
    model = get_model_registry()[model_name](
    **model_config
)

    model_lit = ECGClassifier(
        model, 5, loss, 0.01, wd=0.01, total_optimizer_steps=total_optimizer_steps, task=task)
    
    return model_lit


# In[8]:


def train_model(model_lit, data_module, config):
    wandb_logger = WandbLogger(log_model="all")
    wandb_logger.watch(model_lit, log="all")

    dir_model = create_directory_with_timestamp("./models", "resnet1d_wang")

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=False, mode="min")
    learning_rate_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True)

    # Create the Learner
    trainer = pl.Trainer(
        accumulate_grad_batches=config.ACCUMULATE_GRADIENT_STEPS,
        log_every_n_steps=1,
        max_epochs=config.EPOCHS,
        logger=wandb_logger,
        callbacks=[early_stop_callback, learning_rate_monitor],
    )

    trainer.fit(model_lit, datamodule=data_module)

    return trainer


# In[9]:


def validate_model(trainer, data_module, metrics={}):
    res = trainer.predict(dataloaders=data_module.test_dataloader())

    y_hat, y = torch.concatenate([x[0] for x in res]), torch.concatenate([x[1] for x in res])

    y_hat = torch.nn.functional.sigmoid(y_hat)

    metrics  = {
        'multiclass_accuracy': MulticlassAccuracy(num_classes=y_hat.size(1), average='weighted')
    }

    target = torch.argmax(y, axis=-1)
    preds = torch.argmax(y_hat, axis=-1)



    return {
        k: v(preds, target) for k, v in metrics.items()
    }


# In[10]:


def train_model_with_validation(config, project="ecg_arek_test", name="test_run", entity="phd-dk"):

    run = wandb.init(project=project, name=name, entity=entity, config=config)

    BATCH_SIZE = run.config.BATCH_SIZE
    FILTER_FOR_SINGLELABEL = run.config.FILTER_FOR_SINGLELABEL

    loss = torch.nn.BCEWithLogitsLoss() if not FILTER_FOR_SINGLELABEL else torch.nn.CrossEntropyLoss()
    task = "multilabel" if not FILTER_FOR_SINGLELABEL else "multiclass"

    data_module = get_datamodule(run, FILTER_FOR_SINGLELABEL, BATCH_SIZE)
    print(len(data_module.val_dataset))

    total_optimizer_steps = int(len(data_module.train_dataset) * run.config.EPOCHS / run.config.ACCUMULATE_GRADIENT_STEPS)

    model_lit = get_model(total_optimizer_steps, run.config.model_config, run.config.model_name, task, loss)

    trainer = train_model(model_lit, data_module, run.config)


    trainer.test(model=trainer.model, dataloaders=data_module.test_dataloader())



    # results = validate_model(trainer=trainer, data_module=data_module)

    # wandb_code = run.log({
    #     f"test/{metric_name}": metric_value for metric_name, metric_value in results.items()
    # })

    # print(wandb_code, {
    #     f"test/{metric_name}": metric_value for metric_name, metric_value in results.items()
    # })

    run.finish()

    return trainer, data_module, model_lit


# In[9]:


# model_config = dict(
#     num_classes = 5,
#     k = 12,
#     headers = 5,
#     depth = 1,
#     seq_length=250
# )
# config = {
#     "BATCH_SIZE": 128,
#     "EPOCHS": 50,
#     "ACCUMULATE_GRADIENT_STEPS": 1,
#     "FILTER_FOR_SINGLELABEL" : False,
#     "model_config": model_config,
#     "model_name": "conv_transformer"
# }

model_config = dict(
    num_classes=5,
    input_channels=12,
    kernel_size=5 * 8,
    ps_head=0.5,
    lin_ftrs_head=[128],
)

config = {
    "BATCH_SIZE": 128,
    "EPOCHS": 50,
    "ACCUMULATE_GRADIENT_STEPS": 1,
    "FILTER_FOR_SINGLELABEL" : False,
    "model_config": model_config,
    "model_name": "inception1d"
}


# In[10]:


trainer, data_module, model = train_model_with_validation(config, name="conv transformer default")




