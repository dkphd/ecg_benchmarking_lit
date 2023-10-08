import pytorch_lightning as pl
import torch


from src.utils.metrics import insert_metrics
from torch.optim.lr_scheduler import OneCycleLR


class ECGClassifier(pl.LightningModule):
    def __init__(self, model, num_classes, loss_fn, learning_rate, wd, total_optimizer_steps, **kwargs):
        super(ECGClassifier, self).__init__()

        self.kwargs = kwargs
        self.model = model

        # Loss function
        self.loss_fn = loss_fn

        # Learning rate
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.total_optimizer_steps = total_optimizer_steps

        self.wd = wd
        self.train_metrics, self.val_metrics, self.test_metrics = self._configure_metrics()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=True
        )  # Log to progress bar and logger
        self._calculate_metrics(self.train_metrics, y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=True
        )  # Log to progress bar and logger
        self._calculate_metrics(self.val_metrics, y_pred, y)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        return y_pred

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.wd)
        scheduler = OneCycleLR(opt, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches)
        lr_scheduler = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    def _configure_metrics(self):
        metrics_dict_train = insert_metrics(self, self.num_classes, prefix="train", task="multiclass")
        metrics_dict_val = insert_metrics(self, self.num_classes, prefix="val", task="multiclass")
        metrics_dict_test = insert_metrics(self, self.num_classes, prefix="test", task="multiclass")

        return metrics_dict_train, metrics_dict_val, metrics_dict_test

    def _calculate_metrics(self, metrics_dict, y_pred, y_true):
        for metric_name, metric_object in metrics_dict.items():
            metric_object(y_pred, y_true.argmax(dim=1))
            self.log(metric_name, metric_object, on_step=True, on_epoch=True)
