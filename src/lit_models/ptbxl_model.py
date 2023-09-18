import pytorch_lightning as pl
import torch


from src.utils.metrics import insert_metrics


class ECGClassifier(pl.LightningModule):
    def __init__(self, model, num_classes, loss_fn, learning_rate, **kwargs):
        super(ECGClassifier, self).__init__()

        self.kwargs = kwargs

        self.model = model

        # Loss function
        self.loss_fn = loss_fn

        # Learning rate
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.train_metrics, self.val_metrics, self.test_metrics = self._configure_metrics()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)  # Log to progress bar and logger
        self._calculate_metrics(self.train_metrics, y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)  # Log to progress bar and logger
        self._calculate_metrics(self.val_metrics, y_pred, y)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)

    def configure_optimizers(self):
        from torch.optim import Adam
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True),
            "monitor": "val_loss",  # The metric to monitor for learning rate reduction
            "interval": "epoch",  # Reduce LR every epoch
            "frequency": 1,  # How frequent within one epoch
            "strict": True,
        }
        return {"optimizer": optimizer, "lr_scheduler": self.scheduler, "monitor": "val_loss"}

    def _configure_metrics(self):

        metrics_dict_train = insert_metrics(self, self.num_classes, prefix="train", task="multiclass")
        metrics_dict_val = insert_metrics(self, self.num_classes, prefix="val", task="multiclass")
        metrics_dict_test = insert_metrics(self, self.num_classes, prefix="test", task="multiclass")

        return metrics_dict_train, metrics_dict_val, metrics_dict_test

    def _calculate_metrics(self, metrics_dict, y_pred, y_true):

        for metric_name, metric_object in metrics_dict.items():
            metric_object(y_pred, y_true.argmax(dim=1))
            self.log(metric_name, metric_object, on_step=True, on_epoch=True)
