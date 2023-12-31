from torchmetrics import (
    Accuracy,
    AUROC,
    AveragePrecision,
    ConfusionMatrix,
    F1Score,
    MatthewsCorrCoef,
    Precision,
    Recall,
    Specificity,
)


def insert_metrics(ecg_classifier, num_classes, prefix="train", task="multiclass"):
    metrics_dict = {}

    for metric in (
        Accuracy,
        AUROC,
        AveragePrecision,
        # ConfusionMatrix,
        F1Score,
        MatthewsCorrCoef,
        Precision,
        Recall,
        Specificity,
    ):
        metric_name = f"{prefix}_{metric.__name__.lower()}"

        if task == "multiclass":
            setattr(ecg_classifier, metric_name, metric(num_classes=num_classes, task="multiclass"))
        elif task == "multilabel":
            setattr(ecg_classifier, metric_name, metric(num_labels=num_classes, task="multilabel"))
        metrics_dict[metric_name] = getattr(ecg_classifier, metric_name)

    return metrics_dict
