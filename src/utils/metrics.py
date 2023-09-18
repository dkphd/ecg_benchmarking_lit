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
        setattr(ecg_classifier, metric_name, metric(num_classes=num_classes, task=task))
        metrics_dict[metric_name] = getattr(ecg_classifier, metric_name)

    # accuracy_no_reduction = Accuracy(num_classes=num_classes, task=task, average=None)
    # setattr(ecg_classifier, f"{prefix}_accuracy_no_reduction", accuracy_no_reduction)
    # metrics_dict[f"{prefix}_accuracy_no_reduction"] = accuracy_no_reduction

    return metrics_dict
