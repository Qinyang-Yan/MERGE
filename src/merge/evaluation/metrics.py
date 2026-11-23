"""Fixed three-class metrics computed from auditable confusion matrices."""

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

LABELS = (0, 1, 2)


def _filter_ignore(y_true, y_pred, y_proba=None, ignore_index: int = -1):
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred shapes differ: {true.shape} versus {pred.shape}")
    keep = true != ignore_index
    true = true[keep].astype(np.int64)
    pred = pred[keep].astype(np.int64)
    if ((true < 0) | (true > 2)).any() or ((pred < 0) | (pred > 2)).any():
        raise ValueError("Direction labels and predictions must be in {0, 1, 2}")
    probabilities = None
    if y_proba is not None:
        probabilities = np.asarray(y_proba, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape != (len(keep), 3):
            raise ValueError(f"Expected probabilities with shape ({len(keep)}, 3), got {probabilities.shape}")
        probabilities = probabilities[keep]
        if not np.isfinite(probabilities).all():
            raise ValueError("Probabilities contain non-finite values")
    return true, pred, probabilities


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    ignore_index: int = -1,
) -> dict:
    """Compute aggregate, class-resolved, and optional ranking metrics."""
    true, pred, probabilities = _filter_ignore(y_true, y_pred, y_proba, ignore_index)
    if true.size == 0:
        raise ValueError("No valid labels remain after ignore_index filtering")
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (true, pred), 1)
    output = compute_metrics_from_confusion(confusion)
    output["auroc_macro_ovr"] = None
    output["auprc_macro_ovr"] = None
    if probabilities is not None and np.unique(true).size == 3:
        binary = label_binarize(true, classes=LABELS)
        output["auroc_macro_ovr"] = float(
            roc_auc_score(binary, probabilities, average="macro", multi_class="ovr")
        )
        output["auprc_macro_ovr"] = float(
            average_precision_score(binary, probabilities, average="macro")
        )
    return output


def compute_metrics_from_confusion(conf: np.ndarray) -> dict:
    """Compute metrics over Down/Unchanged/Up, including absent-class zeros."""
    confusion = np.asarray(conf, dtype=np.int64)
    if confusion.shape != (3, 3) or (confusion < 0).any():
        raise ValueError("Expected a non-negative 3x3 confusion matrix")
    total = int(confusion.sum())
    if total == 0:
        raise ValueError("The confusion matrix contains no observations")
    true_positive = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    observed = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(true_positive, observed, out=np.zeros(3), where=observed != 0)
    f1_denominator = precision + recall
    f1 = np.divide(2 * precision * recall, f1_denominator, out=np.zeros(3), where=f1_denominator != 0)
    correct = float(true_positive.sum())
    covariance = correct * total - float(np.dot(observed, predicted))
    predicted_variance = total * total - float(np.dot(predicted, predicted))
    observed_variance = total * total - float(np.dot(observed, observed))
    mcc_denominator = np.sqrt(observed_variance * predicted_variance)
    return {
        "macro_f1": float(np.mean(f1)),
        "micro_f1": float(correct / total),
        "balanced_accuracy": float(np.mean(recall)),
        "accuracy": float(correct / total),
        "mcc": float(covariance / mcc_denominator) if mcc_denominator else 0.0,
        "f1_per_class": f1.tolist(),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "confusion_matrix": confusion.tolist(),
        "n_samples": total,
        "class_distribution": {
            "down": int(observed[0]),
            "unchanged": int(observed[1]),
            "up": int(observed[2]),
        },
        "auroc_macro_ovr": None,
        "auprc_macro_ovr": None,
    }


def compute_per_celltype_metrics(y_true, y_pred, cell_types) -> Dict[str, dict]:
    """Compute metrics independently for each annotated cell type."""
    return _compute_group_metrics(y_true, y_pred, cell_types)


def compute_per_metal_metrics(y_true, y_pred, metal_classes) -> Dict[str, dict]:
    """Compute metrics independently for each material condition."""
    return _compute_group_metrics(y_true, y_pred, metal_classes)


def _compute_group_metrics(y_true, y_pred, groups) -> Dict[str, dict]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    groups = np.asarray(groups).astype(str)
    if len(groups) != len(true):
        raise ValueError("Group vector length does not match labels")
    return {
        group: compute_all_metrics(true[groups == group], pred[groups == group])
        for group in sorted(set(groups))
    }
