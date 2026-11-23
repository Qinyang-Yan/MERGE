import numpy as np

from merge.evaluation.metrics import compute_all_metrics, compute_metrics_from_confusion


def test_macro_f1_perfect():
    labels = np.array([0, 1, 2])
    metrics = compute_all_metrics(labels, labels)
    assert metrics["macro_f1"] == 1.0


def test_metrics_dict_keys():
    labels = np.array([0, 1, 2, 1])
    metrics = compute_all_metrics(labels, labels)
    assert {"macro_f1", "micro_f1", "balanced_accuracy", "accuracy", "mcc", "confusion_matrix"} <= set(metrics)


def test_auroc_single_class_handled():
    labels = np.array([1, 1, 1])
    probabilities = np.tile(np.array([[0.1, 0.8, 0.1]]), (3, 1))
    metrics = compute_all_metrics(labels, labels, probabilities)
    assert metrics["auroc_macro_ovr"] is None
    assert np.isclose(metrics["balanced_accuracy"], 1 / 3)


def test_confusion_metrics_match_full_metrics():
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 0, 2])
    confusion = np.zeros((3, 3), dtype=int)
    for true, pred in zip(y_true, y_pred):
        confusion[true, pred] += 1
    full = compute_all_metrics(y_true, y_pred)
    streamed = compute_metrics_from_confusion(confusion)
    for key in ["macro_f1", "micro_f1", "balanced_accuracy", "accuracy", "mcc"]:
        assert np.isclose(streamed[key], full[key])
    assert streamed["confusion_matrix"] == full["confusion_matrix"]
