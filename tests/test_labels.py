import numpy as np
import pytest

from merge.data.labels import assign_direction_labels, compute_class_weights, compute_log2fc


def test_log2fc_zero_when_equal():
    values = np.ones((2, 3), dtype=np.float32)
    assert np.allclose(compute_log2fc(values, values), 0)


def test_label_threshold_boundaries():
    response = np.array([[-0.26, -0.25, 0.0, 0.25, 0.26]])
    labels = assign_direction_labels(response, threshold=0.25)
    assert labels.tolist() == [[0, 1, 1, 1, 2]]


def test_class_weights_balanced():
    labels = np.array([0, 1, 1, 2, 2, 2, -1])
    weights = compute_class_weights(labels)
    raw = np.array([6 / 3, 6 / 6, 6 / 9], dtype=np.float32)
    raw = raw / raw.mean()
    assert np.allclose(weights, raw)


def test_class_weights_require_all_classes():
    with pytest.raises(ValueError):
        compute_class_weights(np.array([0, 1, 1]))
