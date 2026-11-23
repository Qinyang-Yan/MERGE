import numpy as np

from merge.evaluation.topk_direction import compute_topk_direction_accuracy


def test_topk_uses_three_class_labels():
    response = np.array([[1.0, -2.0, 0.1]])
    true = np.array([[2, 0, 1]])
    pred = np.array([[2, 0, 1]])
    output = compute_topk_direction_accuracy(response, true, pred, k_list=[1, 3])
    assert output[1]["overall"] == 1.0
    assert output[3]["overall"] == 1.0


def test_topk_masks_invalid_genes():
    response = np.array([[10.0, np.nan, 1.0]])
    true = np.array([[2, -1, 0]])
    pred = np.array([[2, 2, 0]])
    output = compute_topk_direction_accuracy(response, true, pred, k_list=[3])
    assert output[3]["overall"] == 1.0


def test_topk_handles_fully_missing_group():
    output = compute_topk_direction_accuracy(
        np.array([[np.nan, np.nan]]), np.array([[-1, -1]]), np.array([[0, 0]]), k_list=[2]
    )
    assert output[2]["overall"] is None
