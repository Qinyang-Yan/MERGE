"""Top-K three-class accuracy for the strongest measured responses."""

from typing import Dict, Sequence

import numpy as np


def compute_topk_direction_accuracy(
    log2fc_true: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_list: Sequence[int] = (20, 50, 100, 200),
) -> Dict[int, dict]:
    """Rank valid genes by absolute measured response and compare three-class labels.

    Unlike sign-only scoring, this function uses the original Down/Unchanged/Up
    targets. Missing groups and ignored genes never enter the ranking.
    """
    response = np.asarray(log2fc_true)
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if response.ndim != 2 or response.shape != true.shape or true.shape != pred.shape:
        raise ValueError("Response, true-label, and prediction matrices must share shape (groups, genes)")
    if any(int(k) <= 0 for k in k_list):
        raise ValueError("Every K must be positive")
    output: Dict[int, dict] = {}
    for k_value in k_list:
        k = int(k_value)
        per_group = {}
        all_hits = []
        for index in range(response.shape[0]):
            valid = (true[index] >= 0) & np.isfinite(response[index])
            candidates = np.flatnonzero(valid)
            if candidates.size and ((true[index, candidates] > 2) | (pred[index, candidates] < 0) | (pred[index, candidates] > 2)).any():
                raise ValueError("Top-K inputs contain invalid direction classes")
            order = np.argsort(-np.abs(response[index, candidates]), kind="stable")
            selected = candidates[order[:k]]
            if selected.size:
                hits = true[index, selected] == pred[index, selected]
                accuracy = float(hits.mean())
                all_hits.extend(hits.tolist())
            else:
                accuracy = None
            per_group[str(index)] = accuracy
        output[k] = {
            "overall": float(np.mean(all_hits)) if all_hits else None,
            "per_celltype": per_group,
        }
    return output
