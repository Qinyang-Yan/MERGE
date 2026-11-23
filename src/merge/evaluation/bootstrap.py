"""Group-level bootstrap uncertainty for external evaluation."""

import numpy as np

from .metrics import compute_metrics_from_confusion


def bootstrap_group_confidence_intervals(
    group_confusions: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Resample material/cell-type groups, not individual gene labels."""
    confusions = np.asarray(group_confusions, dtype=np.int64)
    if confusions.ndim != 3 or confusions.shape[1:] != (3, 3) or len(confusions) == 0:
        raise ValueError("Expected group confusion matrices with shape (groups, 3, 3)")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    names = ("accuracy", "macro_f1", "balanced_accuracy", "mcc")
    draws = {name: np.empty(n_bootstrap, dtype=np.float64) for name in names}
    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(confusions), size=len(confusions))
        metrics = compute_metrics_from_confusion(confusions[sampled].sum(axis=0))
        for name in names:
            draws[name][index] = metrics[name]
    point = compute_metrics_from_confusion(confusions.sum(axis=0))
    return {
        name: {
            "estimate": float(point[name]),
            "ci95_low": float(np.quantile(draws[name], 0.025)),
            "ci95_high": float(np.quantile(draws[name], 0.975)),
            "bootstrap_unit": "material-cell_type pseudo-bulk group",
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
        }
        for name in names
    }
