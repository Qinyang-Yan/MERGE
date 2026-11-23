"""Simple, fully implemented reference predictors."""

from typing import Optional

import numpy as np


class MajorityClassPredictor:
    """Predict the most frequent class observed in the training labels."""

    def __init__(self) -> None:
        self.majority_class: Optional[int] = None
        self.n_genes: Optional[int] = None

    def fit(self, labels: np.ndarray) -> "MajorityClassPredictor":
        labels = np.asarray(labels)
        valid = labels.reshape(-1)
        valid = valid[valid >= 0]
        if valid.size == 0:
            raise ValueError("Cannot fit a majority predictor without valid labels")
        values, counts = np.unique(valid, return_counts=True)
        self.majority_class = int(values[np.argmax(counts)])
        self.n_genes = int(labels.shape[-1])
        return self

    def predict(self, n_groups: int) -> np.ndarray:
        """Return one prediction matrix per evaluation group."""
        if self.majority_class is None or self.n_genes is None:
            raise RuntimeError("Fit the predictor before calling predict")
        return np.full((n_groups, self.n_genes), self.majority_class, dtype=np.int64)
