"""Loss function for masked three-class direction prediction."""

import torch
from torch import nn
from torch.nn import functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Class-weighted cross entropy over ``(batch, genes, classes)`` logits."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.shape != (3,) or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("class_weights must contain three finite positive values")
        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing must lie in [0, 1)")
        self.register_buffer("class_weights", weights)
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3 or logits.shape[-1] != 3:
            raise ValueError(f"Expected logits with shape (batch, genes, 3), got {tuple(logits.shape)}")
        if labels.shape != logits.shape[:2]:
            raise ValueError(f"Label shape {tuple(labels.shape)} does not match logits")
        labels_flat = labels.reshape(-1).long()
        if (labels_flat != self.ignore_index).any():
            return F.cross_entropy(
                logits.reshape(-1, 3),
                labels_flat,
                weight=self.class_weights.to(device=logits.device, dtype=logits.dtype),
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
            )
        return logits.sum() * 0.0


def build_loss(loss_config: dict, class_weights: torch.Tensor) -> WeightedCrossEntropyLoss:
    """Build the configured MERGE loss."""
    if loss_config.get("type", "weighted_ce") != "weighted_ce":
        raise ValueError("MERGE supports weighted_ce loss")
    return WeightedCrossEntropyLoss(
        class_weights=class_weights,
        ignore_index=int(loss_config.get("ignore_index", -1)),
        label_smoothing=float(loss_config.get("label_smoothing", 0.0)),
    )
