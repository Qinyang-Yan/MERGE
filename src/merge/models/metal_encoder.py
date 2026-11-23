"""Electrochemical descriptor encoder used by MERGE."""

from typing import Sequence

import torch
from torch import nn


class MetalEncoder(nn.Module):
    """Map five electrochemical descriptors into the cell embedding space."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: Sequence[int] = (128, 256),
        output_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim != 5:
            raise ValueError("MERGE expects exactly five electrochemical descriptors")
        dimensions = [input_dim, *hidden_dims, output_dim]
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("All encoder dimensions must be positive")
        layers = []
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.extend(
                    [nn.LayerNorm(dimensions[index + 1]), nn.GELU(), nn.Dropout(dropout)]
                )
        self.net = nn.Sequential(*layers)
        self.output_dim = int(output_dim)

    def forward(self, metal_feat: torch.Tensor) -> torch.Tensor:
        """Encode a tensor with shape ``(batch, 5)``."""
        if metal_feat.ndim != 2 or metal_feat.shape[-1] != 5:
            raise ValueError(f"Expected metal features with shape (batch, 5), got {tuple(metal_feat.shape)}")
        return self.net(metal_feat)
