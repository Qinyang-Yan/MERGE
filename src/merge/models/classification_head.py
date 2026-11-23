"""Per-gene transcriptional direction classification head."""

import torch
from torch import nn


class DirectionClassificationHead(nn.Module):
    """Map a fused embedding to Down/Unchanged/Up logits for every gene."""

    def __init__(
        self,
        hidden_dim: int = 768,
        n_genes: int = 3000,
        mid_dim: int = 256,
        n_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_classes != 3:
            raise ValueError("MERGE direction prediction requires exactly three classes")
        if min(hidden_dim, n_genes, mid_dim) <= 0:
            raise ValueError("Model dimensions must be positive")
        self.n_genes = int(n_genes)
        self.n_classes = int(n_classes)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, n_genes * n_classes),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``(batch, n_genes, 3)``."""
        output = self.net(embedding)
        return output.view(-1, self.n_genes, self.n_classes)
