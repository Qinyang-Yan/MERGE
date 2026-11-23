"""MERGE model assembly."""

from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import nn

from .classification_head import DirectionClassificationHead
from .genecompass_wrapper import GeneCompassWrapper
from .metal_encoder import MetalEncoder


class MERGE(nn.Module):
    """Predict transcriptional response directions from cell and material inputs."""

    def __init__(
        self,
        gene_compass: GeneCompassWrapper,
        metal_encoder: MetalEncoder,
        classification_head: DirectionClassificationHead,
        fusion: str = "add",
    ) -> None:
        super().__init__()
        if fusion != "add":
            raise ValueError("MERGE supports element-wise additive fusion only")
        if gene_compass.hidden_dim != metal_encoder.output_dim:
            raise ValueError("Cell and material embedding dimensions must match")
        if classification_head.hidden_dim != gene_compass.hidden_dim:
            raise ValueError("The classification head must match the cell embedding dimension")
        self.gene_compass = gene_compass
        self.metal_encoder = metal_encoder
        self.classification_head = classification_head
        self.fusion = fusion

    def forward(self, x_ctrl: torch.Tensor, metal_feat: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``(batch, genes, 3)``.

        ``x_ctrl`` contains normalized log1p control-cell expression and
        ``metal_feat`` contains transformed, training-standardized descriptors.
        """
        cell_embedding = self.gene_compass(x_ctrl)
        metal_embedding = self.metal_encoder(metal_feat)
        return self.classification_head(cell_embedding + metal_embedding)


def build_model(
    model_config: dict,
    n_genes: int,
    gene_names: Sequence[str],
    checkpoint_path: Optional[str] = None,
) -> MERGE:
    """Build MERGE and optionally load a complete checkpoint.

    GeneCompass assets are mandatory. Missing assets or incompatible checkpoint
    tensors raise an error instead of substituting a different model.
    """
    if n_genes <= 0:
        raise ValueError("n_genes must be positive")
    gc_config = dict(model_config["genecompass"])
    gene_compass = GeneCompassWrapper(
        pretrained_dir=str(gc_config["pretrained_dir"]),
        token_dict_path=str(gc_config["token_dict_path"]),
        prior_knowledge_dir=str(gc_config["prior_knowledge_dir"]),
        freeze_layers=int(gc_config.get("freeze_layers", 0)),
        hidden_dim=int(gc_config.get("hidden_dim", 768)),
        max_tokens=int(gc_config.get("max_tokens", 2048)),
        gene_names=[str(gene) for gene in gene_names],
    )
    metal_encoder = MetalEncoder(**dict(model_config["metal_encoder"]))
    head_config = dict(model_config["classification_head"])
    head_config["n_genes"] = int(n_genes)
    classification_head = DirectionClassificationHead(**head_config)
    model = MERGE(
        gene_compass=gene_compass,
        metal_encoder=metal_encoder,
        classification_head=classification_head,
        fusion=str(model_config.get("fusion", "add")),
    )
    if checkpoint_path is not None:
        load_model_checkpoint(model, checkpoint_path)
    return model


def load_model_checkpoint(model: MERGE, checkpoint_path: str, map_location: str = "cpu") -> dict:
    """Load all model tensors strictly and return checkpoint metadata."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint does not contain a model state dictionary: {path}")
    model.load_state_dict(state_dict, strict=True)
    return checkpoint
