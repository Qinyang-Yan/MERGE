import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import anndata as ad
from merge.data.dataset import SingleCellPerturbDataset
from merge.data.pairing import CellTypeStratifiedPairer
from merge.models.classification_head import DirectionClassificationHead
from merge.models.full_model import MERGE
from merge.models.metal_encoder import MetalEncoder
from merge.training.losses import WeightedCrossEntropyLoss


class LightweightCellEncoder(nn.Module):
    """Small trainable encoder used only to test the MERGE assembly."""

    hidden_dim = 768

    def __init__(self, n_genes):
        super().__init__()
        self.projection = nn.Linear(n_genes, self.hidden_dim)

    def forward(self, expression):
        return self.projection(expression)


def test_full_pipeline_on_tiny_data():
    rng = np.random.default_rng(42)
    control = ad.AnnData(
        X=rng.random((10, 6), dtype=np.float32),
        obs=pd.DataFrame({"cell_type": ["A"] * 5 + ["B"] * 5, "sample_id": ["Ctr"] * 10}),
        var=pd.DataFrame(index=[f"g{i}" for i in range(6)]),
    )
    perturbed = ad.AnnData(
        X=rng.random((10, 6), dtype=np.float32),
        obs=pd.DataFrame({"cell_type": ["A"] * 5 + ["B"] * 5, "sample_id": ["Mg_1"] * 10}),
        var=pd.DataFrame(index=control.var_names),
    )
    perturbed.uns["sample_id_order"] = ["Mg_1"]
    labels = {"Mg_1": rng.integers(0, 3, size=(2, 6))}
    dataset = SingleCellPerturbDataset(
        perturbed,
        control,
        np.zeros((1, 5), dtype=np.float32),
        labels,
        {"A": 0, "B": 1},
        CellTypeStratifiedPairer(control),
    )
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    model = MERGE(
        LightweightCellEncoder(6),
        MetalEncoder(output_dim=768),
        DirectionClassificationHead(hidden_dim=768, n_genes=6),
    )
    logits = model(batch["x_ctrl"], batch["metal_feat"])
    loss = WeightedCrossEntropyLoss(torch.ones(3))(logits, batch["labels"])
    loss.backward()
    assert logits.shape == (2, 6, 3)
    assert torch.isfinite(loss)
