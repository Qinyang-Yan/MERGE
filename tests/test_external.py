import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from merge.data.external import align_to_training_genes, build_external_targets
from merge.inference import MaterialControlDataset, run_material_inference


def test_external_alignment_masks_absent_genes():
    adata = ad.AnnData(
        X=np.array([[1, 2], [3, 4]], dtype=np.float32),
        var=pd.DataFrame(index=["A", "C"]),
        obs=pd.DataFrame(index=["c1", "c2"]),
    )
    aligned, available = align_to_training_genes(adata, ["A", "B", "C"])
    assert aligned.shape == (2, 3)
    assert available.tolist() == [True, False, True]
    assert np.allclose(aligned.X.toarray(), [[1, 0, 2], [3, 0, 4]])


def test_external_targets_mask_unmeasured_gene_and_small_groups():
    processed = {
        "new": ad.AnnData(
            X=np.ones((3, 2), dtype=np.float32),
            obs=pd.DataFrame({"cell_type": ["A", "A", "B"]}),
        )
    }
    labels, responses, counts = build_external_targets(
        processed,
        {"new": np.array([True, False])},
        np.zeros((2, 2), dtype=np.float32),
        ["A", "B"],
        threshold=0.25,
        min_cells=2,
    )
    assert labels.shape == (1, 2, 2)
    assert labels[0, 0].tolist() == [2, -1]
    assert np.isnan(responses[0, 0, 1])
    assert (labels[0, 1] == -1).all()
    assert counts.tolist() == [[2, 1]]


def test_material_control_sampling_is_seeded():
    control = ad.AnnData(
        X=np.arange(20, dtype=np.float32).reshape(10, 2),
        obs=pd.DataFrame({"cell_type": ["A"] * 5 + ["B"] * 5}),
    )
    features = np.zeros((1, 5), dtype=np.float32)
    sizes = np.array([[3, 2]])
    first = MaterialControlDataset(control, features, ["new"], ["A", "B"], sizes, seed=42)
    second = MaterialControlDataset(control, features, ["new"], ["A", "B"], sizes, seed=42)
    assert np.array_equal(first.ctrl_indices, second.ctrl_indices)
    assert len(first) == 5


def test_material_inference_aggregates_votes_and_probabilities():
    control = ad.AnnData(
        X=np.ones((6, 2), dtype=np.float32),
        obs=pd.DataFrame({"cell_type": ["A", "A", "A", "B", "B", "B"]}),
    )

    class ConstantModel(nn.Module):
        def forward(self, expression, features):
            logits = torch.tensor([0.0, 2.0, 0.0], device=expression.device)
            return logits.expand(expression.shape[0], 2, 3)

    dataset = MaterialControlDataset(
        control,
        np.zeros((1, 5), dtype=np.float32),
        ["new"],
        ["A", "B"],
        np.array([[3, 2]]),
        seed=42,
    )
    output = run_material_inference(
        ConstantModel(),
        DataLoader(dataset, batch_size=2),
        n_materials=1,
        n_celltypes=2,
        n_genes=2,
        device="cpu",
    )
    assert output["predictions"].shape == (1, 2, 2)
    assert (output["predictions"] == 1).all()
    assert output["group_prediction_counts"].tolist() == [[3, 2]]
    assert np.allclose(output["mean_probabilities"].sum(axis=-1), 1.0)
