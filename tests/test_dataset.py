import anndata as ad
import numpy as np
import pandas as pd
import pytest

from merge.data.dataset import SingleCellPerturbDataset
from merge.data.pairing import CellTypeStratifiedPairer


def _adata(metal):
    matrix = np.ones((4, 3), dtype=np.float32)
    obs = pd.DataFrame(
        {"cell_type": ["A", "A", "B", "B"], "sample_id": ["s1"] * 4, "metal_class": [metal] * 4}
    )
    adata = ad.AnnData(X=matrix, obs=obs)
    adata.uns["sample_id_order"] = ["s1"]
    return adata


def _dataset(pert=None):
    pert = _adata("M") if pert is None else pert
    ctrl = _adata("Ctr")
    return SingleCellPerturbDataset(
        pert,
        ctrl,
        np.zeros((1, 5)),
        {"s1": np.ones((2, 3), dtype=np.int64)},
        {"A": 0, "B": 1},
        CellTypeStratifiedPairer(ctrl),
    )


def test_dataset_len():
    assert len(_dataset()) == 4


def test_dataset_item_keys_and_dtypes():
    item = _dataset()[0]
    assert {"x_ctrl", "metal_feat", "labels", "cell_type_idx", "sample_id"} <= set(item)
    assert str(item["x_ctrl"].dtype) == "torch.float32"
    assert str(item["labels"].dtype) == "torch.int64"


def test_dataset_uses_metal_feature_id_when_present():
    pert = _adata("Mg")
    pert.obs["sample_id"] = ["Mg_5"] * pert.n_obs
    pert.obs["metal_feature_id"] = ["Mg"] * pert.n_obs
    pert.uns["sample_id_order"] = ["Mg"]
    ctrl = _adata("Ctr")
    dataset = SingleCellPerturbDataset(
        pert,
        ctrl,
        np.array([[1, 2, 3, 4, 5]], dtype=np.float32),
        {"Mg_5": np.ones((2, 3), dtype=np.int64)},
        {"A": 0, "B": 1},
        CellTypeStratifiedPairer(ctrl),
    )
    item = dataset[0]
    assert np.allclose(item["metal_feat"].numpy(), [1, 2, 3, 4, 5])
    assert item["sample_id"] == "Mg_5"
    assert item["metal_feature_id"] == "Mg"


def test_dataset_rejects_missing_feature_vector():
    pert = _adata("M")
    pert.obs["metal_feature_id"] = "missing"
    dataset = _dataset(pert)
    with pytest.raises(KeyError, match="missing"):
        dataset[0]


def test_pairer_requires_annotated_cell_type():
    pairer = CellTypeStratifiedPairer(_adata("Ctr"))
    with pytest.raises(KeyError):
        pairer.sample_paired_ctrl("not_annotated")
