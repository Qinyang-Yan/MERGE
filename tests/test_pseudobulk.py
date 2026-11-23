import anndata as ad
import numpy as np
import pandas as pd

from merge.data.pseudobulk import compute_pseudobulk


def _adata():
    matrix = np.array([[1, 2], [3, 4], [10, 20], [30, 40], [5, 5]], dtype=np.float32)
    obs = pd.DataFrame({"cell_type": ["A", "A", "B", "B", "C"], "sample_id": ["s1"] * 5})
    return ad.AnnData(X=matrix, obs=obs)


def test_pseudobulk_shape():
    matrix, metadata = compute_pseudobulk(_adata(), min_cells_per_group=1)
    assert matrix.shape == (3, 2)
    assert len(metadata) == 3


def test_pseudobulk_aggregation_mean():
    matrix, metadata = compute_pseudobulk(_adata(), min_cells_per_group=1)
    row = metadata.index[metadata["cell_type"] == "A"][0]
    assert np.allclose(matrix[row], [2, 3])


def test_min_cells_filter():
    matrix, metadata = compute_pseudobulk(_adata(), min_cells_per_group=2)
    assert "C" not in set(metadata["cell_type"])
    assert matrix.shape[0] == 2
