import logging

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from merge.data.loader import (
    assign_sample_ids,
    canonicalize_raw_sample_id,
    discover_h5ad_files,
    normalize_sample_id,
)


def test_normalize_sample_id_strategies():
    valid = ["Mg_alloy_1", "Mg_alloy_2"]
    assert normalize_sample_id("Mg_alloy", "Mg_alloy_1", valid) == "Mg_alloy_1"
    assert normalize_sample_id("Mg_alloy", "Mg alloy-2", valid) == "Mg_alloy_2"
    assert normalize_sample_id("Mg_alloy", "prefix_Mg_alloy_1_extra", valid) is None


def test_missing_sample_field_is_rejected():
    adata = ad.AnnData(X=np.ones((5, 2)), obs=pd.DataFrame(index=[f"c{i}" for i in range(5)]))
    features = pd.DataFrame({"sample_id": ["Mg_1", "Mg_2"], "metal_class": ["Mg", "Mg"]})
    with pytest.raises(KeyError):
        assign_sample_ids(adata, "Mg", features, logging.getLogger("test"))


def test_canonicalize_raw_sample_id_keeps_group_suffix():
    assert canonicalize_raw_sample_id("Mg", "Mg_6") == "Mg_6"
    assert canonicalize_raw_sample_id("Mg_alloy", "Mg alloy-5") == "Mg_alloy_5"


def test_assign_sample_ids_metal_class_feature_mode():
    adata = ad.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame({"orig.ident": ["Mg_5", "Mg_6", "Mg_6"]}, index=[f"c{i}" for i in range(3)]),
    )
    features = pd.DataFrame({"sample_id": ["Mg"], "metal_class": ["Mg"]})
    output = assign_sample_ids(
        adata, "Mg", features, logging.getLogger("test"), feature_matching_mode="metal_class"
    )
    assert set(output.obs["sample_id"]) == {"Mg_5", "Mg_6"}
    assert set(output.obs["metal_feature_id"]) == {"Mg"}


def test_discover_h5ad_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_h5ad_files(str(tmp_path))


def test_discover_h5ad_files_rejects_ambiguity(tmp_path):
    for name in ["Ctr.h5ad", "Mg.h5ad", "Mg_alloy.h5ad", "Mg_coat.h5ad", "Mo.h5ad", "Ti.h5ad", "Zn.h5ad", "Zn_old.h5ad"]:
        (tmp_path / name).touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        discover_h5ad_files(str(tmp_path))
