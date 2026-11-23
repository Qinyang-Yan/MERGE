import anndata as ad
import numpy as np
import pandas as pd

from merge.data.preprocessing import preprocess_with_frozen_split


class Logger:
    def info(self, *args):
        pass


def test_gene_fitting_excludes_validation_and_test_perturbations(monkeypatch):
    ctrl = ad.AnnData(
        X=np.ones((2, 4), dtype=np.float32),
        obs=pd.DataFrame({"sample_id": ["Ctr", "Ctr"], "cell_type": ["A", "A"], "metal_class": ["Ctr", "Ctr"]}),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    perturbed = ad.AnnData(
        X=np.ones((6, 4), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "sample_id": ["Mg_1"] * 2 + ["Mg_2"] * 2 + ["Mg_3"] * 2,
                "cell_type": ["A"] * 6,
                "metal_class": ["Mg"] * 6,
            }
        ),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    captured = {}

    def stub_fit(training_data, **kwargs):
        captured["sample_ids"] = set(training_data.obs["sample_id"].astype(str))
        return ["g1", "g2"]

    monkeypatch.setattr("merge.data.preprocessing.qc_filter_cells", lambda adata, **kwargs: adata)
    monkeypatch.setattr("merge.data.preprocessing.normalize_log1p", lambda adata, **kwargs: adata)
    monkeypatch.setattr("merge.data.preprocessing.fit_gene_space", stub_fit)
    split = {
        "train_sample_ids": ["Mg_1"],
        "val_sample_ids": ["Mg_2"],
        "test_sample_ids": ["Mg_3"],
    }
    output = preprocess_with_frozen_split(
        {"Ctr": ctrl, "Mg": perturbed}, {"ctrl_class": "Ctr"}, split, Logger()
    )
    assert captured["sample_ids"] == {"Ctr", "Mg_1"}
    assert output["Mg"].var_names.tolist() == ["g1", "g2"]


def _condition(name, sample_ids, genes, matrix):
    observations = [sample_id for sample_id in sample_ids for _ in range(2)]
    return ad.AnnData(
        X=np.vstack([matrix] * len(observations)).astype(np.float32),
        obs=pd.DataFrame(
            {
                "sample_id": observations,
                "cell_type": ["A"] * len(observations),
                "metal_class": [name] * len(observations),
            }
        ),
        var=pd.DataFrame(index=genes),
    )


def test_mixed_count_layouts_preserve_raw_counts(monkeypatch):
    genes = ["g0", "g1", "g2", "g3"]
    raw = np.array([1, 2, 3, 4], dtype=np.float32)
    ctrl = _condition("Ctr", ["Ctr"], genes, raw)
    perturbed = _condition("Mg", ["Mg_1", "Mg_2", "Mg_3"], genes, np.log1p(raw))
    perturbed.layers["counts"] = np.vstack([raw] * perturbed.n_obs).astype(np.float32)
    captured = {}

    def stub_fit(training_data, **kwargs):
        keep = training_data.obs["sample_id"].astype(str) == "Mg_1"
        counts = training_data.layers["counts"]
        captured["counts"] = counts[keep].toarray() if hasattr(counts[keep], "toarray") else counts[keep]
        return ["g0", "g1"]

    monkeypatch.setattr("merge.data.preprocessing.qc_filter_cells", lambda adata, **kwargs: adata)
    monkeypatch.setattr("merge.data.preprocessing.fit_gene_space", stub_fit)
    split = {"train_sample_ids": ["Mg_1"], "val_sample_ids": ["Mg_2"], "test_sample_ids": ["Mg_3"]}
    preprocess_with_frozen_split({"Ctr": ctrl, "Mg": perturbed}, {"ctrl_class": "Ctr"}, split, Logger())
    assert np.allclose(captured["counts"], np.vstack([raw, raw]))


def test_heldout_only_object_does_not_define_candidate_genes(monkeypatch):
    train_genes = ["g0", "g1", "g2"]
    ctrl = _condition("Ctr", ["Ctr"], train_genes, np.ones(3))
    magnesium = _condition("Mg", ["Mg_1"], train_genes, np.ones(3))
    heldout = _condition("Zn", ["Zn_2", "Zn_3"], [*train_genes, "heldout_only"], np.ones(4))
    captured = {}

    def stub_fit(training_data, **kwargs):
        captured["genes"] = training_data.var_names.astype(str).tolist()
        return ["g0", "g1"]

    monkeypatch.setattr("merge.data.preprocessing.qc_filter_cells", lambda adata, **kwargs: adata)
    monkeypatch.setattr("merge.data.preprocessing.normalize_log1p", lambda adata, **kwargs: adata)
    monkeypatch.setattr("merge.data.preprocessing.fit_gene_space", stub_fit)
    split = {"train_sample_ids": ["Mg_1"], "val_sample_ids": ["Zn_2"], "test_sample_ids": ["Zn_3"]}
    output = preprocess_with_frozen_split(
        {"Ctr": ctrl, "Mg": magnesium, "Zn": heldout}, {"ctrl_class": "Ctr"}, split, Logger()
    )
    assert "heldout_only" not in captured["genes"]
    assert output["Zn"].var_names.tolist() == ["g0", "g1"]
