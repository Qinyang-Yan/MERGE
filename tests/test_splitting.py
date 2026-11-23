import pandas as pd
import pytest

from merge.data.splitting import make_sample_split, validate_sample_split


def _metadata():
    return pd.DataFrame(
        {
            "sample_id": [f"{metal}_{index}" for metal in ["Mg", "Zn"] for index in [1, 2, 3]],
            "metal_class": [metal for metal in ["Mg", "Zn"] for _ in [1, 2, 3]],
        }
    )


def test_sample_split_is_deterministic_and_disjoint():
    first = make_sample_split(_metadata())
    second = make_sample_split(_metadata().sample(frac=1, random_state=7))
    assert first == second
    assert first["test_sample_ids"] == ["Mg_3", "Zn_3"]
    assert first["val_sample_ids"] == ["Mg_2", "Zn_2"]
    validate_sample_split(first)


def test_sample_split_rejects_overlap():
    split = {"train_sample_ids": ["a"], "val_sample_ids": ["a"], "test_sample_ids": ["b"]}
    with pytest.raises(ValueError, match="leakage"):
        validate_sample_split(split)
