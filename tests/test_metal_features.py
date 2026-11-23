import numpy as np
import pandas as pd
import pytest

from merge.data.metal_features import (
    aggregate_features_by_metal_class,
    apply_scaler,
    clean_metal_features,
    fit_scaler,
    log_transform,
)


def _df():
    return pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "metal_class": ["M", "M"],
            "Ecorr": ["−1.0", "-2.0"],
            "icorr": [1e-4, 1e-5],
            "Rp": [10.0, 100.0],
            "f_MAX": [2.0, 8.0],
            "Phase_MAX": [-40.0, -50.0],
        }
    )


def test_log_transform_positive():
    output = log_transform(_df())
    assert np.isclose(output.loc[0, "icorr"], -4.0)


def test_log_transform_negative_raises():
    table = _df()
    table.loc[0, "Rp"] = -1
    with pytest.raises(ValueError):
        log_transform(table)


def test_scaler_roundtrip():
    table = _df()
    stats = fit_scaler(table)
    array = apply_scaler(table, stats)
    assert np.allclose(array.mean(axis=0), 0.0, atol=1e-6)


def test_unicode_minus():
    output = log_transform(_df())
    assert output.loc[0, "Ecorr"] == -1.0


def test_invalid_phase_is_rejected():
    table = _df()
    table.loc[0, "Phase_MAX"] = -400.0
    with pytest.raises(ValueError, match="Correct the source"):
        clean_metal_features(table)


def test_aggregate_features_by_metal_class_uses_geometric_mean_for_log_cols():
    table = pd.DataFrame(
        {
            "sample_id": ["Mg_5", "Mg_6"],
            "metal_class": ["Mg", "Mg"],
            "Ecorr": [-2.0, -4.0],
            "icorr": [1e-4, 1e-6],
            "Rp": [10.0, 1000.0],
            "f_MAX": [1.0, 100.0],
            "Phase_MAX": [-40.0, -60.0],
        }
    )
    output = aggregate_features_by_metal_class(table)
    assert output.loc[0, "sample_id"] == "Mg"
    assert output.loc[0, "metal_feature_id"] == "Mg"
    assert np.isclose(output.loc[0, "Ecorr"], -3.0)
    assert np.isclose(output.loc[0, "icorr"], 1e-5)
    assert np.isclose(output.loc[0, "Rp"], 100.0)
    assert np.isclose(output.loc[0, "f_MAX"], 10.0)
