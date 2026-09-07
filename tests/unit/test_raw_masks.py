import numpy as np
import pandas as pd
import pytest

from missing_imputation.data.masks import RawMasks, create_raw_masks

VARIANTS = ["A", "B", "C"]


def _df(rows):
    return pd.DataFrame(rows)


def test_raw_masks_basic():
    df = _df(
        {
            "location": ["L1", "L1", "L2"],
            "date": ["2021-01-01", "2021-01-15", "2021-01-01"],
            "A": [1.0, np.nan, 0.0],
            "B": [np.nan, 2.0, 0.0],
            "C": [3.0, 0.0, np.nan],
        }
    )
    masks = create_raw_masks(df, VARIANTS)
    assert int(masks.M_target.sum()) == 3
    assert int(masks.M_observed.sum()) == 6
    assert not np.any(masks.M_observed.astype(bool) & masks.M_target.astype(bool))
    assert int(masks.M_padding.sum()) == 0
    assert np.all(masks.M_row == 1)


def test_expected_count_validation():
    df = _df({"location": ["L1"], "date": ["2021-01-01"], "A": [np.nan], "B": [1.0], "C": [np.nan]})
    masks = create_raw_masks(df, VARIANTS, expected_m_target_count=2)
    assert int(masks.M_target.sum()) == 2

    with pytest.raises(ValueError):
        create_raw_masks(df, VARIANTS, expected_m_target_count=5)


def test_observed_zero_is_observed_not_target():
    df = _df({"location": ["L1"], "date": ["2021-01-01"], "A": [0.0], "B": [0.0], "C": [0.0]})
    masks = create_raw_masks(df, VARIANTS)
    assert int(masks.M_observed.sum()) == 3
    assert int(masks.M_target.sum()) == 0


def test_save_load_roundtrip(tmp_path):
    df = _df(
        {
            "location": ["L1", "L2"],
            "date": ["2021-01-01", "2021-01-01"],
            "A": [1.0, np.nan],
            "B": [np.nan, 2.0],
            "C": [0.0, 0.0],
        }
    )
    masks = create_raw_masks(df, VARIANTS, raw_checksum="abc")
    path = tmp_path / "raw_masks.npz"
    masks.save(path)
    loaded = RawMasks.load(path)
    assert np.array_equal(loaded.M_target, masks.M_target)
    assert loaded.raw_checksum == "abc"
    assert loaded.variant_order == VARIANTS