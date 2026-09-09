"""ZPGF weighting is applied to the GAN magnitude, after the GAN, before closure."""

import numpy as np
import pandas as pd
import pytest

from missing_imputation.gan.data import build_gan_panel
from missing_imputation.gan.clr import clr_transform

torch = pytest.importorskip("torch", reason="postprocess is imported alongside the GAN stack")

from missing_imputation.gan.postprocess import postprocess_gan_output  # noqa: E402

VARIANT_COLS = ["v1", "v2", "v3"]


def _panel(w_gan_rows=None):
    df = pd.DataFrame(
        {
            "location": ["A", "A"],
            "date": ["2020-01-01", "2020-01-15"],
            "v1": [10, 8],
            "v2": [20, 24],
            "v3": [30, 28],
            "other": [40, 40],
            "total_sequence": [100, 100],
        }
    )
    # v2 at both timepoints is GAN-editable; everything else is locked.
    M_gan = np.zeros((2, 3), dtype=np.uint8)
    M_gan[:, 1] = 1
    M_fixed = 1 - M_gan
    return df, build_gan_panel(
        df,
        M_fixed,
        M_gan,
        w_gan_rows=w_gan_rows,
        variant_cols=VARIANT_COLS,
    )


def _clr_of(ds):
    flat = ds.counts_raw.reshape(-1, len(ds.feature_names))
    clr, _ = clr_transform(flat / flat.sum(axis=1, keepdims=True), ds.pseudo_count)
    return clr.reshape(ds.counts_raw.shape)


class TestSoftWeightApplication:
    def test_weight_one_is_identity_on_locked_variant_cells(self):
        df, ds = _panel(w_gan_rows=np.ones((2, 3)))
        out = postprocess_gan_output(_clr_of(ds), ds)
        locked = ds.M_fixed.astype(bool).copy()
        locked[..., -1] = False  # residual is free by design
        np.testing.assert_array_equal(out[locked], ds.counts_raw[locked].astype(np.int64))

    def test_residual_absorbs_the_down_weighted_mass(self):
        w_low = np.ones((2, 3))
        w_low[:, 1] = 0.01
        _, ds = _panel(w_gan_rows=w_low)
        out = postprocess_gan_output(_clr_of(ds), ds)
        other_idx = len(ds.feature_names) - 1
        assert out[0, 0, other_idx] > ds.counts_raw[0, 0, other_idx], (
            "mass removed from the gated cell must flow into the residual"
        )

    def test_low_weight_shrinks_the_gan_cell(self):
        w_low = np.ones((2, 3))
        w_low[:, 1] = 0.01
        _, ds_low = _panel(w_gan_rows=w_low)
        _, ds_one = _panel(w_gan_rows=np.ones((2, 3)))

        out_low = postprocess_gan_output(_clr_of(ds_low), ds_low)
        out_one = postprocess_gan_output(_clr_of(ds_one), ds_one)

        gan_idx = 1
        assert out_low[0, 0, gan_idx] < out_one[0, 0, gan_idx], (
            "a near-zero ZPGF weight must pull the imputed magnitude down"
        )

    def test_disabling_weights_reproduces_unweighted_output(self):
        w_low = np.ones((2, 3))
        w_low[:, 1] = 0.01
        _, ds = _panel(w_gan_rows=w_low)
        clr = _clr_of(ds)
        weighted = postprocess_gan_output(clr, ds, apply_soft_weights=True)
        ungated = postprocess_gan_output(clr, ds, apply_soft_weights=False)
        assert not np.array_equal(weighted, ungated)

    def test_closure_and_locks_hold_under_weighting(self):
        w_low = np.ones((2, 3))
        w_low[:, 1] = 0.05
        df, ds = _panel(w_gan_rows=w_low)
        out = postprocess_gan_output(_clr_of(ds), ds)

        for loc in range(out.shape[0]):
            for t in range(out.shape[1]):
                if ds.M_row[loc, t] == 0:
                    continue
                assert out[loc, t].sum() == int(ds.total_sequence[loc, t])
        assert np.all(out >= 0)
        locked = ds.M_fixed.astype(bool).copy()
        locked[..., -1] = False
        np.testing.assert_array_equal(out[locked], ds.counts_raw[locked].astype(np.int64))

    def test_default_weights_are_one_when_not_supplied(self):
        _, ds = _panel(w_gan_rows=None)
        assert np.all(ds.w_gan == 1.0)
