"""Panel construction for the GAN stage: mask/posterior routing and conditioning."""

import numpy as np
import pandas as pd
import pytest

from missing_imputation.gan.data import build_gan_panel

VARIANT_COLS = ["v1", "v2", "v3"]


@pytest.fixture
def panel():
    # Row order is deliberately not (location, date) sorted, so a build that
    # relies on row position instead of joining on keys will misroute cells.
    df = pd.DataFrame(
        {
            "location": ["B", "A", "B", "A"],
            "date": ["2020-01-15", "2020-01-01", "2020-01-01", "2020-01-15"],
            "v1": [5, 10, 0, 3],
            "v2": [0, 2, 7, 0],
            "v3": [1, 1, 1, 1],
            "other": [4, 7, 2, 6],
            "total_sequence": [10, 20, 10, 10],
        }
    )
    M_gan = np.zeros((4, len(VARIANT_COLS)), dtype=np.uint8)
    M_gan[0, 1] = 1  # (B, 2020-01-15, v2)
    M_gan[1, 2] = 1  # (A, 2020-01-01, v3)
    M_fixed = 1 - M_gan

    p_rows = np.zeros((4, len(VARIANT_COLS)))
    p_rows[0, 1] = 0.73
    p_rows[1, 2] = 0.21

    ds = build_gan_panel(df, M_fixed, M_gan, p_nonzero_rows=p_rows, variant_cols=VARIANT_COLS)
    return ds


def _cell(ds, location, date):
    return ds.location_index.index(location), [
        pd.Timestamp(t) for t in ds.time_index
    ].index(pd.Timestamp(date))


class TestPanelRouting:
    def test_counts_joined_by_location_and_date(self, panel):
        loc, t = _cell(panel, "B", "2020-01-15")
        np.testing.assert_array_equal(panel.counts_raw[loc, t], [5, 0, 1, 4])
        loc, t = _cell(panel, "A", "2020-01-01")
        np.testing.assert_array_equal(panel.counts_raw[loc, t], [10, 2, 1, 7])

    def test_gan_mask_joined_by_location_and_date(self, panel):
        loc, t = _cell(panel, "B", "2020-01-15")
        assert panel.M_gan[loc, t, 1] == 1
        loc, t = _cell(panel, "A", "2020-01-01")
        assert panel.M_gan[loc, t, 2] == 1

    def test_other_is_never_gan_editable(self, panel):
        n_variants = len(VARIANT_COLS)
        rows = panel.M_row == 1
        assert np.all(panel.M_gan[rows][:, n_variants] == 0)
        assert np.all(panel.M_fixed[rows][:, n_variants] == 1)


class TestOccurrenceConditioning:
    def test_posterior_reaches_gan_cells(self, panel):
        """Regression: the posterior used to be dropped, leaving the channel all zeros."""
        assert panel.p_nonzero.sum() > 0
        loc, t = _cell(panel, "B", "2020-01-15")
        assert panel.p_nonzero[loc, t, 1] == pytest.approx(0.73)
        loc, t = _cell(panel, "A", "2020-01-01")
        assert panel.p_nonzero[loc, t, 2] == pytest.approx(0.21)

    def test_locked_cells_carry_known_indicator(self, panel):
        loc, t = _cell(panel, "B", "2020-01-15")
        assert panel.p_nonzero[loc, t, 0] == 1.0  # observed positive
        assert panel.p_nonzero[loc, t, 2] == 1.0  # observed positive
        loc, t = _cell(panel, "B", "2020-01-01")
        assert panel.p_nonzero[loc, t, 0] == 0.0  # locked zero

    def test_channel_shape_matches_features(self, panel):
        assert panel.p_nonzero.shape == panel.counts_raw.shape

    def test_posterior_optional(self):
        df = pd.DataFrame(
            {
                "location": ["A"],
                "date": ["2020-01-01"],
                "v1": [5],
                "v2": [3],
                "v3": [1],
                "other": [1],
                "total_sequence": [10],
            }
        )
        M_gan = np.zeros((1, 3), dtype=np.uint8)
        ds = build_gan_panel(df, 1 - M_gan, M_gan, variant_cols=VARIANT_COLS)
        # No posterior supplied: every cell is locked, so the channel is the indicator.
        np.testing.assert_array_equal(ds.p_nonzero[0, 0], [1.0, 1.0, 1.0, 1.0])
