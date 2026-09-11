"""Occurrence features must be computable on the cells we actually predict.

The bug these tests lock down: on this dataset missingness is all-or-nothing per
(location, variant), so the lag/lead block is identically zero on every target
cell while being ~1 on almost every training cell. A model fitted on it cannot
transfer. The compositional block is the replacement and must be well-defined on
target cells.
"""

import numpy as np
import pandas as pd
import pytest

from missing_imputation.occurrence.features import (
    COMPOSITIONAL_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
    build_occurrence_features,
    build_target_prediction_features,
)

VARIANTS = ["A", "B", "C"]


@pytest.fixture
def frame():
    """Two locations, three dates. Missingness is per (location, variant):
    loc L1 never reports C; loc L2 never reports B."""
    return pd.DataFrame(
        {
            "location": ["L1"] * 3 + ["L2"] * 3,
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-15", "2020-01-29"] * 2
            ),
            "A": [10, 12, 11, 4, 5, 6],
            "B": [5, 0, 7, np.nan, np.nan, np.nan],
            "C": [np.nan, np.nan, np.nan, 3, 0, 2],
            "total_sequence": [20, 20, 20, 10, 10, 10],
        }
    )


def _observed(frame):
    return frame[VARIANTS].notna().to_numpy()


class TestTemporalFeaturesAreDeadOnTargets:
    def test_lag_lead_present_on_observed_cells(self, frame):
        obs = _observed(frame)
        f = build_occurrence_features(frame, VARIANTS, obs, use_temporal=True)
        idx = {n: i for i, n in enumerate(f.feature_names)}
        # Cells after the first timepoint of their location have a predecessor.
        assert f.X[:, idx["lag_avail"]].max() == 1.0

    def test_lag_lead_identically_zero_on_target_cells(self, frame):
        """The whole reason the temporal block is off by default."""
        obs = _observed(frame)
        target = ~obs
        X, rows, cols = build_target_prediction_features(
            frame, VARIANTS, target, frame["date"].min(), use_temporal=True
        )
        f = build_occurrence_features(frame, VARIANTS, obs, use_temporal=True)
        idx = {n: i for i, n in enumerate(f.feature_names)}
        for name in TEMPORAL_FEATURE_NAMES:
            assert np.all(X[:, idx[name]] == 0.0), (
                f"{name} is non-zero on target cells; if the data ever contains a "
                "(location, variant) pair that is partly observed this assumption "
                "changes and use_temporal should be revisited"
            )

    def test_temporal_block_is_off_by_default(self, frame):
        f = build_occurrence_features(frame, VARIANTS, _observed(frame))
        assert not any(n in f.feature_names for n in TEMPORAL_FEATURE_NAMES)


class TestCompositionalFeatures:
    def test_present_by_default(self, frame):
        f = build_occurrence_features(frame, VARIANTS, _observed(frame))
        for n in COMPOSITIONAL_FEATURE_NAMES:
            assert n in f.feature_names

    def test_well_defined_and_varying_on_target_cells(self, frame):
        obs = _observed(frame)
        X, _, _ = build_target_prediction_features(
            frame, VARIANTS, ~obs, frame["date"].min()
        )
        f = build_occurrence_features(frame, VARIANTS, obs)
        idx = {n: i for i, n in enumerate(f.feature_names)}
        rs = X[:, idx["residual_share"]]
        assert np.all(np.isfinite(rs))
        assert rs.min() >= 0.0
        # L1 rows leave 20-(10+5)=5 of 20 for C; L2 rows leave 10-(4+3)=3 of 10 for B
        assert rs.std() > 0, "residual_share must vary across target cells"

    def test_residual_share_is_leave_one_out(self, frame):
        """For an observed cell the cell's own count must be excluded."""
        obs = _observed(frame)
        f = build_occurrence_features(frame, VARIANTS, obs)
        idx = {n: i for i, n in enumerate(f.feature_names)}
        # row 0 of L1: A=10, B=5, C missing, total=20.
        # For cell (row 0, A): others observed = B=5 -> residual = (20-5)/20 = 0.75
        sel = (f.row_idx == 0) & (f.variant_idx == 0)
        assert f.X[sel, idx["residual_share"]][0] == pytest.approx(0.75)
        # For cell (row 0, B): others observed = A=10 -> residual = (20-10)/20 = 0.5
        sel = (f.row_idx == 0) & (f.variant_idx == 1)
        assert f.X[sel, idx["residual_share"]][0] == pytest.approx(0.5)

    def test_n_missing_other_excludes_self(self, frame):
        obs = _observed(frame)
        f = build_occurrence_features(frame, VARIANTS, obs)
        idx = {n: i for i, n in enumerate(f.feature_names)}
        # L1 row 0 has exactly one missing variant (C). For observed cell A the
        # count of OTHER missing variants is 1; for target cell C it is 0.
        sel = (f.row_idx == 0) & (f.variant_idx == 0)
        assert f.X[sel, idx["n_missing_other"]][0] == pytest.approx(1.0)

        X, trow, tcol = build_target_prediction_features(
            frame, VARIANTS, ~obs, frame["date"].min()
        )
        sel = (trow == 0) & (tcol == 2)
        assert X[sel, idx["n_missing_other"]][0] == pytest.approx(0.0)

    def test_circulation_excludes_own_location(self, frame):
        """A location that never reports a variant still gets its circulation."""
        obs = _observed(frame)
        f = build_occurrence_features(frame, VARIANTS, obs)
        idx = {n: i for i, n in enumerate(f.feature_names)}
        X, trow, tcol = build_target_prediction_features(
            frame, VARIANTS, ~obs, frame["date"].min()
        )
        # L1 never reports C, but L2 does (3, 0, 2 -> 2 of 3 positive in Jan).
        sel = trow < 3  # L1 rows
        circ_c = X[sel & (tcol == 2), idx["circulation"]]
        assert circ_c.size > 0
        assert np.all(circ_c > 0), "circulation must come from the other location"
        assert circ_c[0] == pytest.approx(2 / 3)


class TestTrainTargetParity:
    def test_same_feature_names_and_width(self, frame):
        obs = _observed(frame)
        f = build_occurrence_features(frame, VARIANTS, obs)
        X, _, _ = build_target_prediction_features(
            frame, VARIANTS, ~obs, frame["date"].min()
        )
        assert X.shape[1] == f.X.shape[1], (
            "training and target design matrices must have identical width; "
            "they are built by the same function for exactly this reason"
        )

    def test_availability_mask_hides_context(self, frame):
        """A fold must be able to hide its held-out cells from the context."""
        obs = _observed(frame)
        hidden = obs.copy()
        hidden[0, 1] = False  # hide L1 row0 B
        f_full = build_occurrence_features(frame, VARIANTS, obs)
        f_hidden = build_occurrence_features(
            frame, VARIANTS, obs, availability_mask=hidden
        )
        idx = {n: i for i, n in enumerate(f_full.feature_names)}
        sel = (f_full.row_idx == 0) & (f_full.variant_idx == 0)
        # With B hidden, cell A's residual_share should rise (less observed mass).
        assert (
            f_hidden.X[sel, idx["residual_share"]][0]
            > f_full.X[sel, idx["residual_share"]][0]
        )
        # The cell set itself is unchanged.
        assert np.array_equal(f_full.row_idx, f_hidden.row_idx)
