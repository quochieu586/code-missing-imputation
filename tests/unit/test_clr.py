import numpy as np
import pytest

from missing_imputation.gan.clr import clr_transform, inverse_clr_transform


def test_clr_proportions_roundtrip():
    """CLR is a bijection on the simplex (proportions)."""
    proportions = np.array([[0.1, 0.2, 0.7], [0.0, 0.5, 0.5], [0.3, 0.3, 0.4]])
    clr, pc = clr_transform(proportions)
    recovered = inverse_clr_transform(clr, pc, n_features=3)
    assert np.allclose(recovered, proportions, atol=1e-6)


def test_clr_zero_sum():
    proportions = np.array([[0.1, 0.2, 0.7]])
    clr, _ = clr_transform(proportions)
    assert np.allclose(clr.sum(axis=1), 0.0, atol=1e-10)


def test_auto_pseudo_count():
    counts = np.array([[2.0, 0.0, 8.0]])
    total_seq = np.array([10.0])
    _, pc = clr_transform(counts, total_sequence=total_seq)
    # min positive proportion = 0.2, pseudo_count = 0.1
    assert abs(pc - 0.1) < 1e-6


def test_clr_with_counts():
    counts = np.array([[1.0, 2.0, 3.0], [0.0, 5.0, 10.0]])
    total_seq = np.array([6.0, 15.0])
    clr, pc = clr_transform(counts, total_sequence=total_seq)
    recovered_props = inverse_clr_transform(clr, pc, n_features=3)
    expected_props = counts / total_seq[:, np.newaxis]
    assert np.allclose(recovered_props, expected_props, atol=1e-6)


def test_inverse_clr_normalises_over_features_on_a_3d_panel():
    """The GAN hands postprocess a (location, time, feature) panel.

    The composition is over features, i.e. the last axis. A fixed axis=1 would
    normalise across time and silently return something that is not a
    composition per timepoint.
    """
    rng = np.random.default_rng(0)
    props = rng.dirichlet(np.ones(4), size=(3, 5))  # (loc, time, feature)
    clr, pc = clr_transform(props)
    recovered = inverse_clr_transform(clr, pc)

    assert recovered.shape == props.shape
    np.testing.assert_allclose(recovered.sum(axis=-1), 1.0, atol=1e-9)
    np.testing.assert_allclose(recovered, props, atol=1e-6)


def test_panel_roundtrip_matches_flattened_roundtrip():
    rng = np.random.default_rng(1)
    props = rng.dirichlet(np.ones(4), size=(3, 5))
    clr_panel, pc = clr_transform(props)
    flat = props.reshape(-1, 4)
    clr_flat, pc_flat = clr_transform(flat)

    assert pc == pytest.approx(pc_flat)
    np.testing.assert_allclose(clr_panel.reshape(-1, 4), clr_flat, atol=1e-12)
    np.testing.assert_allclose(
        inverse_clr_transform(clr_panel, pc).reshape(-1, 4),
        inverse_clr_transform(clr_flat, pc_flat),
        atol=1e-12,
    )