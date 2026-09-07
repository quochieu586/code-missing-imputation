import numpy as np

from missing_imputation.occurrence.gating import (
    wilson_upper_bound,
    selective_false_zero_gate,
    apply_gate_to_targets,
)


def test_wilson_upper_bound_monotonic():
    assert wilson_upper_bound(0, 100) < wilson_upper_bound(5, 100)
    assert wilson_upper_bound(0, 0) == 1.0


def test_gate_selects_threshold():
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.7).astype(int)
    p = np.clip(y * 0.8 + rng.random(n) * 0.2, 0, 1)

    tau, diag = selective_false_zero_gate(
        y, p, [0.05, 0.1, 0.2, 0.3, 0.5], max_false_zero_rate=0.2, min_support=30
    )
    assert tau is not None
    assert diag["selected"]["ucb"] <= 0.2


def test_gate_abstains_when_no_threshold():
    rng = np.random.default_rng(1)
    n = 500
    y = np.ones(n, dtype=int)
    p = rng.random(n) * 0.01
    tau, _ = selective_false_zero_gate(
        y, p, [0.05, 0.1], max_false_zero_rate=0.05, min_support=50
    )
    assert tau is None


def test_apply_gate_partition():
    M_target = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    p = np.array([0.01, 0.9, 0.02])
    tau = {"V1": 0.05, "V2": None, "V3": 0.05}
    M_tz, M_gan, w_gan = apply_gate_to_targets(p, M_target, ["V1", "V2", "V3"], tau)
    assert np.array_equal(M_tz | M_gan, M_target)
    assert not np.any(M_tz & M_gan)
    assert M_tz[0, 0] == 1
    assert M_gan[0, 2] == 1
    assert M_tz[1, 1] == 0 and M_gan[1, 1] == 1