import numpy as np
from scipy.special import expit, logit

from missing_imputation.occurrence.gating import (
    wilson_upper_bound,
    selective_false_zero_gate,
    apply_gate_to_targets,
    compute_soft_fusion,
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


class TestComputeSoftFusion:
    """ZPGF gate (plan S18.3): w = sigmoid(logit(p) - b), capped hard-lock."""

    @staticmethod
    def _setup(n=200, cap=0.9, tau_abs=0.05, b=0.0, seed=0):
        rng = np.random.default_rng(seed)
        p = np.zeros((n, 1))
        target = np.ones((n, 1), dtype=np.uint8)
        p[:, 0] = rng.uniform(0.001, 0.6, n)
        return compute_soft_fusion(
            p_nonzero=p,
            target_mask=target,
            variant_names=["V"],
            bias_per_variant={"V": b},
            zero_lock_cap_fraction={"V": cap},
            tau_hard_abs=tau_abs,
        )

    def test_partition_is_exact(self):
        w, hard, soft, _ = self._setup()
        assert np.array_equal(hard | soft, np.ones((200, 1), dtype=np.uint8))
        assert not np.any(hard & soft)

    def test_hard_locked_fraction_respects_cap(self):
        # A tiny cap must keep almost everything available to the GAN.
        _, hard, _, diag = self._setup(cap=0.1, tau_abs=1.0)
        assert diag["V"]["hard_lock_fraction"] <= 0.1 + 1e-9
        # The old count-based cap drove this to 1.0 and starved the variant.
        assert hard.sum() < 200

    def test_never_locks_entire_variant(self):
        _, hard, soft, _ = self._setup(cap=1.0, tau_abs=1.0)
        assert soft.sum() > 0, "cap of 1.0 must still leave the top-weight cell free"

    def test_absolute_floor_also_binds(self):
        # cap allows 90% but tau_hard_abs=0 forbids locking anything.
        _, hard, soft, _ = self._setup(cap=0.9, tau_abs=0.0)
        assert hard.sum() == 0
        assert soft.sum() == 200

    def test_weight_matches_zpgf_formula(self):
        p = np.array([[0.2]])
        target = np.ones((1, 1), dtype=np.uint8)
        b = 0.75
        w, hard, soft, _ = compute_soft_fusion(
            p_nonzero=p,
            target_mask=target,
            variant_names=["V"],
            bias_per_variant={"V": b},
            zero_lock_cap_fraction={"V": 0.0},
            tau_hard_abs=0.05,
        )
        assert soft[0, 0] == 1
        np.testing.assert_allclose(w[0, 0], expit(logit(0.2) - b), rtol=1e-6)

    def test_bias_shifts_weight_towards_zero(self):
        p = np.full((50, 1), 0.3)
        target = np.ones((50, 1), dtype=np.uint8)
        kw = dict(
            p_nonzero=p,
            target_mask=target,
            variant_names=["V"],
            zero_lock_cap_fraction={"V": 0.0},
            tau_hard_abs=0.05,
        )
        w0, *_ = compute_soft_fusion(bias_per_variant={"V": 0.0}, **kw)
        w1, *_ = compute_soft_fusion(bias_per_variant={"V": 1.5}, **kw)
        assert w1[0, 0] < w0[0, 0], "a positive CRC bias must keep the gate zero-leaning"

    def test_non_target_cells_get_zero_weight(self):
        target = np.array([[1, 0]], dtype=np.uint8)
        p = np.array([[0.4, 0.9]])
        w, hard, soft, _ = compute_soft_fusion(
            p_nonzero=p,
            target_mask=target,
            variant_names=["A", "B"],
            bias_per_variant={"A": 0.0, "B": 0.0},
            zero_lock_cap_fraction={"A": 0.0, "B": 0.0},
        )
        assert w[0, 1] == 0.0 and hard[0, 1] == 0 and soft[0, 1] == 0