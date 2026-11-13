"""Unit tests for core transforms — acceptance tests T1-T6.

T1: clr(c*x) == clr(x)           (scale invariance)
T2: d_A via formula == d_A via clr (equivalence)
T3: clr(softmax(z)) == z - mean(z) (roundtrip)
T4: softmax(z + c*1) == softmax(z) (shift invariance)
T5: sum_k clr(x)_k == 0           (zero-sum)
T6: clr_subcomposition ignores NaN (subcomp safety)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from src.core.transforms import (
    clr, clr_inverse, clr_subcomposition,
    aitchison_distance, aitchison_distance_formula,
    ilr, ilr_inverse,
)


def test_T1_clr_scale_invariance():
    """T1: clr(c*x) == clr(x) for c in {0.1, 1, 1000}."""
    x = np.array([3.0, 7.0, 1.0, 5.0, 2.0])
    z_ref = clr(x)
    for c in [0.1, 1.0, 1000.0]:
        z_c = clr(c * x)
        assert np.allclose(z_ref, z_c, atol=1e-12), \
            f"T1 FAIL: c={c}, max diff={np.max(np.abs(z_ref - z_c))}"
    print("T1 PASS: clr scale invariance")


def test_T2_aitchison_equivalence():
    """T2: d_A via formula (2) == ||clr(x)-clr(y)|| ."""
    np.random.seed(42)
    x = np.random.dirichlet(np.ones(5))
    y = np.random.dirichlet(np.ones(5))

    d_formula = aitchison_distance_formula(x, y)
    d_clr = aitchison_distance(x, y)

    assert abs(d_formula - d_clr) < 1e-10, \
        f"T2 FAIL: formula={d_formula}, clr={d_clr}, diff={abs(d_formula - d_clr)}"
    print(f"T2 PASS: d_A formula={d_formula:.8f}, clr={d_clr:.8f}")


def test_T3_clr_roundtrip():
    """T3: clr(softmax(z)) == z - mean(z)."""
    z = np.array([1.5, -0.3, 2.1, -1.0])
    x = clr_inverse(z)
    z_back = clr(x)
    z_centered = z - np.mean(z)
    assert np.allclose(z_back, z_centered, atol=1e-10), \
        f"T3 FAIL: max diff={np.max(np.abs(z_back - z_centered))}"
    print("T3 PASS: clr roundtrip")


def test_T4_softmax_shift_invariance():
    """T4: softmax(z + c*1) == softmax(z)."""
    z = np.array([1.0, 2.0, 3.0])
    x_ref = clr_inverse(z)
    for c in [-5.0, 0.0, 100.0]:
        x_c = clr_inverse(z + c)
        assert np.allclose(x_ref, x_c, atol=1e-12), \
            f"T4 FAIL: c={c}, max diff={np.max(np.abs(x_ref - x_c))}"
    print("T4 PASS: softmax shift invariance")


def test_T5_clr_zero_sum():
    """T5: sum_k clr(x)_k == 0."""
    x = np.array([3.0, 7.0, 1.0, 5.0, 2.0])
    z = clr(x)
    assert abs(np.sum(z)) < 1e-12, \
        f"T5 FAIL: sum={np.sum(z)}"
    # Also test batch
    X = np.random.dirichlet(np.ones(8), size=10)
    Z = clr(X)
    sums = np.sum(Z, axis=1)
    assert np.allclose(sums, 0, atol=1e-12), \
        f"T5 FAIL batch: max sum={np.max(np.abs(sums))}"
    print("T5 PASS: clr zero-sum")


def test_T6_clr_subcomposition_safety():
    """T6: clr_subcomposition does not touch NaN positions."""
    x = np.array([3.0, np.nan, 1.0, np.nan, 2.0])
    idx = np.array([0, 2, 4])  # observed positions
    z = clr_subcomposition(x, idx)
    assert not np.any(np.isnan(z)), f"T6 FAIL: NaN in output"
    assert len(z) == 3, f"T6 FAIL: expected 3, got {len(z)}"
    # Verify it's same as clr on the subcomposition
    z_direct = clr(x[idx])
    assert np.allclose(z, z_direct, atol=1e-12)
    print("T6 PASS: clr_subcomposition NaN safety")


def test_ilr_roundtrip():
    """Bonus: ilr(ilr_inverse(z)) == z."""
    z = np.array([1.5, -0.3, 2.1, -1.0])
    x = ilr_inverse(z)
    z_back = ilr(x)
    assert np.allclose(z, z_back, atol=1e-10), \
        f"ilr roundtrip FAIL: max diff={np.max(np.abs(z - z_back))}"
    print("ilr roundtrip PASS")


def test_ilr_isometry():
    """Bonus: d_A(x,y) == d_E(ilr(x), ilr(y))."""
    np.random.seed(42)
    x = np.random.dirichlet(np.ones(5))
    y = np.random.dirichlet(np.ones(5))
    d_A = aitchison_distance(x, y)
    d_E = float(np.linalg.norm(ilr(x) - ilr(y)))
    assert abs(d_A - d_E) < 1e-10, \
        f"isometry FAIL: d_A={d_A}, d_E={d_E}"
    print(f"ilr isometry PASS: d_A={d_A:.8f}, d_E={d_E:.8f}")


def test_aitchison_scale_invariance():
    """d_A(c*x, y) == d_A(x, y)."""
    x = np.array([3.0, 7.0, 1.0, 5.0, 2.0])
    y = np.array([1.0, 2.0, 4.0, 3.0, 6.0])
    d_ref = aitchison_distance(x, y)
    for c in [0.1, 1.0, 1000.0]:
        d_c = aitchison_distance(c * x, y)
        assert abs(d_ref - d_c) < 1e-12, \
            f"scale inv FAIL: c={c}, diff={abs(d_ref - d_c)}"
    print("Aitchison scale invariance PASS")


if __name__ == "__main__":
    test_T1_clr_scale_invariance()
    test_T2_aitchison_equivalence()
    test_T3_clr_roundtrip()
    test_T4_softmax_shift_invariance()
    test_T5_clr_zero_sum()
    test_T6_clr_subcomposition_safety()
    test_ilr_roundtrip()
    test_ilr_isometry()
    test_aitchison_scale_invariance()
    print("\n=== ALL CORE TESTS PASSED ===")
