import numpy as np

from missing_imputation.data.closure import allocate_with_lock, validate_closure, compute_other


def test_allocate_with_lock_closure():
    proportions = np.array([[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]])
    totals = np.array([100, 200])
    lock_mask = np.array([[True, False, False], [False, True, False]])
    locked = np.array([[30, 0, 0], [0, 40, 0]])

    result = allocate_with_lock(proportions, totals, lock_mask, locked)

    assert result[0, 0] == 30
    assert result[0].sum() == 100
    assert result[1].sum() == 200
    assert np.all(result >= 0)
    assert result.dtype == np.int64


def test_allocate_locks_confident_zero():
    proportions = np.array([[0.4, 0.6]])
    totals = np.array([50])
    lock_mask = np.array([[False, True]])
    locked = np.array([[0, 0]])

    result = allocate_with_lock(proportions, totals, lock_mask, locked)
    assert result[0, 1] == 0
    assert result[0, 0] == 50


def test_validate_closure_with_locked_cells():
    counts = np.array([[10, 0, 40], [5, 5, 10]])
    totals = np.array([50, 20])
    other = compute_other(counts, totals)
    ok, n = validate_closure(counts, totals, other)
    assert ok
    assert n == 0