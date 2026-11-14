"""T7 scale invariance, and a principled tie-break in the neighbour search."""
import numpy as np

from src.core.transforms import aitchison_distance
from src.stage_a.knn_aitchison import _find_neighbors_2a, knn_aitchison_impute


def test_neighbor_selection_breaks_ties_deterministically():
    # Every candidate sits at the same Aitchison distance from the target, so
    # the k chosen are decided entirely by the tie-break. np.argpartition runs
    # introselect, which returns an arbitrary subset of tied keys; the pool must
    # be large enough for it to actually reorder (it does not at n=10).
    pool = np.tile(np.array([1., 2., 4.]), (400, 1))
    mask = np.ones_like(pool, dtype=bool)
    got = _find_neighbors_2a(np.array([3., 6., 12.]), np.array([0, 1]), 2, pool, mask, 8)
    assert np.array_equal(np.sort(got), np.arange(8))


def test_rescaling_a_row_does_not_change_its_imputation():
    # T7 itself. This held before the tie-break change too -- it is here because
    # scripts/run_stage_a.py printed the number without asserting it.
    rng = np.random.default_rng(7)
    x = np.exp(rng.normal(size=(60, 6)))
    # Duplicate rows guarantee exact distance ties in the neighbor search.
    x[30:] = x[:30]
    m = rng.random((60, 6)) > .25
    x[~m] = np.nan
    target = int(np.flatnonzero(~m.any(1) == False)[0])
    assert (~m[target]).any(), 'test row must have something to impute'

    a = knn_aitchison_impute(x, m, k=8).X_imputed[target]
    scaled = x.copy()
    scaled[target] = x[target] * 5.
    b = knn_aitchison_impute(scaled, m, k=8).X_imputed[target]

    assert aitchison_distance(a, b) < 1e-10
