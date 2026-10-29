"""HronKNN (2a) against Hron et al. (2010) section 3.1 and its section 4 example."""
from pathlib import Path
import numpy as np
import pytest
from longitudinal_impute.baselines.hron import HronKNN
from longitudinal_impute.errors import BaselineError, NotFittedError

FIXTURES = Path(__file__).parent / 'fixtures'


def test_ratio_of_medians_is_not_median_of_ratios():
    donor = np.array([[100., 200., 1., 9.]])
    query = np.array([[10., 30., 60., np.nan]])
    value = HronKNN(1).fit(donor).predict(query)[0, 3]
    assert value == pytest.approx(9 * 30 / 100)
    assert value != pytest.approx(9 * np.median(query[0, :3] / donor[0, :3]))
    assert HronKNN(1, adjustment='sum').fit(donor).predict(query)[0, 3] == pytest.approx(9 * 100 / 301)


def test_available_donors_and_scale_invariance():
    # Row 0 lacks feature 3 but is observed on O_i + {2}, so (2a) may use it for feature 2.
    donors = np.array([[1., 2., 3., np.nan], [2., 4., 10., 8.], [4., 8., 24., 16.]])
    query = np.array([[2., 4., np.nan, np.nan]])
    out = HronKNN(1, ties='index').fit(donors).predict(query)
    assert out[0, 2] == pytest.approx(6)
    assert out[0, 3] == pytest.approx(8)
    np.testing.assert_allclose(HronKNN(1, ties='index').fit(donors).predict(query * 5), out * 5)


def test_expenditures_example_values_differ_from_table_1_at_k4():
    # Paper Table 1 reports 152.1 for kNN (Aitchison) with k = 4; the truth is 147.
    # This implementation gives 152.10 at k = 3 and 144.83 at k = 4 (see
    # tv2_phase1/REPORT_vi.md 4.4: an R impKNNa emulation gives 154.69 at k = 4).
    x = np.genfromtxt(FIXTURES / 'expenditures.csv', delimiter=',', skip_header=1)
    x[0, 2] = np.nan
    assert round(HronKNN(3).fit(x).predict(x)[0, 2], 2) == 152.10
    assert round(HronKNN(4).fit(x).predict(x)[0, 2], 2) == 144.83


def test_ties_group_takes_whole_tie_and_index_keeps_k():
    # Donors 0-2 have equal parts on O_i, so all sit at Aitchison distance 0 from the query.
    donors = np.array([[2., 2., 1.], [3., 3., 6.], [5., 5., 20.], [1., 4., 7.]])
    query = np.array([[1., 1., np.nan]])
    index = HronKNN(1, ties='index').fit(donors)
    group = HronKNN(1, ties='group').fit(donors)
    assert index.predict(query)[0, 2] == pytest.approx(1 / 2)
    assert group.predict(query)[0, 2] == pytest.approx(np.median([1 / 2, 6 / 3, 20 / 5]))
    assert index.info_['tied_cells'] == group.info_['tied_cells'] == 1
    assert index.info_['donors_used_max'] == 1 and group.info_['donors_used_max'] == 3


def test_observed_cells_are_kept_exactly_and_no_nan_remains():
    rng = np.random.default_rng(4)
    x = rng.dirichlet(np.ones(5), 40)
    x[rng.random(x.shape) < .2] = np.nan
    x[:, 0] = np.where(np.isfinite(x).sum(axis=1) == 0, .5, x[:, 0])
    out = HronKNN().fit(x).predict(x)
    observed = np.isfinite(x)
    np.testing.assert_array_equal(out[observed], x[observed])
    assert np.isfinite(out).all() and (out > 0).all()


def test_clear_errors():
    with pytest.raises(BaselineError, match='strictly positive') as error:
        HronKNN().fit([[.5, .5], [.3, 0.]])
    assert error.value.context == {'row': 1, 'feature': 1, 'cells': 1}
    with pytest.raises(BaselineError, match='no donor') as error:
        HronKNN().fit([[1., np.nan]]).predict([[np.nan, 1.]])
    assert error.value.context == {'row': 0, 'feature': 0}
    with pytest.raises(BaselineError, match='no observed part'):
        HronKNN().fit([[1., 2.]]).predict([[np.nan, np.nan]])
    with pytest.raises(NotFittedError):
        HronKNN().predict([[1., np.nan]])
    with pytest.raises(BaselineError):
        HronKNN(ties='random')


def test_rescaling_divides_first_so_tiny_scales_cancel():
    out = HronKNN(1).fit([[1e-320, 1e-320]]).predict([[.2, np.nan]])
    np.testing.assert_allclose(out, [[.2, .2]], rtol=1e-12)
    out = HronKNN(1, adjustment='sum').fit([[1e-320, 1e-320]]).predict([[.2, np.nan]])
    np.testing.assert_allclose(out, [[.2, .2]], rtol=1e-12)


def test_unrepresentable_rescaled_value_is_a_numerical_error():
    from longitudinal_impute.errors import NumericalError
    model = HronKNN(1).fit([[1e-300, 1e300]])
    with pytest.raises(NumericalError, match='not finite') as error:
        model.predict([[1e300, np.nan]])           # the answer, 1e900, is not a float64
    assert error.value.context == {'row': 0, 'feature': 1}
    assert not hasattr(model, 'info_')
