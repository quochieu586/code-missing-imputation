"""A18/R2 experiment: MICoDa with an Other column, parity with Hron/Tsagris budget.

Locked decisions (2026-10-30): variant opt-in via constructor/config `other=True`;
A17/R1 stays the default contract (D07). R2 shares the residual between missing
targets and Other by proposal weight, like data.restore_counts does for
Hron/Tsagris. Two-seed masked comparisons are evidence, not scientific ranking.
"""
import json

import numpy as np
import pytest

from longitudinal_impute import run
from longitudinal_impute.baselines.micoda import MICoDaCovid, prepare, restore
from test_run import call, write, FIXTURES, COVID_SCHEMA


def test_a18_prepare_appends_other_and_keeps_budget():
    # Complete rows: Other = (N - sum targets)/N is a real composition part.
    prepared = prepare([[20., 30.], [10., 15.]], [100., 100.], other=True)
    assert prepared.values.shape == (2, 3)
    assert prepared.other is True
    np.testing.assert_allclose(prepared.values.sum(axis=1), 1., rtol=0, atol=1e-10)
    np.testing.assert_allclose(prepared.values[:, -1], [0.5, 0.75], rtol=0, atol=1e-10)
    # Zero Other is a legitimate observation: replaced by delta, then closed.
    prepared = prepare([[20., 30.], [100., 0.]], [100., 100.], other=True)
    assert prepared.values[1, -1] > 0
    np.testing.assert_allclose(prepared.values.sum(axis=1), 1., rtol=0, atol=1e-10)
    # A17 default is untouched: no Other column, closure over targets only.
    prepared = prepare([[20., 30.], [10., 15.]], [100., 100.])
    assert prepared.values.shape == (2, 2) and prepared.other is False
    np.testing.assert_allclose(prepared.values.sum(axis=1), 1., rtol=0, atol=1e-10)


def test_r2_shares_residual_between_missing_target_and_other():
    # Truth [20, 30], N=100, Other=50. Hiding 30 leaves residual 80. R1 dumps all
    # 80 into the single missing target; R2 must split it by proposal weight so
    # Other keeps its share (parity with data.restore_counts for Hron/Tsagris).
    hidden = np.array([[False, True]])
    prepared = prepare([[20., 30.]], [100.], hidden, other=True)
    proposals = np.array([[0.2, 0.3, 0.5]])
    result = restore(prepared, proposals)
    assert 0 < result.values[0, 1] < 80.          # never the whole residual
    np.testing.assert_allclose(result.values[0, 1], 80. * (0.3 / 0.8), rtol=1e-9)
    implied_other = 100. - (20. + result.values[0, 1])
    np.testing.assert_allclose(implied_other, 80. * (0.5 / 0.8), rtol=1e-9)
    # Same input through R1 (A17): single missing target absorbs everything.
    prepared = prepare([[20., 30.]], [100.], hidden)
    result = restore(prepared, np.array([[0.2, 0.3]]))
    assert result.values[0, 1] == 80.


def test_adapter_other_variant_runs_and_records_r2_config():
    counts = np.array([[10., 20., 70.], [15., 25., 60.], [12., 23., 65.],
                       [11., 21., 68.], [9., 19., 72.], [13., 22., 65.]])
    totals = np.full(6, 100.)
    hidden = np.zeros_like(counts, bool)
    hidden[0, 1] = True
    result = MICoDaCovid(other=True).impute_counts(counts, totals, hidden)
    assert result.status == 'completed'
    assert result.diagnostics['config']['other_column'] is True
    assert result.diagnostics['config']['budget_policy'] == 'R2'
    filled = result.values[0, 1]
    assert np.isfinite(result.values).all()
    assert 0 < filled < 20.                        # residual = 100-(10+70) = 20; Other carries part
    np.testing.assert_array_equal(result.values[~hidden], counts[~hidden])
    # R1 contrast on the same input: the single missing target takes the whole 20.
    result_r1 = MICoDaCovid().impute_counts(counts, totals, hidden)
    assert result_r1.values[0, 1] == 20.
    assert result_r1.diagnostics['config']['budget_policy'] == 'R1'


def test_runner_space_label_follows_other_variant(tmp_path):
    code, out = call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA,
                     '[micoda]\nother = true\n', '--mask', 'none')
    assert code == 0
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert summary['native']['micoda']['space'] == '18_with_other'
    assert summary['native']['micoda']['status'] == 'ok'
    import shutil
    shutil.rmtree(out)                              # the runner never overwrites an output dir
    code, out = call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA,
                     '[micoda]\n', '--mask', 'none')
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert summary['native']['micoda']['space'] == '17_targets'
