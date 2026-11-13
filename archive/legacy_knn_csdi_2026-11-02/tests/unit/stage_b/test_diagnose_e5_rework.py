"""Phase A+ unit tests for the diagnostic building blocks used by the E5 rework.

Covers: scripts/diagnose_e5_rework.{independent_metrics,summary,frozen_hashes}
and src.evaluation.research_pilot.{evaluate,prevalence_bins}, plus the frozen
diffusion numerics src.stage_b.research_csdi.{schedule,forward_noise,reverse_step,
epsilon_loss,project_final,sample_latents}. These are independent of the
diagnostic script's main() side effects.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import diagnose_e5_rework as diag
from diagnose_e5_rework import independent_metrics, summary, frozen_hashes

from src.evaluation.research_pilot import (evaluate, prevalence_bins,
    numerical_diagnostics, failure_records, sha256)
from src.stage_b.research_csdi import (CSDICore, epsilon_loss, forward_noise,
    project_final, reverse_step, sample_latents, schedule)


DATA_SHA = 'bbb1aacdf46ebef345ca818f086d5586f4aded45475b00b6ee9bd89fc94a90ab'


# ---------------------------------------------------------------- summary
def test_summary_keys_and_values():
    s = summary(np.array([1., 2., 3., 4.]))
    assert set(s) == {'n', 'min', 'q01', 'median', 'q99', 'max', 'rms'}
    assert s['n'] == 4
    assert s['min'] == 1.0 and s['max'] == 4.0
    assert abs(s['rms'] - np.sqrt(np.mean(np.array([1., 2., 3., 4.])**2))) < 1e-12
    assert s['median'] == 2.5
    s2 = summary(np.array([[1., 2.], [3., 4.]]))
    assert s2['n'] == 4 and s2['median'] == 2.5


# ---------------------------------------------------------------- independent_metrics
def test_independent_metrics_matches_evaluate():
    rng = np.random.default_rng(0)
    truth = np.abs(rng.standard_normal((10, 17)))
    truth[2] = 0.0  # zero-sum row must be excluded
    hidden = rng.random((10, 17)) < 0.4
    hidden[5] = False
    pred = truth.copy()
    pred[hidden] = np.abs(rng.standard_normal(hidden.sum()))
    # ensure some hidden cells are truth==0 for clr_mae_zero_hidden
    truth[0, 5] = 0.0
    hidden[0, 5] = True
    delta = 0.5
    bins = {'rare': list(range(5)), 'middle': list(range(5, 11)), 'common': list(range(11, 17))}
    independent, _, eligible = independent_metrics(truth, pred, hidden)
    result = evaluate(truth, pred, hidden, np.where(hidden, np.nan, truth), delta, bins)
    for key in ('m2', 'm3', 'clr_mae_all_hidden', 'clr_mae_nonzero_hidden'):
        assert abs(independent[key] - result[key]) < 1e-12, key
    assert independent['m2'] > 0


def test_independent_metrics_zero_sum_excluded():
    truth = np.array([[1., 2., 3.], [0., 0., 0.], [2., 3., 4.]])
    hidden = np.array([[True, False, True], [True, True, True], [False, True, False]])
    pred = truth.copy()
    _, _, eligible = independent_metrics(truth, pred, hidden)
    assert eligible.tolist() == [True, False, True]


# ---------------------------------------------------------------- frozen_hashes
def test_frozen_hashes_includes_data_and_pdfs():
    hashes = frozen_hashes()
    assert isinstance(hashes, dict)
    data_key = 'data' + str(Path('/')).replace('/', '\\') + 'covariants.csv'
    assert data_key in hashes
    assert hashes[data_key] == DATA_SHA
    pdf_keys = [k for k in hashes if k.lower().endswith('.pdf')]
    assert any('Hron.pdf' in k for k in pdf_keys)
    assert any('phylo.pdf' in k for k in pdf_keys)


# ---------------------------------------------------------------- evaluate
def test_evaluate_zero_sum_prediction_kept():
    truth = np.array([[1., 2., 3.], [2., 3., 4.], [5., 6., 7.]])
    hidden = np.ones_like(truth, dtype=bool)
    query = np.where(hidden, np.nan, truth)
    pred = truth.copy()
    pred[0] = 0.0
    bins = {'rare': [0], 'middle': [1], 'common': [2]}
    result = evaluate(truth, pred, hidden, query, 0.5, bins)
    assert result['n_prediction_zero_sum_on_eligible'] == 1
    assert result['eligible_row_positions'] == [0, 1, 2]
    assert result['native_jsd_status'] == 'UNDEFINED_ZERO_PREDICTION_MASS'
    assert result['native_jsd_diagnostic'] is None
    assert result['observed_restoration_exact'] is True


def test_evaluate_rejects_hidden_visible_conflict():
    truth = np.array([[1., 2.]])
    hidden = np.array([[True, True]])
    query = np.array([[1., 2.]])  # both visible and hidden -> conflict
    bins = {'rare': [0], 'common': [1]}
    with pytest.raises(ValueError):
        evaluate(truth, truth, hidden, query, 0.5, bins)


# ---------------------------------------------------------------- prevalence_bins
def test_prevalence_denominator_excludes_natural_missing():
    train = np.array([[1., 0.], [np.nan, 1.], [np.nan, 0.]])
    _, rate = prevalence_bins(train, ['a', 'b'])
    assert rate[0] == 1.0
    assert abs(rate[1] - 1.0/3.0) < 1e-12


def test_prevalence_bins_labels():
    rng = np.random.default_rng(0)
    train = np.abs(rng.standard_normal((20, 17)))
    train[train < 0.5] = 0
    bins, _ = prevalence_bins(train, [f'v{i}' for i in range(17)])
    all_bins = bins['rare'] + bins['middle'] + bins['common']
    assert sorted(all_bins) == list(range(17))
    assert len(bins['rare']) == 5
    assert len(bins['middle']) == 6
    assert len(bins['common']) == 6


# ---------------------------------------------------------------- schedule
def test_schedule_terminal_alpha_bar_is_gaussian():
    beta, ab = schedule()
    assert len(beta) == 20
    assert abs(float(beta.max()) - 0.5) < 1e-12
    assert float(ab[-1]) < 0.02  # old schedule ended at .633
    assert float(ab[-1]) > 0
    assert torch.equal(ab, torch.cumprod(1 - beta, dim=0))


# ---------------------------------------------------------------- forward_noise
def test_forward_noise_is_linear_interpolation():
    beta, ab = schedule()
    z0 = torch.randn(2, 3, 16, dtype=torch.float64)
    noise = torch.randn(2, 3, 16, dtype=torch.float64)
    t = 5
    expected = ab[t].sqrt() * z0 + (1 - ab[t]).sqrt() * noise
    actual = forward_noise(z0, noise, ab[t])
    assert torch.equal(expected, actual)


# ---------------------------------------------------------------- reverse_step
def test_reverse_step_terminal_ignores_noise():
    beta, ab = schedule()
    x = torch.tensor([1., -2.], dtype=torch.float64)
    eps = x * 0.1
    a = reverse_step(x, eps, 0, beta, ab, torch.full_like(x, float('nan')))
    b = reverse_step(x, eps, 0, beta, ab)
    assert torch.equal(a, b)


def test_reverse_step_requires_noise_for_nonterminal():
    beta, ab = schedule()
    x = torch.ones(3, dtype=torch.float64)
    eps = torch.zeros_like(x)
    with pytest.raises(ValueError):
        reverse_step(x, eps, 1, beta, ab)


# ---------------------------------------------------------------- epsilon_loss
def test_epsilon_loss_unknown_values_not_arithmetic():
    noise = torch.tensor([1., 2., float('nan')], dtype=torch.float64)
    pred = torch.tensor([1., 1., float('nan')], dtype=torch.float64)
    eligible = torch.tensor([True, True, False])
    assert epsilon_loss(pred, noise, eligible).item() == 0.5


def test_epsilon_loss_rejects_no_labels():
    noise = torch.randn(3, dtype=torch.float64)
    pred = torch.zeros_like(noise)
    with pytest.raises(ValueError):
        epsilon_loss(pred, noise, torch.zeros(3, dtype=torch.bool))


# ---------------------------------------------------------------- project_final
def test_project_final_clr_zero_mean():
    z = torch.tensor([[1., 2., 4., 8.]], dtype=torch.float64)
    clr = project_final(z, 'clr16')
    assert abs(float(clr.sum())) < 1e-12


def test_project_final_ilr_identity():
    z = torch.randn(2, 16, dtype=torch.float64)
    assert torch.equal(project_final(z, 'ilr16'), z)


def test_project_final_hkglr_mean_zero():
    z = torch.tensor([[1., 2., 4., 8.]], dtype=torch.float64)
    hk = project_final(z, 'hkglr16_actual', (0, 2))
    assert abs(float(hk[:, [0, 2]].mean())) < 1e-12


# ---------------------------------------------------------------- sample_latents
def test_sample_latents_finite_and_shape():
    torch.set_num_threads(2)
    torch.manual_seed(21)
    model = CSDICore(8)
    model.eval()
    c = torch.randn(1, 1, 85, dtype=torch.float64)
    t = torch.zeros(1, 1, dtype=torch.float64)
    beta, ab = schedule()
    out, diag = sample_latents(model, c, t, beta, ab,
                               torch.zeros(1, 1, 8, dtype=torch.float64),
                               torch.tensor(1.0), 'ilr16', (), samples=3, seed=7)
    assert out.shape == (3, 1, 1, 8)
    assert torch.isfinite(out).all()
    assert diag['clipped_fraction'] == 0.0


def test_sample_latents_nonfinite_raises():
    torch.set_num_threads(2)
    class BadModel(torch.nn.Module):
        latent_dim = 8
        def forward(self, noisy, condition, times, steps):
            return torch.full_like(noisy, float('nan'))
    model = BadModel()
    model.eval()
    beta, ab = schedule()
    with pytest.raises(FloatingPointError):
        sample_latents(model, torch.randn(1, 1, 85, dtype=torch.float64),
                       torch.zeros(1, 1, dtype=torch.float64), beta, ab,
                       torch.zeros(1, 1, 8, dtype=torch.float64),
                       torch.tensor(1.0), 'ilr16', (), samples=1, seed=7)


# ---------------------------------------------------------------- failure_records
def test_failure_records_stable_ids_no_modification():
    methods = {'example': {'status': 'OK', 'eligible_row_positions': [0, 1],
                           'numerical_diagnostics': {'scale_guard_row_positions': [1],
                                                     'guard_threshold': 1000.,
                                                     'guard_basis': 'visible_sum_or_one'}}}
    records = failure_records(methods, [14, 103], 'test')
    assert records[0]['row_ids'] == [103]
    assert records[0]['predictions_modified'] is False
    assert records[0]['code'] == 'SCALE_GUARD_EXCEEDED'


# ---------------------------------------------------------------- frozen schedule sanity
def test_terminal_alpha_bar_matches_directive():
    beta, ab = schedule()
    assert abs(float(ab[-1]) - 0.0147804172) < 1e-6
    assert abs(float(beta.max()) - 0.5) < 1e-12
    amplification = float(1.0 / ab[-1].sqrt())
    assert abs(amplification - 8.225392918039187) < 1e-6
