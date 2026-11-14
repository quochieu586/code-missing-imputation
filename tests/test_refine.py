"""Deterministic stopping tests at the existing train/predict boundary."""
import numpy as np
import pytest
from longitudinal_impute import main_model as main
from longitudinal_impute.data import Dataset, check_completed


def views():
    raw = np.array([[20., 30.], [10., 20.], [30., 10.], [20., np.nan]])
    data = Dataset(raw, np.full(4, 100.), ('a', 'b'), ('s1', 's1', 's1', 's2'),
                   [], [], [], times=('3', '1', '2', '5'))
    q = np.zeros_like(raw, bool)
    q[0, 1] = True
    return main.prepare_views(data, train_mask=q)


def test_unchanged_completion_stops_after_one_round(monkeypatch, capsys):
    inference, training, q, labels = views()
    calls = []
    def train(network, view, mask, truth, **kwargs):
        calls.append((network, view, mask.copy()))
        return {'loss': .2, 'steps': 1}
    monkeypatch.setattr(main, 'train_one_round', train)
    monkeypatch.setattr(main, 'predict_counts', lambda network, view, **kwargs: view.counts.copy())
    net = object()
    counts, info = main.refine(net, inference, training, q, labels, outer_iters=5)
    assert len(calls) == info['iterations'] == 1
    assert calls[0][0] is net and calls[0][1] is training
    assert info['converged'] is True and info['termination_reason'] == 'tolerance'
    assert info['history'][0]['max_change_proportion'] == 0.
    assert info['history'][0]['loss'] == .2
    assert info['history'][0]['converged'] is True
    assert 'converged=True' in capsys.readouterr().out
    np.testing.assert_array_equal(counts, inference.counts)


def test_limit_uses_proportion_change_updates_both_views_and_warm_starts(monkeypatch):
    inference, training, q, labels = views()
    seen_networks, seen_training_counts = [], []
    initial_training = training.counts.copy()
    initial_inference = inference.counts.copy()
    def train(network, view, mask, truth, **kwargs):
        seen_networks.append(network)
        seen_training_counts.append(view.counts.copy())
        return {'loss': .1, 'steps': 1}
    def predict(network, view, **kwargs):
        counts = view.counts.copy()
        counts[~view.observed_mask[:, :-1]] += 1.
        return counts
    monkeypatch.setattr(main, 'train_one_round', train)
    monkeypatch.setattr(main, 'predict_counts', predict)
    net = object()
    counts, info = main.refine(net, inference, training, q, labels, outer_iters=3, tol=.001)
    assert info['iterations'] == 3 and info['converged'] is False
    assert info['termination_reason'] == 'max_iter'
    assert all(n is net for n in seen_networks)
    np.testing.assert_array_equal(seen_training_counts[0], initial_training)
    assert seen_training_counts[1][q][0] == initial_training[q][0] + 1.
    for row in info['history']:
        assert row['max_change_proportion'] == pytest.approx(.01)
        assert row['seconds'] >= 0
        assert row['converged'] is False
    np.testing.assert_array_equal(counts[~inference.observed_mask[:, :-1]],
                                  initial_inference[~inference.observed_mask[:, :-1]] + 3.)
    check_completed(inference.data, counts, np.zeros_like(q))


def test_invalid_prediction_is_rejected_at_loop_boundary(monkeypatch):
    inference, training, q, labels = views()
    monkeypatch.setattr(main, 'train_one_round', lambda *a, **k: {'loss': .1})
    monkeypatch.setattr(main, 'predict_counts', lambda network, view, **k: np.full_like(view.counts, np.nan))
    with pytest.raises(ValueError, match='Output'):
        main.refine(object(), inference, training, q, labels)
