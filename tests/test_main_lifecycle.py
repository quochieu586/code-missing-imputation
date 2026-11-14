"""Orchestrator configuration and case lifecycle, not stage persistence."""
import builtins
from dataclasses import replace
import numpy as np
import pytest
from longitudinal_impute.main_model import MainModel
from longitudinal_impute.data import Dataset, check_completed


def data_for_main():
    return Dataset(np.array([[20., 30.], [10., 20.], [30., 10.], [20., np.nan]]),
                   np.full(4, 100.), ('a', 'b'), ('s1', 's1', 's1', 's2'), [], [], [],
                   times=('3', '1', '2', '5'))


@pytest.mark.parametrize('config', [
    {'outer_iters': 0}, {'epochs_per_iter': True}, {'batch_size': -1}, {'init_k': 0},
    {'channels': 7}, {'channels': 8, 'heads': 3}, {'seed': -1}, {'seed': True},
    {'tol': float('nan')}, {'tol': -1.}, {'learning_rate': 0.},
    {'train_mask_rate': 1.}, {'device': 'cuda'},
])
def test_bad_config_is_rejected_without_fit(config):
    with pytest.raises(ValueError):
        MainModel(**config)


def test_complete_input_skips_torch_and_training(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            raise AssertionError('No torch import on no-missing case or constructor')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    model = MainModel()
    data = data_for_main()
    data.counts[-1, 1] = 30.
    completed = model.impute(data)
    np.testing.assert_array_equal(completed, data.counts)
    assert model.info_['trained'] is False and model.info_['iterations'] == 0
    assert model.info_['termination_reason'] == 'no_missing'
    assert model.network_ is None


@pytest.mark.parametrize('cause', ['missing', 'dll'])
def test_missing_torch_reports_actionable_error(monkeypatch, cause):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'torch':
            if cause == 'dll':
                raise OSError(126, 'controlled Windows torch DLL load failure')
            raise ModuleNotFoundError('torch deliberately unavailable', name='torch')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    model = MainModel()
    with pytest.raises(ValueError, match='extra main'):
        model.impute(data_for_main())
    assert model.network_ is None and model.info_['termination_reason'] == 'failed'


def test_empty_q_is_not_silently_redrawn_or_reported_as_trained():
    model = MainModel(train_mask_rate=0.)
    with pytest.raises(ValueError, match='nonempty'):
        model.impute(data_for_main())


def test_one_outer_round_matches_existing_api_and_resets_each_call():
    torch = pytest.importorskip('torch')
    from longitudinal_impute.main_model import prepare_views, train_one_round, predict_counts
    from longitudinal_impute.network import CompositionNet
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = data_for_main()
        inference, training, q, labels = prepare_views(data, seed=0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            expected_network = CompositionNet(3, channels=8, heads=2)
        train_one_round(expected_network, training, q, labels, epochs=1, batch_size=2)
        training.counts = predict_counts(expected_network, training, batch_size=2)
        expected = predict_counts(expected_network, inference, batch_size=2)
        model = MainModel(channels=8, heads=2, outer_iters=1, epochs_per_iter=1, batch_size=2, tol=0.)
        before_rng = torch.get_rng_state().clone()
        actual = model.impute(data)
        np.testing.assert_array_equal(actual, expected)
        assert torch.equal(torch.get_rng_state(), before_rng)
        assert model.info_['trained'] is True and model.info_['iterations'] == 1
        first_network = model.network_
        again = model.impute(data)
        assert model.network_ is not first_network
        np.testing.assert_array_equal(actual, again)
        check_completed(data, actual, np.zeros_like(q))
        with pytest.raises(ValueError, match='time'):
            model.impute(replace(data, times=None))
        assert model.network_ is None and model.info_['trained'] is False
    finally:
        torch.set_num_threads(previous)


def test_real_multi_rounds_keep_observations_and_isolate_hidden_truth():
    torch = pytest.importorskip('torch')
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = data_for_main()
        hidden = np.zeros_like(data.counts, bool)
        hidden[0, 1] = True
        model = MainModel(channels=8, heads=2, outer_iters=3, epochs_per_iter=1, tol=0.)
        first = model.impute(data, hidden)
        weights = [p.detach().clone() for p in model.network_.parameters()]
        first_info = model.info_
        changed = data.counts.copy()
        changed[hidden] = 70.
        second = model.impute(replace(data, counts=changed), hidden)
        np.testing.assert_array_equal(first, second)
        assert 1 <= first_info['iterations'] <= 3
        assert len(first_info['history']) == first_info['iterations']
        assert model.info_['converged'] == first_info['converged']
        for a, b in zip(weights, model.network_.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        check_completed(data, first, hidden)
    finally:
        torch.set_num_threads(previous)
