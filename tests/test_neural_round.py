"""One neural round: physical allocation, real gradients and no evaluation labels."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from longitudinal_impute.main_model import allocate_residual, masked_composition_loss
from longitudinal_impute.data import Dataset, restore_counts


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_allocation_matches_numpy_and_loss_has_gradient():
    raw = torch.tensor([[20., float('nan')], [100., float('nan')], [20., 30.]], dtype=torch.float64)
    totals = torch.full((3,), 100., dtype=torch.float64)
    proposals = torch.tensor([[.2, .3, .5], [.2, .3, .5], [.2, .3, .5]], dtype=torch.float64, requires_grad=True)
    actual = allocate_residual(proposals, raw, totals)
    data = Dataset(raw.numpy(), totals.numpy(), ('a', 'b'), ('s1', 's2', 's3'), [], [], [])
    expected = restore_counts(data, proposals.detach().numpy(), np.zeros((3, 2), bool))
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-12)
    torch.testing.assert_close(actual, torch.tensor([[20., 30.], [100., 0.], [20., 30.]], dtype=torch.float64))
    q = torch.tensor([[False, True], [False, False], [False, False]])
    labels = torch.full_like(raw, float('nan'))
    labels[q] = .1
    loss = masked_composition_loss(proposals, raw, totals, q, labels)
    assert loss.item() == pytest.approx(.04)
    loss.backward()
    assert torch.isfinite(proposals.grad).all()
    assert proposals.grad[0, 1].abs() > 0 and proposals.grad[0, 2].abs() > 0
    assert (proposals.grad[1:] == 0).all()


def test_allocation_is_stable_for_large_weights_and_does_not_mutate_raw():
    raw = torch.tensor([[20., float('nan')]], dtype=torch.float64)
    before = raw.clone()
    result = allocate_residual(torch.tensor([[1., 3e300, 5e300]], dtype=torch.float64), raw,
                               torch.tensor([100.], dtype=torch.float64))
    assert result[0, 1].item() == pytest.approx(30.)
    torch.testing.assert_close(raw, before, equal_nan=True)


@pytest.mark.parametrize('kind', ['empty_q', 'visible_q', 'nan_label', 'negative_weight', 'no_free_mass'])
def test_bad_training_contract_is_rejected(kind):
    raw = torch.tensor([[20., float('nan')]], dtype=torch.float64)
    totals = torch.tensor([100.], dtype=torch.float64)
    p = torch.tensor([[.2, .3, .5]], requires_grad=True)
    q = torch.tensor([[False, True]])
    labels = torch.tensor([[float('nan'), .3]], dtype=torch.float64)
    if kind == 'empty_q':
        q[:] = False
    elif kind == 'visible_q':
        q[0, 0] = True
    elif kind == 'nan_label':
        labels[0, 1] = float('nan')
    elif kind == 'negative_weight':
        p = torch.tensor([[.2, -.3, .5]])
    else:
        p = torch.tensor([[1., 0., 0.]])
    with pytest.raises(ValueError):
        masked_composition_loss(p, raw, totals, q, labels)


def round_data():
    from longitudinal_impute.data import Dataset
    counts = np.array([[20., 30.], [10., 20.], [30., 10.], [20., np.nan]])
    return Dataset(counts, np.full(4, 100.), ('a', 'b'), ('s1', 's1', 's1', 's2'),
                   [], [], [1, 2, 3, 4], times=('3', '1', '2', '5'))


def test_train_round_changes_weights_and_predicts_valid_counts():
    from longitudinal_impute.main_model import prepare_views, train_one_round, predict_counts
    from longitudinal_impute.network import CompositionNet
    from longitudinal_impute.data import check_completed
    data = round_data()
    q = np.zeros_like(data.counts, bool)
    q[0, 1] = True
    inference, training, q, labels = prepare_views(data, train_mask=q)
    before_counts = training.counts.copy()
    torch.manual_seed(18)
    net = CompositionNet(3, channels=8, heads=2)
    before = [p.detach().clone() for p in net.parameters()]
    info = train_one_round(net, training, q, labels, epochs=2, batch_size=1)
    assert info['epochs'] == 2 and info['steps'] == 2 and info['train_cells'] == 1
    assert np.isfinite(info['loss']) and info['loss'] >= 0
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))
    np.testing.assert_array_equal(training.counts, before_counts)
    for view in (training, inference):
        counts = predict_counts(net, view, batch_size=1)
        check_completed(view.data, counts, np.zeros_like(data.counts, bool))
        assert counts.dtype == np.float64
        np.testing.assert_array_equal(view.data.counts[np.isfinite(view.data.counts)],
                                      counts[np.isfinite(view.data.counts)])


def test_evaluation_truth_does_not_change_training_or_predictions():
    from longitudinal_impute.main_model import prepare_views, train_one_round, predict_counts
    from longitudinal_impute.network import CompositionNet
    from dataclasses import replace
    data = round_data()
    h = np.zeros_like(data.counts, bool)
    h[1, 0] = True
    q = np.zeros_like(h)
    q[0, 1] = True
    def one_round(source):
        inference, training, train_mask, labels = prepare_views(source, h, train_mask=q)
        torch.manual_seed(20)
        net = CompositionNet(3, channels=8, heads=2)
        info = train_one_round(net, training, train_mask, labels, epochs=1)
        return predict_counts(net, inference), info, [p.detach().clone() for p in net.parameters()]
    expected, expected_info, expected_weights = one_round(data)
    changed = data.counts.copy()
    changed[h] = 65.
    actual, info, weights = one_round(replace(data, counts=changed))
    np.testing.assert_array_equal(actual, expected)
    assert info == expected_info
    for a, b in zip(weights, expected_weights):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_empty_train_mask_is_not_a_successful_training_round():
    from longitudinal_impute.main_model import prepare_views, train_one_round
    from longitudinal_impute.network import CompositionNet
    data = round_data()
    q = np.zeros_like(data.counts, bool)
    _, training, q, labels = prepare_views(data, train_mask=q)
    net = CompositionNet(3, channels=8, heads=2)
    before = [p.detach().clone() for p in net.parameters()]
    with pytest.raises(ValueError, match='nonempty'):
        train_one_round(net, training, q, labels)
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters()))


@pytest.mark.parametrize('case', ['covid', 'microbiome'])
def test_one_round_runs_on_both_dataset_schemas(tmp_path, case):
    from pathlib import Path
    from test_run import COVID_SCHEMA, MICROBIOME_SCHEMA
    from longitudinal_impute.data import load_csv, check_completed
    from longitudinal_impute.main_model import prepare_views, train_one_round, predict_counts
    from longitudinal_impute.network import CompositionNet
    schema = tmp_path / 'schema.toml'
    schema.write_text(COVID_SCHEMA if case == 'covid' else MICROBIOME_SCHEMA)
    data = load_csv(Path(__file__).parent / 'fixtures' / f'{case}_small.csv', schema)
    # One predetermined observed cell; never redraw to avoid a failure.
    q = np.zeros_like(data.counts, bool)
    q[1, 1] = True
    inference, training, q, labels = prepare_views(data, train_mask=q)
    torch.manual_seed(21)
    net = CompositionNet(len(data.targets) + 1, channels=8, heads=2)
    info = train_one_round(net, training, q, labels, epochs=1, batch_size=2)
    assert info['steps'] > 0 and np.isfinite(info['loss'])
    pred = predict_counts(net, inference, batch_size=2)
    check_completed(data, pred, np.zeros_like(q))
