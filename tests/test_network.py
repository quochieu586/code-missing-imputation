"""Behavior of the optional CPU network, not its internal class layout."""
import pytest

torch = pytest.importorskip('torch')
from longitudinal_impute.network import CompositionNet


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(k=3):
    torch.manual_seed(17)
    z = torch.randn(2, 3, k)
    observed = torch.ones_like(z, dtype=torch.bool)
    observed[0, 1, 0] = False
    times = torch.tensor([[0., 2., 9.], [4., 0., 0.]])
    valid = torch.tensor([[True, True, True], [True, False, False]])
    return z, observed, times, valid


@pytest.mark.parametrize('k', [3, 6])
def test_forward_outputs_compositions_and_backpropagates(k):
    z, observed, times, valid = inputs(k)
    z.requires_grad_()
    net = CompositionNet(k, channels=8, heads=2)
    p = net(z, observed, times, valid)
    assert p.shape == z.shape and torch.isfinite(p).all()
    torch.testing.assert_close(p[valid].sum(-1), torch.ones(valid.sum()))
    assert (p[valid] > 0).all() and (p[~valid] == 0).all()
    (p * torch.arange(k)).sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad[valid].abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def test_padding_values_cannot_change_real_predictions():
    z, observed, times, valid = inputs()
    net = CompositionNet(3, channels=8, heads=2).eval()
    with torch.no_grad():
        expected = net(z, observed, times, valid)
        z[~valid] = float('nan')
        observed[~valid] = False
        times[~valid] = float('nan')
        actual = net(z, observed, times, valid)
    torch.testing.assert_close(actual, expected)


def test_entities_are_independent_and_padding_length_is_irrelevant():
    z, observed, times, valid = inputs()
    net = CompositionNet(3, channels=8, heads=2).eval()
    with torch.no_grad():
        batch = net(z, observed, times, valid)
        alone = net(z[1:, :1], observed[1:, :1], times[1:, :1], valid[1:, :1])
        torch.testing.assert_close(batch[1, :1], alone[0], rtol=1e-5, atol=1e-6)
        z[0] *= 100.
        torch.testing.assert_close(net(z, observed, times, valid)[1], batch[1])


def test_empty_entity_and_nonfinite_real_features_are_rejected():
    z, observed, times, valid = inputs()
    net = CompositionNet(3, channels=8, heads=2)
    valid[1] = False
    with pytest.raises(ValueError, match='valid'):
        net(z, observed, times, valid)
    valid[1, 0] = True
    z[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        net(z, observed, times, valid)


def test_distant_timepoint_actually_influences_prediction():
    z, observed, times, valid = inputs()
    z.requires_grad_()
    net = CompositionNet(3, channels=8, heads=2)
    p = net(z, observed, times, valid)
    p[0, 0, 0].backward()
    # No local projection, head or feature attention can create this cross-time path.
    assert z.grad[0, 2, 0].abs() > 1e-8


def test_feature_information_flows_beyond_softmax_normalization():
    z, observed, times, valid = inputs()
    z.requires_grad_()
    net = CompositionNet(3, channels=8, heads=2)
    p = net(z, observed, times, valid)
    # log(p0/p2) cancels the final softmax denominator. Its dependence on z1
    # therefore requires actual cross-feature information, not mere row closure.
    (p[0, 0, 0].log() - p[0, 0, 2].log()).backward()
    assert z.grad[0, 0, 1].abs() > 1e-8
