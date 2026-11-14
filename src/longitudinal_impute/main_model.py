"""Direct-composition main model, with explicit preparation and bounded refinement.

An InitialView owns only visible raw data and its temporary completion. Hidden
truth never survives in its numeric counts or CSV target fields. Torch imports
stay inside the neural path; configuration and complete-input fast paths use NumPy.
"""
from dataclasses import dataclass, replace
from datetime import date
from time import perf_counter

import numpy as np

from .baselines.tsagris import TsagrisKNN
from .data import Dataset, model_input, restore_counts, check_completed, make_mask
from .errors import BaselineError, BaselineInputError, InsufficientDataError, NumericalError


@dataclass
class InitialView:
    data: Dataset                  # visible raw counts; unknowns remain NaN
    counts: np.ndarray             # temporary completion, N x D
    observed_mask: np.ndarray      # N x (D+Other), before completion
    delta: float                   # visible min-positive / 2, frozen for this view


def _time_values(data):
    """Numeric visits or ISO calendar dates, with a shared origin across entities."""
    n = len(data.counts)
    if not n or data.times is None or len(data.times) != n or len(data.entities) != n:
        raise BaselineInputError('main time: one time and entity per nonempty row is required')
    try:
        times = np.array([float(t) for t in data.times])
    except (TypeError, ValueError):
        try:
            times = np.array([date.fromisoformat(t).toordinal() for t in data.times], dtype=float)
        except (TypeError, ValueError):
            raise BaselineInputError('main time: use numeric visits or ISO dates consistently') from None
    if not np.isfinite(times).all():
        raise BaselineInputError('main time: values must be finite')
    keys = set()
    for i, (entity, t) in enumerate(zip(data.entities, times)):
        if not entity or (entity, t) in keys:
            raise BaselineInputError('main time: empty entity or duplicate parsed time',
                                     context={'row': i, 'entity': entity})
        keys.add((entity, t))
    times -= times.min()
    if not np.isfinite(times).all():
        raise NumericalError('main time: relative times overflowed')
    return times


def pack_sequences(data, values):
    """Return (packed, times, valid_steps, row_indices); -1 marks padding.

    All entities are packed here; a later training loop may slice this first
    axis into minibatches. No data is resampled or interpolated.
    """
    times = _time_values(data)
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[0] != len(times):
        raise BaselineInputError('pack: expected one value row per dataset row')
    entities = np.asarray(data.entities)
    groups = []
    for entity in sorted(set(data.entities)):
        rows = np.flatnonzero(entities == entity)
        groups.append(rows[np.argsort(times[rows], kind='stable')])
    shape = (len(groups), max(map(len, groups)))
    packed = np.zeros((*shape, *values.shape[1:]), dtype=values.dtype)
    packed_times = np.zeros(shape)
    row_indices = np.full(shape, -1, dtype=int)
    for b, rows in enumerate(groups):
        packed[b, :len(rows)] = values[rows]
        packed_times[b, :len(rows)] = times[rows]
        row_indices[b, :len(rows)] = rows
    return packed, packed_times, row_indices >= 0, row_indices


def unpack_sequences(packed, row_indices):
    """Restore exact CSV row order, ignoring padding; reject broken mappings."""
    packed, row_indices = np.asarray(packed), np.asarray(row_indices)
    if (row_indices.ndim != 2 or packed.ndim < 2 or packed.shape[:2] != row_indices.shape
            or not np.issubdtype(row_indices.dtype, np.integer) or (row_indices < -1).any()):
        raise BaselineInputError('unpack: invalid row mapping or shape')
    valid = row_indices >= 0
    rows = row_indices[valid]
    if not len(rows) or not np.array_equal(np.sort(rows), np.arange(len(rows))):
        raise BaselineInputError('unpack: row mapping must cover each CSV row exactly once')
    values = np.empty((len(rows), *packed.shape[2:]), dtype=packed.dtype)
    values[rows] = packed[valid]
    return values


def initialize(data, hidden=None, *, init_k=3):
    """Initialize one isolated visible view with the existing Tsagris/R2 route."""
    _time_values(data)
    if hidden is None:
        hidden = np.zeros(data.counts.shape, dtype=bool)
    # Reuse mask validation and visible-only Other derivation before any fit.
    values = model_input(data, hidden)
    observed = np.isfinite(values)
    positives = values[observed & (values > 0)]
    if not len(positives):
        raise InsufficientDataError('initialize: no visible positive proportion for CLR')
    delta = float(positives.min() / 2)
    if delta <= 0:
        raise NumericalError('initialize: CLR delta is not representable')
    visible = _visible_data(data, observed)
    raw = visible.counts
    model = TsagrisKNN(k=init_k)
    proposals = model.fit(values).predict(values)
    nothing = np.zeros(raw.shape, dtype=bool)
    completed = restore_counts(visible, proposals * visible.totals[:, None], nothing)
    check_completed(visible, completed, nothing)
    return InitialView(visible, completed, observed, delta)


def _visible_data(data, observed):
    """Scrub numeric and CSV target truth; shared by initializer and saved-state loader."""
    raw = np.where(observed[:, :-1], data.counts, np.nan)
    rows = [dict(row) for row in data.rows]
    for i, row in enumerate(rows):
        for j, target in enumerate(data.targets):
            if not observed[i, j]:
                row[target] = ''
    return replace(data, counts=raw, totals=data.totals.copy(), rows=rows,
                   fields=list(data.fields), lines=list(data.lines))


def prepare_views(data, hidden=None, *, train_mask=None, mask_rate=.1, seed=0, init_k=3):
    """Return inference view, training view, Q and proportion labels (NaN outside Q).

    Q is removed before the training initializer. Only labels, never features,
    carry its values. Evaluation truth is absent from both views.
    """
    inference = initialize(data, hidden, init_k=init_k)
    if train_mask is None:
        train_mask, _ = make_mask(inference.data, 'random_cell', mask_rate, seed)
    train_mask = np.asarray(train_mask)
    if (train_mask.shape != data.counts.shape or train_mask.dtype != bool
            or (train_mask & ~np.isfinite(inference.data.counts)).any()):
        raise BaselineInputError('train mask must be a boolean subset of visible targets')
    train_mask = train_mask.copy()
    labels = np.full(data.counts.shape, np.nan)
    labels[train_mask] = (inference.data.counts / inference.data.totals[:, None])[train_mask]
    training = initialize(inference.data, train_mask, init_k=init_k)
    return inference, training, train_mask, labels


def clr_features(view, counts=None):
    """CLR on a disposable feature copy; raw observed zeros remain untouched."""
    counts = view.counts if counts is None else np.asarray(counts)
    check_completed(view.data, counts, np.zeros(view.data.counts.shape, dtype=bool))
    other = view.data.totals - counts.sum(axis=1)
    # Validator already bounded negative roundoff by the shared budget tolerance.
    parts = np.column_stack((counts, np.maximum(other, 0.))) / view.data.totals[:, None]
    if not np.isfinite(view.delta) or view.delta <= 0:
        raise NumericalError('CLR: delta must be finite and positive')
    parts[parts == 0] = view.delta
    logs = np.log(parts)
    z = logs - logs.mean(axis=1, keepdims=True)
    if not np.isfinite(z).all():
        raise NumericalError('CLR: non-finite features')
    return z


def allocate_residual(proposals, raw_counts, totals):
    """Differentiable R2 allocation; torch tensors, counts remain float64."""
    import torch
    if (raw_counts.ndim != 2 or not len(raw_counts) or raw_counts.shape[1] < 2
            or proposals.shape != (len(raw_counts), raw_counts.shape[1] + 1)
            or totals.shape != (len(raw_counts),)):
        raise BaselineInputError('allocation: expected N x D counts, N x (D+Other) proposals, N totals')
    if (not torch.isfinite(proposals).all() or (proposals < 0).any()
            or torch.isinf(raw_counts).any() or (raw_counts < 0).any()
            or not torch.isfinite(totals).all() or (totals <= 0).any()):
        raise BaselineInputError('allocation: nonnegative finite weights/counts and positive totals required')
    raw, totals = raw_counts.to(torch.float64), totals.to(torch.float64)
    weights = proposals.to(torch.float64)
    visible = torch.isfinite(raw)
    anchors = torch.where(visible, raw, 0.)
    residual = totals - anchors.sum(dim=-1)
    if (residual < 0).any():
        raise BaselineInputError('allocation: visible counts exceed total')
    free = torch.cat((~visible, torch.ones_like(visible[:, :1])), dim=-1)
    weights = torch.where(free, weights, 0.)
    scale = weights.max(dim=-1, keepdim=True).values
    active = (~visible).any(dim=-1) & (residual > 0)
    if ((scale.squeeze(-1) == 0) & active).any():
        raise NumericalError('allocation: no proposed mass for missing targets and Other')
    scaled = weights / torch.where(scale > 0, scale, 1.)
    total_weight = scaled.sum(dim=-1, keepdim=True)
    allocation = scaled / torch.where(total_weight > 0, total_weight, 1.) * residual[:, None]
    return torch.where(visible, anchors, allocation[:, :-1])


def masked_composition_loss(proposals, raw_counts, totals, train_mask, labels):
    """Proportion MSE on Q only; labels outside Q are never read."""
    import torch
    if (train_mask.shape != raw_counts.shape or train_mask.dtype != torch.bool
            or labels.shape != raw_counts.shape or not train_mask.any()
            or (train_mask & torch.isfinite(raw_counts)).any()):
        raise BaselineInputError('loss: nonempty Q must be missing in the training input')
    truth = labels[train_mask]
    if not torch.isfinite(truth).all() or (truth < 0).any() or (truth > 1).any():
        raise BaselineInputError('loss: Q needs finite proportion labels in [0, 1]')
    counts = allocate_residual(proposals, raw_counts, totals)
    predicted = (counts / totals[:, None])[train_mask]
    return (predicted - truth).square().mean()


def train_one_round(network, view, train_mask, labels, *, epochs=10, batch_size=8, learning_rate=1e-3):
    """Fit on one fixed safe training view, without updating its imputed counts."""
    import torch
    for name, value in (('epochs', epochs), ('batch_size', batch_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise BaselineInputError(f'train: {name} must be a positive integer')
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise BaselineInputError('train: learning_rate must be finite and positive')
    train_mask, labels = np.asarray(train_mask), np.asarray(labels)
    if (train_mask.shape != view.data.counts.shape or train_mask.dtype != bool
            or labels.shape != train_mask.shape or not train_mask.any()
            or (train_mask & np.isfinite(view.data.counts)).any()):
        raise BaselineInputError('train: nonempty Q must be missing in the training view')
    truth = labels[train_mask]
    if not np.isfinite(truth).all() or (truth < 0).any() or (truth > 1).any():
        raise BaselineInputError('train: Q needs finite proportion labels in [0, 1]')
    tensors, indices = _network_inputs(network, view)
    features, observed, times, valid = tensors
    device = features.device
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    network.train()
    steps, mean_loss = 0, None
    for _ in range(epochs):
        loss_sum = 0.
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            rows = indices[start:end][indices[start:end] >= 0]
            q = train_mask[rows]
            if not q.any():
                continue  # No labels in these entities, not a successful training step.
            optimizer.zero_grad(set_to_none=True)
            proposals = network(features[start:end], observed[start:end], times[start:end], valid[start:end])
            proposals = proposals[valid[start:end]]
            loss = masked_composition_loss(
                proposals, torch.as_tensor(view.data.counts[rows], dtype=torch.float64, device=device),
                torch.as_tensor(view.data.totals[rows], dtype=torch.float64, device=device),
                torch.as_tensor(q, device=device),
                torch.as_tensor(labels[rows], dtype=torch.float64, device=device))
            if not torch.isfinite(loss):
                raise NumericalError('train: non-finite loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
                raise NumericalError('train: non-finite gradient')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in network.parameters()):
                raise NumericalError('train: non-finite weights after optimizer step')
            loss_sum += float(loss.detach()) * int(q.sum())
            steps += 1
        mean_loss = loss_sum / int(train_mask.sum())
    return {'loss': mean_loss, 'epochs': epochs, 'steps': steps, 'train_cells': int(train_mask.sum())}


def predict_counts(network, view, *, batch_size=8):
    """Network proposals → authoritative NumPy restoration → checked raw counts."""
    import torch
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise BaselineInputError('predict: batch_size must be a positive integer')
    tensors, indices = _network_inputs(network, view)
    features, observed, times, valid = tensors
    packed = np.zeros(features.shape, dtype=np.float64)
    network.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            p = network(features[start:end], observed[start:end], times[start:end], valid[start:end])
            packed[start:end] = p.cpu().numpy()
    proposals = unpack_sequences(packed, indices)
    nothing = np.zeros(view.data.counts.shape, dtype=bool)
    counts = restore_counts(view.data, proposals * view.data.totals[:, None], nothing)
    check_completed(view.data, counts, nothing)
    return counts


def _network_inputs(network, view):
    """Pack the same row map for features and masks; never pass labels to forward."""
    import torch
    features, times, valid, indices = pack_sequences(view.data, clr_features(view))
    observed = pack_sequences(view.data, view.observed_mask)[0]
    parameter = next(network.parameters())
    tensors = (torch.as_tensor(features, dtype=parameter.dtype, device=parameter.device),
               torch.as_tensor(observed, device=parameter.device),
               torch.as_tensor(times, dtype=parameter.dtype, device=parameter.device),
               torch.as_tensor(valid, device=parameter.device))
    return tensors, indices


def refine(network, inference, training, train_mask, labels, *, outer_iters=5, tol=1e-4,
           epochs_per_iter=10, batch_size=8, learning_rate=1e-3):
    """Bounded warm-start refinement; train and inference views stay separate."""
    if not isinstance(outer_iters, int) or isinstance(outer_iters, bool) or outer_iters < 1:
        raise BaselineInputError('refine: outer_iters must be a positive integer')
    if not np.isfinite(tol) or tol < 0:
        raise BaselineInputError('refine: tol must be finite and nonnegative')
    missing = ~inference.observed_mask[:, :-1]
    history, converged = [], False
    if not missing.any():
        return inference.counts.copy(), {'iterations': 0, 'converged': None,
                                        'termination_reason': 'no_missing', 'history': history}
    for iteration in range(1, outer_iters + 1):
        start = perf_counter()
        before = inference.counts
        stage = 'train'
        try:
            fit = train_one_round(network, training, train_mask, labels, epochs=epochs_per_iter,
                                  batch_size=batch_size, learning_rate=learning_rate)
            stage = 'training_prediction'
            training.counts = predict_counts(network, training, batch_size=batch_size)
            check_completed(training.data, training.counts, np.zeros(training.counts.shape, bool))
            stage = 'inference_prediction'
            inference.counts = predict_counts(network, inference, batch_size=batch_size)
            check_completed(inference.data, inference.counts, np.zeros(inference.counts.shape, bool))
        except BaselineError as exc:
            raise exc.add_context(stage=stage, iteration=iteration)
        except (ValueError, RuntimeError) as exc:
            raise NumericalError(f'main {stage}: {exc}',
                                 context={'stage': stage, 'iteration': iteration}) from exc
        change = float((np.abs(inference.counts - before) / inference.data.totals[:, None])[missing].max())
        converged = change <= tol
        history.append({'iteration': iteration, 'loss': fit['loss'],
                        'max_change_proportion': change, 'converged': converged,
                        'seconds': perf_counter() - start})
        print(f'[main] iteration={iteration} loss={fit["loss"]:.6g} '
              f'max_change_proportion={change:.6g} converged={converged} '
              f'seconds={history[-1]["seconds"]:.3g}')
        if converged:
            break
    return inference.counts.copy(), {'iterations': len(history), 'converged': converged,
                                     'termination_reason': 'tolerance' if converged else 'max_iter',
                                     'history': history}


class MainModel:
    """Small counts-facing orchestrator; configuration must not import torch."""
    def __init__(self, *, init_k=3, channels=32, heads=4, outer_iters=5, epochs_per_iter=10,
                 batch_size=8, learning_rate=1e-3, tol=1e-4, train_mask_rate=.1, seed=0, device='cpu'):
        integers = {'init_k': init_k, 'channels': channels, 'heads': heads,
                    'outer_iters': outer_iters, 'epochs_per_iter': epochs_per_iter, 'batch_size': batch_size}
        for name, value in integers.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise BaselineInputError(f'main: {name} must be a positive integer')
        if channels < 2 or channels % 2 or channels % heads:
            raise BaselineInputError('main: channels must be even and divisible by heads')
        for name, value in (('learning_rate', learning_rate), ('tol', tol), ('train_mask_rate', train_mask_rate)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
                raise BaselineInputError(f'main: {name} must be a finite number')
        if learning_rate <= 0 or tol < 0 or not 0 <= train_mask_rate < 1:
            raise BaselineInputError('main: learning_rate > 0, tol >= 0, train_mask_rate in [0, 1) required')
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise BaselineInputError('main: seed must be an integer in [0, 2**63)')
        if device != 'cpu':
            raise BaselineInputError('main: this version supports device="cpu" only')
        self.init_k, self.channels, self.heads = init_k, channels, heads
        self.outer_iters, self.epochs_per_iter, self.batch_size = outer_iters, epochs_per_iter, batch_size
        self.learning_rate, self.tol, self.train_mask_rate = learning_rate, tol, train_mask_rate
        self.seed, self.device = seed, device
        self.network_, self.info_ = None, {}

    def prepare(self, data, hidden=None):
        """Torch-free stage boundary, also used by the full path."""
        _time_values(data)
        if hidden is None:
            hidden = np.zeros(data.counts.shape, bool)
        values = model_input(data, hidden)
        if np.isfinite(values[:, :-1]).all():
            return None
        initial = prepare_views(data, hidden, seed=self.seed, mask_rate=self.train_mask_rate, init_k=self.init_k)
        if not initial[2].any():
            raise InsufficientDataError('main: nonempty training Q required; no mask redraw or fallback')
        return initial

    def impute(self, data, hidden=None, *, initial=None):
        # Reset on EVERY call, including a failed refit: never expose stale weights/results.
        self.network_ = None
        self.info_ = {'method': 'iterative_direct_composition', 'budget_policy': 'R2',
                      'seed': self.seed, 'device': self.device, 'trained': False,
                      'iterations': 0, 'converged': None, 'history': []}
        stage = 'input'
        try:
            _time_values(data)
            if hidden is None:
                hidden = np.zeros(data.counts.shape, bool)
            values = model_input(data, hidden)  # validates H without exposing its truth
            if np.isfinite(values[:, :-1]).all():
                completed = data.counts.copy()
                check_completed(data, completed, hidden)
                self.info_['termination_reason'] = 'no_missing'
                return completed
            stage = 'initialize'
            inference, training, q, labels = self.prepare(data, hidden) if initial is None else initial
            if not q.any():
                raise InsufficientDataError('main: nonempty training Q required; no mask redraw or fallback')
            stage = 'torch_setup'
            try:
                import torch
                from .network import CompositionNet
            except (ImportError, OSError) as exc:
                raise BaselineInputError('main requires extra main: uv sync --frozen --extra main --extra test; '
                                         f'original import error: {exc}') from exc
            # Isolate construction RNG; training has no dropout or random batch shuffle.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                network = CompositionNet(len(data.targets) + 1, self.channels, self.heads)
            stage = 'refine'
            completed, result = refine(network, inference, training, q, labels,
                                       outer_iters=self.outer_iters, tol=self.tol,
                                       epochs_per_iter=self.epochs_per_iter, batch_size=self.batch_size,
                                       learning_rate=self.learning_rate)
            check_completed(data, completed, hidden)
            self.network_ = network
            self.info_.update(result, trained=True, train_cells=int(q.sum()),
                              delta_inference=inference.delta, delta_training=training.delta)
            return completed
        except BaselineError as exc:
            self.info_.update(termination_reason='failed', error=str(exc))
            raise exc.add_context(stage=exc.context.get('stage', stage))
        except (ValueError, RuntimeError) as exc:
            self.info_.update(termination_reason='failed', error=str(exc))
            raise NumericalError(f'main {stage}: {exc}', context={'stage': stage}) from exc


def initial_arrays(initial):
    """Numeric-only NPZ payload; no dataset object, CSV truth, torch or pickle."""
    if initial is None:
        return {'no_missing': np.array(True)}
    inference, training, q, labels = initial
    return {'no_missing': np.array(False), 'inference': inference.counts, 'training': training.counts,
            'q': q, 'labels': labels, 'delta_inference': np.array(inference.delta),
            'delta_training': np.array(training.delta)}


def initial_from_arrays(data, hidden, arrays, model):
    """Validate saved state against allowed raw input, without fitting an initializer."""
    _time_values(data)
    values = model_input(data, hidden)
    no_missing = np.isfinite(values[:, :-1]).all()
    flag = arrays.get('no_missing')
    if flag is None or flag.shape != () or flag.dtype != bool or bool(flag) != no_missing:
        raise BaselineInputError('initial artifact: invalid no_missing flag')
    if no_missing:
        if set(arrays) != {'no_missing'}:
            raise BaselineInputError('initial artifact: unexpected no_missing payload')
        return None
    expected_keys = {'no_missing', 'inference', 'training', 'q', 'labels', 'delta_inference', 'delta_training'}
    if set(arrays) != expected_keys:
        raise BaselineInputError('initial artifact: missing or unexpected arrays')
    observed = np.isfinite(values)
    visible = _visible_data(data, observed)
    q = arrays['q']
    if q.shape != data.counts.shape or q.dtype != bool or not q.any() or (q & ~observed[:, :-1]).any():
        raise BaselineInputError('initial artifact: invalid Q mask')
    expected_q = make_mask(visible, 'random_cell', model.train_mask_rate, model.seed)[0]
    if not np.array_equal(q, expected_q):
        raise BaselineInputError('initial artifact: Q does not match the configured recipe')
    labels = np.full(data.counts.shape, np.nan)
    labels[q] = (visible.counts / visible.totals[:, None])[q]
    if not np.array_equal(labels, arrays['labels'], equal_nan=True):
        raise BaselineInputError('initial artifact: labels do not match visible Q')
    result = []
    for name, mask in (('inference', hidden), ('training', hidden | q)):
        parts = model_input(data, mask)
        positives = parts[np.isfinite(parts) & (parts > 0)]
        expected_delta = float(positives.min() / 2)
        delta = arrays[f'delta_{name}']
        if delta.shape != () or not np.isfinite(delta) or float(delta) != expected_delta:
            raise BaselineInputError('initial artifact: incompatible CLR delta')
        raw = _visible_data(data, np.isfinite(parts))
        counts = arrays[name]
        check_completed(raw, counts, np.zeros(data.counts.shape, bool))
        result.append(InitialView(raw, counts.copy(), np.isfinite(parts), expected_delta))
    return result[0], result[1], q.copy(), labels
