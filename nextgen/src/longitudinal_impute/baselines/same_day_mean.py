"""Same-day mean: the proportion of target j seen around the same date in the same group.

For a missing cell (i, j) the reference is the N-weighted mean of the visible
proportions c/N of target j, i.e. sum(c) / sum(N), over the OTHER rows that
  * lie in the same group as row i (schema group_col; the whole table without one),
  * have a time within +-window of row i's time,
  * and, with exclude_same_location = true, belong to another entity than row i.
Only visible cells enter: a hidden or missing cell is NaN in the model input, so it
never contributes, and row i itself is missing on j.

A window with fewer than min_obs visible cells is doubled, at most max_widen times;
if it is still short, the cell gets the N-weighted mean of target j over the whole
table (all groups; without the same entity when exclude_same_location = true).
info_['cells_by_level'] counts the cells resolved at each level; levels_ (N x D) keeps the
level of every filled cell (0 = first window, k = k-th widening, max_widen + 1 = global,
-1 = not filled), so a report can count the hidden cells alone.

The window depends on Dataset.time_kind:
  * 'date': +-window_days days, then 2x, 4x, ... (max_widen doublings);
  * 'numeric' (visit index 1, 2, 3 ...): the same time value first, then +-1, +-2, +-4 ...;
  * 'label' (free text): the same label only, no widening;
  * no time column: the whole group.

The proposals of a row are scaled down when they exceed its residual R = 1 - sum(observed);
Other gets the rest (common.fill_within_residual), and restore_counts turns them into counts.
"""
import numpy as np
from ..errors import BaselineInputError, InsufficientDataError, NotFittedError
from .common import fill_within_residual, observed_columns


def _codes(labels, n):
    if labels is None:
        return np.zeros(n, dtype=np.int64)
    if len(labels) != n:
        raise BaselineInputError('context labels must have one entry per row')
    return np.unique(np.asarray(labels, dtype=object).astype(str), return_inverse=True)[1].astype(np.int64)


class SameDayMean:
    input_domain = 'nonnegative'
    needs_context = True        # run.run_model passes totals, time and groups of the rows

    def __init__(self, window_days=28, *, min_obs=20, max_widen=2, exclude_same_location=False):
        for name, value in (('window_days', window_days), ('min_obs', min_obs), ('max_widen', max_widen)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if min_obs < 1:
            raise ValueError('min_obs must be at least 1')
        if not isinstance(exclude_same_location, bool):
            raise ValueError('exclude_same_location must be true or false')
        self.window_days, self.min_obs, self.max_widen = window_days, min_obs, max_widen
        self.exclude_same_location = exclude_same_location

    def windows(self, kind):
        """Half-widths of the successive windows, in the unit of Dataset.days."""
        if kind is None:
            return [np.inf]
        if kind == 'date':
            return [self.window_days * 2 ** level for level in range(self.max_widen + 1)]
        if kind == 'numeric':
            return [0] + [2 ** (level - 1) for level in range(1, self.max_widen + 1)]
        if kind == 'label':
            return [0]
        raise BaselineInputError(f'unknown time kind {kind!r}')

    def fit(self, values, context=None):
        x = observed_columns(values, 'SameDayMean')
        n = len(x)
        context = dict(context or {})
        totals = context.get('totals')
        if totals is None:
            raise BaselineInputError('SameDayMean needs the row totals N in its context')
        totals = np.asarray(totals, float)
        if totals.shape != (n,) or not np.isfinite(totals).all() or (totals <= 0).any():
            raise BaselineInputError('row totals must be finite, positive, one per row')
        kind, days = context.get('time_kind'), context.get('days')
        if (kind is None) != (days is None):
            raise BaselineInputError('time_kind and days must be given together')
        days = np.zeros(n) if days is None else np.asarray(days, float)
        if days.shape != (n,) or not np.isfinite(days).all():
            raise BaselineInputError('days must be finite, one per row')
        groups = _codes(context.get('groups'), n)
        entities = _codes(context.get('entities'), n)
        if self.exclude_same_location and context.get('entities') is None:
            raise BaselineInputError('exclude_same_location needs the entity of every row')
        # (group, entity) pairs, so the same-entity correction stays inside the group window.
        pairs = np.unique(np.column_stack((groups, entities)), axis=0, return_inverse=True)[1].ravel()
        self.visible_ = x[:, :-1].copy()
        self.totals_, self.days_, self.kind_ = totals, days, kind
        self.groups_, self.entities_, self.pairs_ = groups, entities, pairs
        return self

    def predict(self, values):
        if not hasattr(self, 'visible_'):
            raise NotFittedError('Fit SameDayMean before prediction')
        x = observed_columns(values, 'SameDayMean')
        if x.shape[0] != len(self.visible_) or x.shape[1] - 1 != self.visible_.shape[1]:
            raise BaselineInputError('SameDayMean predicts the rows it was fitted on')
        windows = self.windows(self.kind_)
        names = ['window'] + [f'widened_{level}' for level in range(1, len(windows))] + ['global']
        counts = dict.fromkeys(names, 0)
        per_target = {}
        days, totals = self.days_, self.totals_
        axis = np.unique(days)
        position = np.searchsorted(axis, days)
        proposal = np.zeros(self.visible_.shape)
        levels = np.full(self.visible_.shape, -1, dtype=np.int64)
        for j in range(self.visible_.shape[1]):
            need = np.flatnonzero(~np.isfinite(x[:, j]))
            if not len(need):
                continue
            seen = np.isfinite(self.visible_[:, j])
            mass = np.where(seen, self.visible_[:, j] * totals, 0.)
            weight = np.where(seen, totals, 0.)
            # Cumulative sums over the sorted time axis, per group and per (group, entity).
            tables = {}
            for key, codes in (('group', self.groups_), ('pair', self.pairs_)):
                if key == 'pair' and not self.exclude_same_location:
                    continue
                shape = (codes.max() + 1, len(axis) + 1)
                stack = np.zeros((3, *shape))
                for k, add in enumerate((seen.astype(float), mass, weight)):
                    np.add.at(stack[k], (codes, position + 1), add)
                tables[key] = np.cumsum(stack, axis=2)
            level = np.full(len(need), -1)
            reference = np.zeros(len(need))
            for step, width in enumerate(windows):
                todo = np.flatnonzero(level < 0)
                if not len(todo):
                    break
                rows = need[todo]
                if np.isinf(width):
                    lo, hi = np.zeros(len(rows), np.int64), np.full(len(rows), len(axis))
                else:
                    lo = np.searchsorted(axis, days[rows] - width, side='left')
                    hi = np.searchsorted(axis, days[rows] + width, side='right')
                table = tables['group'][:, self.groups_[rows]]
                window = table[:, np.arange(len(rows)), hi] - table[:, np.arange(len(rows)), lo]
                if self.exclude_same_location:
                    own = tables['pair'][:, self.pairs_[rows]]
                    window -= own[:, np.arange(len(rows)), hi] - own[:, np.arange(len(rows)), lo]
                cells, numerator, denominator = window
                ok = cells >= self.min_obs - .5      # counts are float sums of ones
                reference[todo[ok]] = numerator[ok] / denominator[ok]
                level[todo[ok]] = step
                counts[names[step]] += int(ok.sum())
            todo = np.flatnonzero(level < 0)
            if len(todo):
                rows = need[todo]
                numerator = np.full(len(rows), mass.sum())
                denominator = np.full(len(rows), weight.sum())
                cells = np.full(len(rows), float(seen.sum()))
                if self.exclude_same_location:
                    own = self.entities_[rows]
                    size = self.entities_.max() + 1
                    numerator -= np.bincount(self.entities_, mass, size)[own]
                    denominator -= np.bincount(self.entities_, weight, size)[own]
                    cells -= np.bincount(self.entities_, seen.astype(float), size)[own]
                if (cells < .5).any():
                    i = int(rows[np.argmax(cells < .5)])
                    raise InsufficientDataError(f'row {i}, feature {j}: no visible cell of the target to average',
                                                context={'row': i, 'feature': j})
                reference[todo] = numerator / denominator
                counts['global'] += len(todo)
            proposal[need, j] = np.clip(reference, 0., 1.)
            levels[need, j] = np.where(level < 0, len(windows), level)
            per_target[j] = {name: int((level == k).sum()) if name != 'global' else int((level < 0).sum())
                             for k, name in enumerate(names)}
        out, scaled = fill_within_residual(x, proposal)
        self.levels_ = levels
        self.info_ = {'method': 'same_day_mean', 'time_kind': self.kind_,
                      'window_days': self.window_days, 'min_obs': self.min_obs, 'max_widen': self.max_widen,
                      'exclude_same_location': self.exclude_same_location,
                      'windows': [None if np.isinf(w) else w for w in windows],
                      'groups': int(self.groups_.max() + 1), 'filled_cells': int(sum(counts.values())),
                      'cells_by_level': counts, 'cells_by_level_per_feature': per_target,
                      'rows_scaled_to_residual': scaled}
        return out
