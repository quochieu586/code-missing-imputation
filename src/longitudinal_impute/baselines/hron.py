"""Hron, Templ & Filzmoser (2010), section 3.1: kNN imputation, strategy (2a).

Each missing cell (i, j) is filled on its own. Donors are rows observed on
O_i and j; neighbours use the Aitchison distance (clr + Euclid) on O_i; the
donor cells are rescaled by Eq. 7 (median/median, default) or Eq. 5
(sum/sum) and combined with the Eq. 6 median. Input must be strictly
positive; zero replacement happens before the model (data.model_input).
"""
import numpy as np
from ..errors import BaselineError, NotFittedError, NumericalError

TIE_TOLERANCE = 1e-12


def _clr(parts):
    logs = np.log(parts)
    return logs - logs.mean(axis=-1, keepdims=True)


def _check_positive(values, width=None):
    x = np.array(values, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 2:
        raise BaselineError('Hron expects a nonempty N x D matrix with D >= 2')
    if width is not None and x.shape[1] != width:
        raise BaselineError('Feature dimension differs from fitted data',
                            context={'expected': width, 'got': x.shape[1]})
    bad = np.argwhere(np.isinf(x) | (np.isfinite(x) & (x <= 0)))
    if len(bad):
        i, j = bad[0]
        raise BaselineError(f'Hron requires strictly positive parts; row {i}, feature {j} is {x[i, j]!r}',
                            context={'row': int(i), 'feature': int(j), 'cells': len(bad)})
    return x


class HronKNN:
    input_domain = 'positive'

    def __init__(self, k=3, *, adjustment='median', ties='group'):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise BaselineError('k must be a positive integer')
        if adjustment not in ('median', 'sum'):
            raise BaselineError("adjustment must be 'median' (Eq. 7) or 'sum' (Eq. 5)")
        if ties not in ('group', 'index'):
            raise BaselineError("ties must be 'group' or 'index'")
        self.k, self.adjustment, self.ties = k, adjustment, ties

    def fit(self, values):
        self.donors_ = _check_positive(values)
        return self

    def _select(self, distances):
        """Positions of the chosen donors among `distances` (already in donor-index order)."""
        if len(distances) <= self.k:
            return np.arange(len(distances)), False
        kth = np.partition(distances, self.k - 1)[self.k - 1]
        threshold = TIE_TOLERANCE * max(1., abs(kth))
        closer = np.flatnonzero(distances < kth - threshold)
        tied = np.flatnonzero(np.abs(distances - kth) <= threshold)
        tie = len(closer) + len(tied) > self.k
        if self.ties == 'index':
            tied = tied[:self.k - len(closer)]
        return np.r_[closer, tied], tie

    def predict(self, values):
        if not hasattr(self, 'donors_'):
            raise NotFittedError('Fit HronKNN before prediction')
        x = _check_positive(values, self.donors_.shape[1])
        out = x.copy()
        valid = np.isfinite(self.donors_)
        reduce = np.median if self.adjustment == 'median' else np.sum
        cells = ties = few = 0
        used = []
        observed = np.isfinite(x)
        patterns = {}
        for i in np.flatnonzero(~observed.all(axis=1)):
            patterns.setdefault(observed[i].tobytes(), []).append(i)
        for rows in patterns.values():
            obs = np.flatnonzero(observed[rows[0]])
            if not len(obs):
                raise BaselineError(f'row {rows[0]}: no observed part to measure distance',
                                    context={'row': int(rows[0])})
            # Donor clr and scale on O_i are shared by every target j of this pattern.
            base = np.flatnonzero(valid[:, obs].all(axis=1))
            parts = self.donors_[base][:, obs]
            donor_clr, donor_scale = _clr(parts), reduce(parts, axis=1)
            for i in rows:
                distances = np.linalg.norm(donor_clr - _clr(x[i, obs]), axis=1)
                row_scale = reduce(x[i, obs])
                for j in np.flatnonzero(~observed[i]):
                    eligible = np.flatnonzero(valid[base, j])
                    if not len(eligible):
                        raise BaselineError(f'row {i}, feature {j}: no donor observed on O_i and j',
                                            context={'row': int(i), 'feature': int(j)})
                    chosen, tie = self._select(distances[eligible])
                    chosen = eligible[chosen]
                    # Divide before multiplying: x_lj / scale_l stays representable when both are tiny.
                    with np.errstate(over='ignore'):    # overflow is caught by the check below
                        value = np.median(self.donors_[base[chosen], j] / donor_scale[chosen] * row_scale)
                    if not np.isfinite(value):
                        raise NumericalError(f'row {i}, feature {j}: rescaled donor value is not finite',
                                             context={'row': int(i), 'feature': int(j)})
                    out[i, j] = value
                    cells += 1
                    ties += tie
                    few += len(eligible) < self.k
                    used.append(len(chosen))
        self.info_ = {'method': 'hron_knn_2a', 'k': self.k, 'adjustment': self.adjustment,
                      'ties': self.ties, 'filled_cells': cells, 'tied_cells': int(ties),
                      'cells_with_fewer_than_k_donors': int(few),
                      'donors_used_mean': float(np.mean(used)) if used else 0.,
                      'donors_used_max': int(max(used)) if used else 0}
        return out
