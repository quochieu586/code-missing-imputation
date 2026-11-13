"""One composition is one location/date across K variants; never across time."""
import numpy as np
from scipy.stats import rankdata
from src.core.transforms import clr


def feature_order(x, train_idx, groups):
    """Geometric mean absolute Spearman, fitted exclusively on training rows."""
    order = []
    for group in sorted(set(groups)):
        idx = np.flatnonzero(np.asarray(groups) == group)
        if len(idx) == 1:
            order.extend(idx)
            continue
        ranks = np.apply_along_axis(rankdata, 0, x[np.asarray(train_idx)][:, idx])
        with np.errstate(invalid='ignore', divide='ignore'):
            rho = np.nan_to_num(np.abs(np.corrcoef(ranks.T)))
            scores = np.exp(np.mean(np.log(rho), axis=1))
        order.extend(idx[np.argsort(-scores, kind='stable')])
    return np.asarray(order)


def bridge(x1, m0):
    z = clr(x1)
    mask = np.asarray(m0, dtype=bool).copy()
    if z.shape != mask.shape:
        raise ValueError('X1 and M0 shape mismatch')
    mask.setflags(write=False)
    return z, mask
