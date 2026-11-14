"""Handoff M1/M2/M3 on complete ground-truth rows only."""
import numpy as np
from src.core.transforms import clr


def evaluate(truth, prediction, heldout, zero):
    z, zp = clr(truth), clr(prediction)
    error = np.abs(z-zp)
    def mae(mask):
        return float(error[mask].mean()) if mask.any() else None
    affected = heldout.any(-1)
    p = prediction/prediction.sum(-1, keepdims=True)
    q = truth/truth.sum(-1, keepdims=True)
    return dict(mae_clr_all=mae(heldout), mae_clr_nonzero=mae(heldout & ~zero),
                mae_clr_pseudocount=mae(heldout & zero),
                n_target=int(heldout.sum()), n_nonzero=int((heldout & ~zero).sum()),
                n_pseudocount=int((heldout & zero).sum()), n_rows=int(affected.sum()),
                m2=float(((z-zp)[affected]**2).sum(-1).mean()),
                m3=float(np.linalg.norm(np.cov(z.T)-np.cov(zp.T))/(z.shape[-1]-1)),
                shannon_true=float(-(q*np.log(q)).sum(-1).mean()),
                shannon_pred=float(-(p*np.log(p)).sum(-1).mean()),
                zero_fraction_true=float(zero.mean()),
                below_detection_fraction_true=float((q < 1e-4).mean()),
                below_detection_fraction_pred=float((p < 1e-4).mean()))
