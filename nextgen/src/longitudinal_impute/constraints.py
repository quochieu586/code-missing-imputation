"""Counts output contract, independent of network/baseline implementation.

Copy C exactly in evaluation and O in final native imputation. Allocate only
R=N-sum(visible counts) to unknown parts; optional Other takes unused mass.
R<0 is an error; R=0 forces missing counts to zero. Never close all 17 parts
to N unless an independently justified equality constraint actually applies.
"""
from typing import Protocol
from .contracts import Array, ConditioningPanel, ImputationResult


class CountConstraint(Protocol):
    def apply(self, scores: Array, panel: ConditioningPanel) -> ImputationResult: ...


class CountBudgetConstraint:
    """Adapt nonnegative count proposals; never mutate original observations.

    With D+1 columns the last is an unknown Other proposal on partial rows;
    allocate residual jointly over missing targets and Other. With D columns,
    keep proposals when feasible, shrink missing parts only if they exceed R.
    These are declared dataset adaptations, not paper-original operations.
    """
    def apply(self, scores: Array, panel: ConditioningPanel) -> ImputationResult:
        import numpy as np
        from .errors import BaselineError, NumericalError
        from .baselines.panels import safe_counts
        raw = safe_counts(panel)
        proposal = np.array(scores, dtype=float, copy=True)
        n, d = raw.shape
        if proposal.shape not in ((n, d), (n, d + 1)):
            raise BaselineError('Count proposal must be N x D or N x (D+Other)')
        if not np.isfinite(proposal).all() or (proposal < 0).any():
            raise BaselineError('Count proposals must be finite nonnegative')
        mask = np.asarray(panel.conditioning_mask)
        totals = np.asarray(panel.totals, float)
        result = np.where(mask, raw, 0.)
        other = np.empty(n)
        explicit = proposal.shape[1] == d + 1
        records = []
        for i in range(n):
            unknown = ~mask[i]
            residual = totals[i] - result[i].sum()
            if not unknown.any():
                other[i] = residual
                continue
            if residual == 0:
                other[i] = 0.
                records.append({'row': i, 'policy': 'zero_residual'})
                continue
            weights = proposal[i, :d][unknown]
            joint = np.r_[weights, proposal[i, -1]] if explicit else weights
            scale = float(joint.max(initial=0.))
            if explicit and scale == 0:
                raise BaselineError('No proposed mass for unknown targets and Other')
            # A negligible relative part may underflow here; retain original
            # weights for multiplication when the common factor is representable.
            with np.errstate(under='ignore'):
                scaled = joint / scale if scale else joint.copy()
            scaled_mass = float(scaled.sum())
            with np.errstate(under='ignore'):
                cap = bool(explicit or (scale > 0 and scale > residual / scaled_mass))
            if cap:
                with np.errstate(over='ignore', under='ignore'):
                    factor = (residual / scaled_mass) / scale
                    allocated = (joint * factor if np.isfinite(factor) and factor > 0
                                 else (scaled / scaled_mass) * residual)
                result[i, unknown] = allocated[:len(weights)]
                if not allocated.any():
                    raise NumericalError('Residual allocation is not representable at float64 precision')
            else:
                result[i, unknown] = weights
            other[i] = float(allocated[-1]) if explicit else max(0., totals[i] - result[i].sum())
            records.append({'row': i, 'policy': 'joint_other_allocation' if explicit else 'inequality_cap',
                            'weight_scale': scale, 'scaled_mass': scaled_mass,
                            'residual': float(residual), 'rescaled': cap})
        if not np.isfinite(result).all() or not np.isfinite(other).all():
            raise NumericalError('Count allocation produced nonfinite outputs')
        if not np.array_equal(result[mask], raw[mask]):
            raise AssertionError('Observed count preservation violated')
        with np.errstate(over='ignore'):
            row_sums = result.sum(axis=1)
        if not np.isfinite(row_sums).all():
            raise NumericalError('Allocated count sum is nonfinite')
        if ((result < 0).any() or (other < 0).any()
                or (row_sums - totals > 1e-10 * np.maximum(1., totals)).any()):
            raise NumericalError('Count budget violated during allocation')
        differences = np.abs(result - proposal[:, :d])
        repair_scale = float(differences.max(initial=0.))
        with np.errstate(under='ignore'):
            scaled_repair = float((differences / repair_scale).sum()) if repair_scale else 0.
        repair_l1 = (repair_scale * scaled_repair
                     if scaled_repair <= 1 or repair_scale <= np.finfo(float).max / scaled_repair else None)
        return ImputationResult(panel.row_ids, result, {
            'other': other.tolist(), 'explicit_other_proposal': explicit,
            'proposal_counts': proposal.tolist(), 'repair_l1': repair_l1,
            'repair_scale': repair_scale, 'scaled_repair_l1': scaled_repair,
            'records': records, 'observed_lock': 'conditioning_mask'})
