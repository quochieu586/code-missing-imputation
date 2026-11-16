"""Evaluation v1 of the count-table imputations: the five tiers (README.md in this folder).

    tier0     run status, reliability, pairing of run directories (F2), failure policy
    tier1     errors on masked cells with known truth, common-seed pairs (F07), sign test
    tier2     D1 / D2 / D3 on the real missing cells: description, not accuracy
    tier3     use cases U1-U4: what the tier 2 diagnostics are good for, on masked cells
    metrics   per-cell scores and the long prediction file of masked runs (used by run.py)
    masks     the whole_feature_matched mask (used by data.make_mask)
    protocol  frozen configuration, unused seeds, decisions on the hypotheses
    pipeline  the commands check / masks / run / evaluate (python -m longitudinal_impute.evaluation)

The public names below are loaded on first use, so that run.py and data.py can import metrics and
masks without loading the pipeline (which itself imports run.py).
"""
import importlib

_API = {
    'cell_errors': 'metrics', 'masked_errors': 'metrics', 'forced_zero': 'metrics', 'read_predictions': 'metrics',
    'score_predictions': 'metrics', 'read_target': 'masks', 'native_target': 'masks',
    'status_rows': 'tier0', 'reliability': 'tier0', 'validate_pair': 'tier0',
    'load_run': 'tier1', 'table_runs': 'tier1', 'pair': 'tier1', 'sign_test': 'tier1',
    'diagnose': 'tier2', 'native_cells': 'tier2', 'calibration_cells': 'tier2',
    'one_case': 'tier3', 'read_config': 'protocol', 'frozen_sha256': 'protocol', 'check_unused': 'protocol',
    'setup': 'pipeline', 'draw_masks': 'pipeline', 'run_all': 'pipeline', 'evaluate': 'pipeline',
}
__all__ = sorted(_API)


def __getattr__(name):
    if name in _API:
        return getattr(importlib.import_module(f'.{_API[name]}', __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
