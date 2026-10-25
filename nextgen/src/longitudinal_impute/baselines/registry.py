"""Compatibility helper used inside the unchanged Tsagris modules.

No registry or runner uses this module. Remove with the baseline API review.
"""
import copy
from .tsagris import TsagrisKNN, TsagrisAdaptive


def configure_core(core, *, degenerate_policies=False):
    """Copy the model with its existing optional zero policies."""
    configured = copy.deepcopy(core)
    if degenerate_policies and isinstance(configured, (TsagrisKNN, TsagrisAdaptive)):
        configured.zero_anchor = 'global'
        configured.zero_missing_mass = 'uniform'
    return configured
