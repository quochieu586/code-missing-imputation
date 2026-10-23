"""Named variants; parameters are explicit and serializable by the caller."""
from dataclasses import dataclass
from typing import Literal
import copy
import inspect
from .common import BaselineError
from .hron import HronKNN, HronIterative
from .micoda import MICoDa
from .tsagris import TsagrisKNN, TsagrisAdaptive

METHODS = ('hron_knn', 'hron_ls', 'hron_lts', 'micoda_v1', 'micoda_v2',
           'tsagris_basic', 'tsagris_alpha', 'tsagris_global_cv', 'tsagris_adaptive')


@dataclass(frozen=True)
class AdapterRecipe:
    """Declared contract for adapting one numerical core to count panels."""

    input_domain: Literal['positive', 'nonnegative']
    prediction_policy: Literal['rowwise_adaptation', 'population_adaptation']


def adapter_recipe(name):
    core = make_core(name)
    return AdapterRecipe(core.input_domain, core.prediction_policy)


def core_configuration(core):
    """Effective constructor options only, valid before or after fitting."""
    config = {'class': type(core).__name__}
    for name in inspect.signature(type(core).__init__).parameters:
        if name == 'self' or not hasattr(core, name):
            continue
        value = getattr(core, name)
        if value is None or isinstance(value, (str, int, float, bool, tuple, list)):
            config[name] = value
        elif type(value).__module__.startswith('longitudinal_impute.baselines'):
            config[name] = core_configuration(value)
    if isinstance(core, HronIterative):
        nested = core_configuration(core.initializer_)
        config.update({k: nested[k] for k in ('k', 'donor_policy', 'adjustment')})
    return config


def configure_core(core, *, degenerate_policies=False):
    """Single policy owner, also used by the direct adapter compatibility API."""
    configured = copy.deepcopy(core)
    if degenerate_policies and isinstance(configured, (TsagrisKNN, TsagrisAdaptive)):
        configured.zero_anchor = 'global'
        configured.zero_missing_mass = 'uniform'
    return configured


def make_core(name, *, degenerate_policies=False, **parameters):
    constructors = {'hron_knn': HronKNN,
                    'hron_ls': lambda **p: HronIterative(regression='ls', **p),
                    'hron_lts': lambda **p: HronIterative(regression='lts', **p),
                    'micoda_v1': lambda **p: MICoDa(version='v1', **p),
                    'micoda_v2': lambda **p: MICoDa(version='v2', **p),
                    'tsagris_basic': lambda **p: TsagrisKNN(alpha=1., **p),
                    'tsagris_alpha': TsagrisKNN,
                    'tsagris_global_cv': lambda **p: TsagrisAdaptive(adaptive=False, **p),
                    'tsagris_adaptive': lambda **p: TsagrisAdaptive(adaptive=True, **p)}
    if name not in constructors:
        raise BaselineError('Unknown method: ' + name)
    try:
        return configure_core(constructors[name](**parameters), degenerate_policies=degenerate_policies)
    except TypeError as exc:
        raise BaselineError(f'Unsupported parameters for {name}: {exc}') from exc
