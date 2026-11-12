"""Step 7 checks on the COVID table: step 5 cut points, old masks unchanged, firewall of the matched mask.

    PYTHONPATH=nextgen/src python -W error::RuntimeWarning -m pytest tv2_step7/test_matched.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('matched', HERE / 'matched.py')
matched = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matched)
t2, mm = matched.t2, matched.mm
from longitudinal_impute.data import Dataset, make_mask  # noqa: E402


def covid():
    return matched.tier2.load_csv(matched.tier2.DATA, matched.tier2.SCHEMA)


def test_cut_points_are_read_from_step_5():
    cuts = matched.step5_cuts()
    data = covid()
    assert cuts == [17., 149.]
    assert tuple(cuts) == t2.n_strata(t2.row_facts(data), t2.native_cells(data)['eval'])[1]   # consistency only


def test_whole_feature_and_random_cell_masks_are_unchanged():
    data = covid()
    manifest = json.loads((matched.REPO / 'tv2_step1/results_manifest.json').read_text())
    for kind in ('whole_feature', 'random_cell'):
        for seed, digest in manifest[kind][0]['mask_sha256'].items():
            assert matched.tier01.step1.digest(make_mask(data, kind, .05, int(seed))[0]) == digest


def test_firewall_on_covid_counts():
    data = covid()
    target = mm.native_target(data, [17., 149.])
    hidden, stats = make_mask(data, 'whole_feature_matched', .05, 0, target)
    rng = np.random.default_rng(1)
    other = np.floor(rng.dirichlet(np.ones(len(data.targets) + 1), len(data.totals))[:, :-1] * data.totals[:, None])
    other[~np.isfinite(data.counts)] = np.nan
    changed = Dataset(other, data.totals, data.targets, data.entities, data.fields, data.rows, data.lines, data.days,
                      data.week, data.time_kind, data.groups, data.group_map)
    assert np.array_equal(make_mask(changed, 'whole_feature_matched', .05, 0, target)[0], hidden)
    assert stats['reached_goal'] and not (hidden & ~np.isfinite(data.counts)).any()
