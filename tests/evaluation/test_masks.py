"""whole_feature_matched (step 7): selection, firewall, guard, weights, strata cut points, runner, COVID checks."""
import hashlib
import json

import numpy as np
import pytest

from longitudinal_impute import run
from longitudinal_impute.evaluation import masks as mm, tier2 as t2
from longitudinal_impute.evaluation.report import ROOT, digest
from longitudinal_impute.data import Dataset, make_mask
from longitudinal_impute.evaluation.metrics import forced_zero
from test_tier2 import panel

TARGET = {'n_cuts': [17., 149.], 'pi': {'G1|N1': .2, 'G2|N1': .5, 'G2|N2': .3}}


def skewed(seed=0):
    """G1: 12 locations with large N; G2: 4 locations with small N. Counts never read by the mask."""
    rng = np.random.default_rng(seed)
    entities, days, groups, totals = [], [], [], []
    for g, locations, n in (('G1', 12, (150, 400)), ('G2', 4, (5, 17))):
        for loc in range(locations):
            for t in range(20):
                entities.append(f'{g}L{loc}')
                days.append(7 * t)
                groups.append(g)
                totals.append(rng.integers(*n))
    totals = np.array(totals, float)
    counts = np.floor(rng.dirichlet(np.ones(5) * .5, len(totals))[:, :4] * totals[:, None])
    counts[:, 3][np.array([e.endswith('L0') for e in entities])] = np.nan       # one missing series per group
    return Dataset(counts, totals, ('a', 'b', 'c', 'd'), tuple(entities), [], [], [], np.array(days), None, 'date',
                   tuple(groups))


# ---------- selection ----------

def test_mask_hides_whole_observed_series_reaches_the_rate_and_repeats():
    data = panel(seed=1)
    hidden, stats = make_mask(data, 'whole_feature_matched', .05, 3, TARGET)
    observed = np.isfinite(data.counts)
    assert not (hidden & ~observed).any()
    entities = np.asarray(data.entities)
    for e, j in {(entities[i], j) for i, j in zip(*np.nonzero(hidden))}:
        series = (entities == e) & observed[:, j]
        assert hidden[series, j].all()                                    # the whole observed series
    goal = .05 * observed.sum()
    assert stats['reached_goal'] and hidden.sum() >= goal
    assert stats['pairs_hidden'] >= 1 and hidden.sum() - goal < max(
        ((entities == e) & observed[:, j]).sum() for e, j in {(entities[i], j) for i, j in zip(*np.nonzero(hidden))})
    again, _ = make_mask(data, 'whole_feature_matched', .05, 3, TARGET)
    assert np.array_equal(hidden, again)
    assert not np.array_equal(hidden, make_mask(data, 'whole_feature_matched', .05, 4, TARGET)[0])


def test_firewall_counts_never_change_the_mask():
    data = skewed()
    hidden, _ = make_mask(data, 'whole_feature_matched', .1, 0, TARGET)
    rng = np.random.default_rng(9)
    other = np.floor(rng.dirichlet(np.ones(5), len(data.totals))[:, :4] * data.totals[:, None])
    other[~np.isfinite(data.counts)] = np.nan                            # same N, dates, places, NaN pattern
    changed = Dataset(other, data.totals, data.targets, data.entities, [], [], [], data.days, None, 'date', data.groups)
    assert np.array_equal(make_mask(changed, 'whole_feature_matched', .1, 0, TARGET)[0], hidden)


def test_guard_never_leaves_a_row_without_a_visible_target():
    data = panel(seed=2)
    hidden, stats = make_mask(data, 'whole_feature_matched', .6, 0, TARGET)
    observed = np.isfinite(data.counts)
    assert ((observed & ~hidden).any(axis=1) | ~observed.any(axis=1)).all()
    assert stats['pairs_skipped'] > 0


# ---------- weights, strata, matching ----------

def test_weights_follow_the_formula_on_a_hand_toy():
    counts = np.array([[1., 2.], [3., np.nan], [0., 1.], [2., 2.]])
    data = Dataset(counts, np.array([10., 200., 10., 10.]), ('t0', 't1'), ('A', 'A', 'B', 'B'), [], [], [],
                   np.array([0, 7, 0, 7]), None, 'date', ('G1', 'G1', 'G2', 'G2'))
    observed = np.isfinite(counts)
    pairs = [('A', 0), ('A', 1), ('B', 0), ('B', 1)]
    target = {'n_cuts': [17., 149.], 'pi': {'G1|N1': .5, 'G2|N1': .25, 'G2|N2': .25}}
    w, info = mm.weights(data, np.asarray(data.entities), observed, pairs, target)
    # p: G1|N1 2/7, G2|N1 4/7, G2|N2 0 (dropped); A row 1 is G1|N3, outside pi.
    assert w == pytest.approx([.5 / (2 / 7), .5 / (2 / 7), 2 * .25 / (4 / 7), 2 * .25 / (4 / 7)])
    assert info['unreachable_strata'] == ['G2|N2'] and info['unreachable_mass'] == pytest.approx(.25)
    assert info['p'] == pytest.approx({'G1|N1': 2 / 7, 'G2|N1': 4 / 7, 'G2|N2': 0.})
    order = mm.draw_order(np.random.default_rng(0), np.array([1., 0., 2.]))
    assert sorted(order.tolist()) == [0, 2]                               # weight 0 is never drawn


def test_strata_use_the_given_cut_points_never_recomputed():
    labels = mm.strata(['G', 'G', 'G', 'G'], [17, 18, 149, 150], [17., 149.])
    assert labels.tolist() == ['G|N1', 'G|N2', 'G|N2', 'G|N3']
    data = skewed()
    w1, info1 = mm.weights(data, np.asarray(data.entities), np.isfinite(data.counts),
                           sorted(set(zip(np.asarray(data.entities)[np.nonzero(np.isfinite(data.counts))[0]],
                                          np.nonzero(np.isfinite(data.counts))[1]))), TARGET)
    assert set(info1['p']) == set(TARGET['pi'])                          # strata of pi with the target's cuts
    with pytest.raises(ValueError, match='n_cuts'):
        mm.check_target({'n_cuts': [149., 17.], 'pi': {'G1|N1': 1.}})
    with pytest.raises(ValueError, match='group'):
        mm.strata(None, [1.], [17., 149.])


def test_matched_mask_is_closer_to_the_target_than_whole_feature():
    data = skewed()
    labels = mm.strata(data.groups, data.totals, TARGET['n_cuts'])
    pi = {'G2|N1': 1.}
    tv = {}
    for kind, target in (('whole_feature', None), ('whole_feature_matched', {'n_cuts': [17., 149.], 'pi': pi})):
        values = []
        for seed in range(5):
            hidden = make_mask(data, kind, .1, seed, target)[0]
            values.append(mm.total_variation(mm.shares(labels, hidden & ~forced_zero(data, hidden)), pi))
        tv[kind] = np.mean(values)
    assert tv['whole_feature_matched'] < tv['whole_feature'] - .3


def test_native_target_counts_the_model_decided_missing_cells():
    counts = np.array([[4., np.nan], [1., np.nan], [np.nan, 2.]])
    data = Dataset(counts, np.array([4., 5., 200.]), ('a', 'b'), ('x', 'y', 'z'), [], [], [], np.array([0, 0, 0]),
                   None, 'date', ('G1', 'G1', 'G2'))
    out = mm.native_target(data, [17., 149.])
    assert out == {'n_cuts': [17., 149.], 'pi': {'G1|N1': .5, 'G2|N3': .5}}   # row 0 is forced to 0
    assert mm.total_variation({'a': .5, 'b': .5}, {'a': 1.}) == pytest.approx(.5)


# ---------- runner ----------

def grouped_csv(tmp_path):
    lines = ['location,date,continent,N,a,b,c']
    rng = np.random.default_rng(0)
    for k, loc in enumerate(['P', 'Q', 'R', 'S', 'T', 'U']):
        for t in range(6):
            n = int(rng.integers(5, 300))
            a, b = int(rng.integers(0, n // 3 + 1)), int(rng.integers(0, n // 3 + 1))
            c = '' if loc == 'P' else str(int(rng.integers(0, n // 3 + 1)))
            lines.append(f'{loc},2021-{1 + t:02d}-04,{"G1" if k < 3 else "G2"},{n},{a},{b},{c}')
    data = tmp_path / 'd.csv'
    data.write_text('\n'.join(lines) + '\n')
    schema = tmp_path / 's.toml'
    schema.write_text('[dataset]\nentity="location"\ntime="date"\ntotal="N"\ntargets=["a","b","c"]\ngroup_col="continent"\n')
    models = tmp_path / 'm.toml'
    models.write_text('[mean_fill]\n')
    return data, schema, models


def test_runner_needs_the_target_for_the_matched_mask_only(tmp_path):
    data, schema, models = grouped_csv(tmp_path)
    target = tmp_path / 't.json'
    target.write_text(json.dumps({'n_cuts': [17., 149.], 'pi': {'G1|N2': .5, 'G2|N3': .5}}))
    base = ['--data', str(data), '--schema', str(schema), '--models', str(models), '--mask-rate', '.2']
    with pytest.raises(ValueError, match='--mask-target'):
        run.main(base + ['--output', str(tmp_path / 'a'), '--mask', 'whole_feature_matched'])
    with pytest.raises(ValueError, match='--mask-target'):
        run.main(base + ['--output', str(tmp_path / 'b'), '--mask', 'whole_feature', '--mask-target', str(target)])
    assert not (tmp_path / 'a').exists() and not (tmp_path / 'b').exists()
    assert run.main(base + ['--output', str(tmp_path / 'c'), '--mask', 'whole_feature_matched', '--mask-target',
                            str(target), '--seeds', '0', '1']) == 0
    config = json.loads((tmp_path / 'c/config.json').read_text())
    assert config['mask_target']['sha256'] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert config['mask_target']['pi'] == {'G1|N2': .5, 'G2|N3': .5}
    saved = np.load(tmp_path / 'c/mask_whole_feature_matched_seed1.npy')
    from longitudinal_impute.data import load_csv
    assert np.array_equal(saved, make_mask(load_csv(data, schema), 'whole_feature_matched', .2, 1,
                                           json.loads(target.read_text()))[0])


# ---------- COVID checks (step 7) ----------

# Masks of steps 1-7 (rate 0.05, seeds 0-4), as recorded by the step 1 results manifest at tag
# evaluation-v1-results: make_mask must still draw exactly these.
STEP1_MASKS = {
    'whole_feature': {
        '0': '9ea04b56c640bb21d13e7f7516807a6f6106b62aec287270558e6039e1c9c25a',
        '1': '5643b7cf219fa5a574e4d495b116a0d404e6d15e3f6455ed376277d231edde4a',
        '2': '4a8e20c64e9e64f30bbbbd94d367fb059a2a61e7d483594010887fc8d4e6b356',
        '3': '5933da2450e9ca7d8d32e3d3819b1da7df3c6384fce5b6f7365e4c623eb292ca',
        '4': '3b6291af5e5a111c87256cdaa8518277615d2a558eff2c3304ad2fffe20433ec'
    },
    'random_cell': {
        '0': 'b1b423d6a4265974001c3b103f8ba33876bee19fd15dbe546a900df266bb925c',
        '1': '2a3216055856fea24a153e6966840c930ebd9abba5baa4b57a0686a1485ab8c4',
        '2': '032bff14e5506297fa88acc36fd11f5a119b46ff5dc7c227f166980fc35a104b',
        '3': 'a9df976eac6247c50687356713b7461a5a3270a9eab0ecd87f8c08e00ce68656',
        '4': '6ef6d0fdd0792a61c0bbda08f7f3689e58a0a1fdf5dfa8100ed78607d192b3d0'
    }
}


def covid():
    return t2.load_csv(t2.DATA, t2.SCHEMA)


def test_cut_points_are_read_from_step_5():
    cuts = mm.step5_cuts(ROOT / 'configs/evaluation/step5_tier2/tier2_results.md')
    data = covid()
    assert cuts == [17., 149.]
    assert tuple(cuts) == t2.n_strata(t2.row_facts(data), t2.native_cells(data)['eval'])[1]   # consistency only


def test_whole_feature_and_random_cell_masks_are_unchanged():
    data = covid()
    for kind, by_seed in STEP1_MASKS.items():
        for seed, expected in by_seed.items():
            assert digest(make_mask(data, kind, .05, int(seed))[0]) == expected


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
