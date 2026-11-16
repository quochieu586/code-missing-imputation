"""Protocol and pipeline (step 8): config, unused seeds, sign test, failure policy, hypotheses, tier 2
against step 5, the run commands and the frozen hash.

    PYTHONPATH=src python -W error::RuntimeWarning -m pytest tests/evaluation/test_protocol.py
"""
import json
import os
from pathlib import Path

import pytest

from longitudinal_impute.evaluation import pipeline as ev, protocol as v1, tier1
from longitudinal_impute.evaluation.__main__ import main
from longitudinal_impute.evaluation.report import ROOT

CONFIGS = ROOT / 'configs/evaluation'
TEXT = (CONFIGS / 'eval_v1.toml').read_text(encoding='utf-8')


def write(tmp_path, text):
    path = tmp_path / 'c.toml'
    path.write_text(text, encoding='utf-8')
    return path


# ---------- config ----------

def test_the_frozen_config_reads_and_lists_nine_hypotheses():
    config = v1.read_config(CONFIGS / 'eval_v1.toml')
    assert [h['id'] for h in config['hypotheses']] == [f'H{k}' for k in range(1, 10)]
    assert config['masks']['seeds'] == list(range(200, 210)) and config['tier0']['max_failed_seeds'] == 2


@pytest.mark.parametrize('old,new,message', [
    ('rate = 0.05', 'rate = 0.05\nratio = 1', r'\[data\]: unknown keys \[\'ratio\'\]'),
    ('max_failed_seeds = 2', '', r'\[tier0\]: unknown keys \[\], missing keys \[\'max_failed_seeds\'\]'),
    ('version = "v1"', 'version = "v1"\nextra = 1', 'top level: unknown keys'),
    ('min_wins = 8\n\n[[hypotheses]]\nid = "H2"', 'min_wins = 8\nnote = "x"\n\n[[hypotheses]]\nid = "H2"',
     'hypothesis H1: unknown keys'),
    ('expect = "fail"', 'expect = "maybe"', 'expect pass / fail'),
])
def test_unknown_or_missing_keys_stop_before_any_side_effect(tmp_path, monkeypatch, old, new, message):
    assert TEXT.count(old) == 1
    path = write(tmp_path, TEXT.replace(old, new))
    with pytest.raises(ValueError, match=message):
        v1.read_config(path)
    with pytest.raises(ValueError):
        main(['masks', '--config', str(path)])                # manifest would go next to it
    assert not (tmp_path / 'masks_manifest.json').exists()


def test_seeds_used_by_earlier_steps_are_refused(tmp_path):
    (tmp_path / 'step3' / 'runs' / 'a').mkdir(parents=True)
    (tmp_path / 'step3' / 'runs' / 'a' / 'config.json').write_text(json.dumps({'seeds': [0, 201]}))
    (tmp_path / 'step6').mkdir()
    (tmp_path / 'step6' / 'x_manifest.json').write_text(json.dumps({'m': {'mask_sha256': {'208': 'h'}}}))
    patterns = ['step[1-7]/**/config.json', 'step[1-7]/*manifest*.json']
    assert v1.used_seeds(tmp_path, patterns) == {0: ['step3/runs/a/config.json'],
                                                 201: ['step3/runs/a/config.json'],
                                                 208: ['step6/x_manifest.json']}
    snapshot = tmp_path / 'used_seeds.json'
    snapshot.write_text(json.dumps(v1.seed_snapshot(tmp_path, patterns)))
    with pytest.raises(ValueError, match=r'already used: \[201, 208\]'):
        v1.check_unused(list(range(200, 210)), [snapshot])
    assert v1.check_unused([300], [snapshot]) == ['step3/runs/a/config.json', 'step6/x_manifest.json']


def test_the_shipped_seed_snapshot_lists_steps_1_to_7_and_refuses_their_seeds():
    snapshot = json.loads((CONFIGS / 'used_seeds.json').read_text(encoding='utf-8'))
    assert snapshot['seeds'] == [0, 1, 2, 3, 4, 109] and len(snapshot['files']) == 29
    assert len(v1.check_unused(list(range(200, 210)), [CONFIGS / 'used_seeds.json'])) == 29
    with pytest.raises(ValueError, match=r'already used: \[109\]'):
        v1.check_unused([109, 200], [CONFIGS / 'used_seeds.json'])


# ---------- statistics and decisions ----------

def test_exact_sign_test_by_hand():
    assert tier1.sign_test(10, 0) == pytest.approx(2 / 1024)
    assert tier1.sign_test(8, 2) == pytest.approx(2 * (1 + 10 + 45) / 1024)
    assert tier1.sign_test(5, 5) == 1.
    assert tier1.sign_test(0, 0) is None
    p = ev.pair_rows({'a': run([.1] * 7 + [.2] * 3), 'b': run([.2] * 7 + [.2] * 3)}, [('a', 'b')], SEEDS)[0]
    assert (p['wins'], p['ties'], p['sign_p']) == (7, 3, pytest.approx(2 / 128))   # ties dropped: 7 / 7


SEEDS = [str(s) for s in range(200, 210)]


def run(values, failed=()):
    return {s: {'eligible': s not in failed, 'status': 'failed' if s in failed else 'completed', 'code': '',
                'mae_prop': None if s in failed else v,
                'mse_prop': None if s in failed else v} for s, v in zip(SEEDS, values)}


def hypothesis(kind='pair_better', b=('b',), **extra):
    h = {'id': 'H', 'text': '', 'kind': kind, 'metric': 'mae_prop', 'a': 'a', 'b': list(b)}
    return {**h, **({'min_wins': 8} if kind == 'pair_better' else {'max_wins': 5}), **extra}


def judge(runs, h=None):
    return v1.judge_pairs(h or hypothesis(), runs, SEEDS, 2, tier1.pair)[0]


def test_hypotheses_confirmed_not_confirmed_and_insufficient():
    better = [.1] * 8 + [.3] * 2
    assert judge({'a': run(better), 'b': run([.2] * 10)}) == 'confirmed'               # 8 wins, mean < 0
    assert judge({'a': run([.1] * 7 + [.3] * 3), 'b': run([.2] * 10)}) == 'not_confirmed'
    assert judge({'a': run([.19] * 8 + [.9] * 2), 'b': run([.2] * 10)}) == 'not_confirmed'   # 8 wins, mean > 0
    assert judge({'a': run(better, SEEDS[:2]), 'b': run([.2] * 10)}) == 'not_confirmed'    # 2 failed: 6 wins
    assert judge({'a': run(better, SEEDS[:3]), 'b': run([.2] * 10)}) == 'insufficient'     # 3 failed > 2
    assert judge({'a': run(better), 'b': run([.2] * 10), 'c': run([.05] * 10)},
                 hypothesis(b=('b', 'c'))) == 'not_confirmed'
    assert judge({'a': run(better), 'b': run([.2] * 10), 'c': run([.2] * 10, SEEDS[:4])},
                 hypothesis(b=('b', 'c'))) == 'insufficient'
    not_better = hypothesis('pair_not_better')
    assert judge({'a': run([.1] * 5 + [.3] * 5), 'b': run([.2] * 10)}, not_better) == 'confirmed'
    assert judge({'a': run([.1] * 6 + [.3] * 4), 'b': run([.2] * 10)}, not_better) == 'not_confirmed'


def test_tier3_hypotheses_and_the_three_mask_rule():
    h = {'id': 'H', 'text': '', 'kind': 'tier3', 'use_case': 'u3', 'expect': 'pass'}
    assert v1.judge_tier3(h, 'pass', {'a': 0, 'b': 2}, 2) == 'confirmed'
    assert v1.judge_tier3(h, 'pass', {'a': 3}, 2) == 'insufficient'                        # a real model failed 3 / 10
    assert v1.judge_tier3(h, 'fail', {'a': 0}, 2) == 'not_confirmed'
    assert v1.judge_tier3({**h, 'expect': 'fail'}, 'fail', {'a': 0}, 2) == 'confirmed'
    assert v1.judge_tier3(h, 'insufficient', {'a': 0}, 2) == 'insufficient'
    assert v1.combine_masks({'x': 'pass', 'y': 'pass'}) == 'pass'
    assert v1.combine_masks({'x': 'pass', 'y': 'insufficient', 'z': 'fail'}) == 'fail'
    assert v1.combine_masks({'x': 'pass', 'y': 'insufficient'}) == 'insufficient'
    assert ev.failed_seeds({'a': run([.1] * 10, SEEDS[:3])}, 'a', SEEDS) == SEEDS[:3]


# ---------- tier 2 and the run commands ----------

def test_tier2_through_v1_equals_step5(tmp_path):
    roots = os.environ.get('EVALUATION_EXPLORATORY_RUNS')
    if not roots:
        pytest.skip('set EVALUATION_EXPLORATORY_RUNS to the directories of the seed 0-4 runs')
    ev.tier2.summarize(tmp_path, roots=roots.split(os.pathsep))
    for name in ev.STEP5_FILES:                           # manifest: without its provenance (no rerun here)
        reference = CONFIGS / 'step5_tier2' / name
        if name == 'tier2_manifest.json':
            new, old = (ev.without_provenance(json.loads(path.read_text(encoding='utf-8')))
                        for path in (tmp_path / name, reference))
            for record in (*new['models'].values(), *old['models'].values()):
                record.pop('rerun_identical', None)       # step 5 also checked a rerun; not run here
            assert new == old
        else:
            assert (tmp_path / name).read_bytes() == reference.read_bytes(), name


def test_manifest_comparison_ignores_only_the_provenance(tmp_path):
    reference = {'cells': {'eval': 3}, 'models': {'hron': {'csv_sha256': 'a', 'git_commit': 'c'}}}
    new = {'cells': {'eval': 3}, 'diagnostics_sha256': 'x', 'models': {'hron': {'csv_sha256': 'a', 'git_commit': 'c',
                                                                              'dir': 'somewhere/main_whole_feature'}}}
    (tmp_path / 'ref.json').write_text(json.dumps(reference))
    (tmp_path / 'new.json').write_text(json.dumps(new))
    assert ev.same_file('tier2_manifest.json', tmp_path / 'new.json', tmp_path / 'ref.json')
    new['models']['hron']['csv_sha256'] = 'b'
    (tmp_path / 'new.json').write_text(json.dumps(new))
    assert not ev.same_file('tier2_manifest.json', tmp_path / 'new.json', tmp_path / 'ref.json')


def test_run_commands_follow_the_config():
    config = v1.read_config(CONFIGS / 'eval_v1.toml')
    commands = ev.run_commands(config, Path('runs'))
    assert len(commands) == 9
    matched = [c for c in commands if 'whole_feature_matched' in c]
    assert all('--mask-target' in c for c in matched) and len(matched) == 3
    assert all(c[c.index('--seeds') + 1:c.index('--seeds') + 11] == SEEDS for c in commands)


def test_amended_config_still_matches_the_frozen_hash_and_any_other_edit_does_not(tmp_path):
    """eval_v1.toml after the restructure (paths moved, amendment block) is the frozen file once
    v1.frozen_bytes() undoes exactly those edits; masks_manifest.json keeps the frozen sha256."""
    frozen = json.loads((CONFIGS / 'masks_manifest.json').read_text(encoding='utf-8'))['config_sha256']
    assert v1.frozen_sha256(CONFIGS / 'eval_v1.toml') == frozen
    text = TEXT
    for old, new in (('min_wins = 8', 'min_wins = 7'), ('# --- end of amendment ---\n', '# --- end of amendment ---\n#\n')):
        changed = tmp_path / 'changed.toml'
        changed.write_text(text.replace(old, new, 1), encoding='utf-8')
        assert v1.frozen_sha256(changed) != frozen
    broken = tmp_path / 'broken.toml'
    broken.write_text(text.replace('# --- end of amendment ---\n', ''), encoding='utf-8')
    with pytest.raises(ValueError, match='malformed amendment'):
        v1.frozen_bytes(broken)
