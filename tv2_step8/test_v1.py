"""Step 8 tests: config, unused seeds, sign test, failure policy, hypotheses, tier 2 against step 5.

    PYTHONPATH=nextgen/src python -W error::RuntimeWarning -m pytest tv2_step8/test_v1.py
"""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v1  # noqa: E402

spec = importlib.util.spec_from_file_location('evaluate_v1', HERE / 'evaluate_v1.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
TEXT = (HERE / 'eval_v1.toml').read_text(encoding='utf-8')


def write(tmp_path, text):
    path = tmp_path / 'c.toml'
    path.write_text(text, encoding='utf-8')
    return path


# ---------- config ----------

def test_the_frozen_config_reads_and_lists_nine_hypotheses():
    config = v1.read_config(HERE / 'eval_v1.toml')
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
    monkeypatch.setattr(ev, 'CONFIG', path)
    monkeypatch.setattr(ev, 'MASKS_MANIFEST', tmp_path / 'masks_manifest.json')
    with pytest.raises(ValueError):
        ev.main(['masks'])
    assert not (tmp_path / 'masks_manifest.json').exists()


def test_seeds_used_by_earlier_steps_are_refused(tmp_path):
    (tmp_path / 'tv2_step3' / 'runs' / 'a').mkdir(parents=True)
    (tmp_path / 'tv2_step3' / 'runs' / 'a' / 'config.json').write_text(json.dumps({'seeds': [0, 201]}))
    (tmp_path / 'tv2_step6').mkdir()
    (tmp_path / 'tv2_step6' / 'x_manifest.json').write_text(json.dumps({'m': {'mask_sha256': {'208': 'h'}}}))
    patterns = ['tv2_step[1-7]/**/config.json', 'tv2_step[1-7]/*manifest*.json']
    assert v1.used_seeds(tmp_path, patterns) == {0: ['tv2_step3/runs/a/config.json'],
                                                 201: ['tv2_step3/runs/a/config.json'],
                                                 208: ['tv2_step6/x_manifest.json']}
    with pytest.raises(ValueError, match=r'already used: \[201, 208\]'):
        v1.check_unused(list(range(200, 210)), tmp_path, patterns)
    assert v1.check_unused([300], tmp_path, patterns) == ['tv2_step3/runs/a/config.json', 'tv2_step6/x_manifest.json']


# ---------- statistics and decisions ----------

def test_exact_sign_test_by_hand():
    assert v1.sign_test(10, 0) == pytest.approx(2 / 1024)
    assert v1.sign_test(8, 2) == pytest.approx(2 * (1 + 10 + 45) / 1024)
    assert v1.sign_test(5, 5) == 1.
    assert v1.sign_test(0, 0) is None
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
    return v1.judge_pairs(h or hypothesis(), runs, SEEDS, 2, ev.tier01.pair)[0]


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
    assert v1.failed_seeds({'a': run([.1] * 10, SEEDS[:3])}, 'a', SEEDS) == SEEDS[:3]


# ---------- tier 2 and the run commands ----------

def test_tier2_through_v1_equals_step5(tmp_path):
    if not (ev.REPO / 'tv2_step2/runs/main_whole_feature/hron_completed.csv').exists():
        pytest.skip('step 2 run directories are not available')
    ev.tier2.summarize(tmp_path)
    for name in ev.STEP5_FILES[:4]:                       # the manifest also records the rerun check
        assert (tmp_path / name).read_bytes() == (ev.REPO / 'tv2_step5' / name).read_bytes(), name


def test_run_commands_follow_the_config():
    config = v1.read_config(HERE / 'eval_v1.toml')
    commands = ev.run_commands(config)
    assert len(commands) == 9
    matched = [c for c in commands if 'whole_feature_matched' in c]
    assert all('--mask-target' in c for c in matched) and len(matched) == 3
    assert all(c[c.index('--seeds') + 1:c.index('--seeds') + 11] == SEEDS for c in commands)
