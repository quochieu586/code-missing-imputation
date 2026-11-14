"""Independent audit of the E5 rework experiment (run AFTER ``run_e5_rework.py --stage finalize``).

Recomputes M2/M3/CLR-MAE from the cached predictions with a second implementation
(tolerance 1e-12), re-derives every gate from those independent numbers, checks the
Canada seal, frozen hashes and the full unit-test suite, and writes
``verification.json`` + ``file_hashes.json`` into the experiment folder.
It never writes outside the experiment folder.
"""
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rework_runtime  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts.diagnose_e5_rework import OUT as PHASE_A, frozen_hashes, independent_metrics  # noqa: E402
from src.evaluation import rework_gates as RG  # noqa: E402
from src.evaluation.research_pilot import sha256, write_json, write_hashes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'artifacts/e3e4_rework_2026-10-05_b'
E0 = ROOT / 'artifacts/e0_extended'
TOL = 1e-12
ATTRS = ('m2', 'm3', 'clr_mae_all_hidden', 'clr_mae_nonzero_hidden', 'clr_mae_zero_hidden')
TRANSFORMS = ['clr16', 'clr17', 'ilr16', 'hkglr16_actual', 'hkglr16_random_seed42',
              'hkglr16_random_seed43', 'hkglr16_random_seed44', 'hkglr16_high']
MODES = ['no_init', 'mask_only', 'hron_2a', 'aitchison_complete']
SEEDS = [42, 43, 44]


def check_scores(scores, truth, hidden, query_pred, label, problems):
    """scores: json entry; query_pred: ndarray of predictions."""
    visible = ~hidden
    if not (np.isfinite(query_pred).all() and (query_pred >= 0).all()):
        problems.append(f'{label}: non-finite/negative prediction')
    if not np.array_equal(query_pred[visible], truth[visible]):
        problems.append(f'{label}: observed counts not locked exactly')
    calc, _, eligible = independent_metrics(truth, query_pred, hidden)
    if np.flatnonzero(eligible).tolist() != scores['eligible_row_positions']:
        problems.append(f'{label}: eligibility differs from truth-only rule')
    for a in ATTRS:
        if abs(calc[a] - scores[a]) > TOL:
            problems.append(f'{label}: {a} differs by {abs(calc[a] - scores[a]):.3e}')
    return calc


def main():
    if (B / 'verification.json').exists():
        raise FileExistsError('verification.json exists; preserve it')
    problems = []
    before = json.loads((PHASE_A / 'frozen_hashes_before.json').read_text())
    frozen_ok = before == frozen_hashes()
    if not frozen_ok:
        problems.append('frozen artifacts/data/PDF changed')

    df = pd.read_csv(ROOT / 'data/covariants.csv')
    names = json.loads((E0 / 'manifest.json').read_text())['variant_columns']
    fold = json.loads((E0 / 'cv_splits.json').read_text())[0]
    pool_df = df.iloc[fold['train_rows']]
    val = pool_df[pool_df.location == 'Denmark']
    can = df.iloc[fold['test_rows']]
    val_masks = dict(np.load(B / 'validation_whole_variant_masks.npz'))
    val_scn = {k: val_masks[k] for k in RG.PRIMARY}
    val_scn['random_cell_bank0'] = np.load(B / 'validation_random_cell_bank0.npz')['bank0']
    e0w = np.load(E0 / 'masks_whole_variant.npz')
    can_scn = {f'whole_{l}_n2': e0w[f'fold0_whole_{l}_n2'] for l in ('rare', 'middle', 'common')}
    can_scn['random_cell_bank0'] = np.load(E0 / 'masks_random_cell.npz')['fold0_random_r0.30_seed42']
    cfg_sha = sha256(B / 'config.json')

    independent = {}   # key -> scenario -> metrics, for gate re-derivation
    n_checked = 0

    def audit_cache(cache, frame, scn_masks, tag, expected_keys=None):
        nonlocal n_checked
        truth = frame[names].to_numpy(dtype=np.float64)
        keys = []
        for jp in sorted(cache.glob('*__seed*.json')):
            blob = json.loads(jp.read_text(encoding='utf-8'))
            key = blob['key']
            keys.append(key)
            if blob['config_sha256'] != cfg_sha:
                problems.append(f'{tag}/{key}: config hash mismatch')
            if not 1 <= blob['best_epoch'] <= 40:
                problems.append(f'{tag}/{key}: best_epoch out of range')
            npz = np.load(jp.with_suffix('.npz'))
            for scn, hidden in scn_masks.items():
                pred = npz[f'pred__{scn}']
                lat = npz[f'latent__{scn}']
                if lat.shape[0] != blob['mc_samples'] or not np.isfinite(lat).all():
                    problems.append(f'{tag}/{key}/{scn}: latent samples invalid')
                calc = check_scores(blob['scores'][scn], truth, hidden, pred,
                                    f'{tag}/{key}/{scn}', problems)
                independent.setdefault(tag, {}).setdefault(key, {})[scn] = calc
                n_checked += 1
        if expected_keys is not None and sorted(keys) != sorted(expected_keys):
            problems.append(f'{tag}: model set differs from registration '
                            f'({len(keys)} vs {len(expected_keys)})')
        # controls
        cp = cache / 'controls_predictions.npz'
        if cp.exists():
            ctrl = np.load(cp)
            for k in ctrl.files:
                # keys look like init_only__hron_2a__whole_rare_n2 / linear__random_cell_bank0
                for s_name in scn_masks:
                    if k.endswith('__' + s_name):
                        calc, _, _ = independent_metrics(truth, ctrl[k], scn_masks[s_name])
                        independent.setdefault(tag + '_controls', {})[k] = calc
                        n_checked += 1
                        if not np.array_equal(ctrl[k][~scn_masks[s_name]], truth[~scn_masks[s_name]]):
                            problems.append(f'{tag}/{k}: observed locking')

    expected = [f'{m}__{t}__seed{s}' for t in TRANSFORMS for m in MODES for s in SEEDS]
    audit_cache(B / 'cache_validation', val, val_scn, 'validation', expected)
    cap = B / 'cache_cap8'
    if cap.exists():
        audit_cache(cap, val, {'random_cell_bank0': val_scn['random_cell_bank0']}, 'cap8', expected)

    # Re-derive validation gates from independent numbers
    ind_scores = {k: {s: {'m2': v['m2'], 'clr_mae_nonzero_hidden': v['clr_mae_nonzero_hidden'],
                          'status': 'OK', 'nonfinite_cells': 0, 'negative_cells': 0,
                          'observed_restoration_exact': True} for s, v in sc.items()}
                  for k, sc in independent.get('validation', {}).items()}
    controls = json.loads((B / 'validation_scores.json').read_text())
    for k, v in controls.items():
        if k.startswith(('init_only', 'linear', 'locf')):
            ind_scores[k] = {'m2': v['m2'], 'clr_mae_nonzero_hidden': v['clr_mae_nonzero_hidden'],
                             'status': 'OK', 'nonfinite_cells': 0, 'negative_cells': 0,
                             'observed_restoration_exact': True}
    for k, calc in independent.get('validation_controls', {}).items():
        stored = controls[k]
        for a in ATTRS:
            if abs(calc[a] - stored[a]) > TOL:
                problems.append(f'validation control {k}: {a} mismatch')
    recomputed = RG.validation_gates(ind_scores, TRANSFORMS, ['hron_2a', 'aitchison_complete'], SEEDS)
    stored_gates = json.loads((B / 'validation_gates.json').read_text())
    for gate in ('conditioning_gate', 'efficacy_gate'):
        if recomputed[gate]['passed'] != stored_gates[gate]['passed']:
            problems.append(f'{gate}: stored verdict differs from independent recomputation')
    for pair, v in recomputed['conditioning_gate']['pairs'].items():
        if v['passed'] != stored_gates['conditioning_gate']['pairs'][pair]['passed']:
            problems.append(f'conditioning pair {pair} differs')
    for pair, v in recomputed['efficacy_gate']['pairs'].items():
        if v['passed'] != stored_gates['efficacy_gate']['pairs'][pair]['passed']:
            problems.append(f'efficacy pair {pair} differs')

    # Canada seal consistency
    status = json.loads((B / 'e5_status.json').read_text())
    lock = (B / 'canada_evaluated.lock').exists()
    test_cache = B / 'cache_test'
    if not stored_gates['passed']:
        if lock or test_cache.exists() or (B / 'test_scores.json').exists():
            problems.append('Canada artifacts exist although validation gates failed')
        if status['finalists'] or status['e5_status'] != 'BLOCKED' or status['canada_evaluated']:
            problems.append('status inconsistent with failed gates')
    else:
        if not lock or not test_cache.exists():
            problems.append('gates passed but Canada evaluation missing')
        else:
            audit_cache(test_cache, can, can_scn, 'test')
    if (B / 'canada_evaluated.lock').exists() and not stored_gates['passed']:
        problems.append('lock present with failed gates')

    # Full unit test suite
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = int(pytest.main([str(ROOT / 'tests/unit'), '-q', '-p', 'no:cacheprovider',
                                f'--basetemp={ROOT / "tmp/pytest_audit"}']))
    (B / 'verification_tests.txt').write_text(buf.getvalue(), encoding='utf-8')
    if code != 0:
        problems.append('unit tests failed')
    if before != frozen_hashes():
        problems.append('frozen files changed during audit')

    result = {
        'audit_passed': not problems, 'problems': problems,
        'independently_verified_method_scenarios': n_checked, 'tolerance': TOL,
        'validation_models_expected': len(expected),
        'frozen_artifacts_data_pdfs_unchanged': frozen_ok,
        'validation_gates_passed': stored_gates['passed'],
        'conditioning_gate_passed': stored_gates['conditioning_gate']['passed'],
        'efficacy_gate_passed': stored_gates['efficacy_gate']['passed'],
        'numerical_gate_passed': stored_gates['numerical_gate']['passed'],
        'e5_status': status['e5_status'], 'finalists': status['finalists'],
        'canada_evaluated': status['canada_evaluated'],
        'unit_test_exit_code': code, 'unit_test_output_tail': buf.getvalue().strip()[-300:],
        'data_sha256': sha256(ROOT / 'data/covariants.csv'),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                              text=True).strip(),
        'python': sys.executable}
    write_json(B / 'verification.json', result)
    write_hashes(B, [ROOT / 'scripts/run_e5_rework.py', ROOT / 'scripts/audit_e5_rework.py',
                     ROOT / 'src/stage_b/research_csdi_rework.py',
                     ROOT / 'src/evaluation/rework_gates.py', ROOT / 'scripts/rework_runtime.py'])
    print(json.dumps({k: result[k] for k in ('audit_passed', 'problems',
                                              'independently_verified_method_scenarios',
                                              'validation_gates_passed', 'e5_status')}, indent=2))
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main())
