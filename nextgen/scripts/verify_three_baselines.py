"""Run the audit suite against THIS checkout; retain logs, hashes and JUnit evidence.

Usage: python nextgen/scripts/verify_three_baselines.py --output <new-directory>
Requires the project's test dependencies. Never reads or writes production datasets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--schema', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(root / 'nextgen/src'))
    command = [sys.executable, '-W', 'error::RuntimeWarning', '-m', 'pytest',
               'nextgen/tests', '-q', '--basetemp', str(output / 'pytest-tmp'),
               '--junitxml', str(output / 'tests.xml')]
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    (output / 'pytest.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
    files = list((root / 'nextgen/src').rglob('*.py')) + list((root / 'nextgen/tests').rglob('*.py'))
    report = {'root': str(root), 'python': sys.executable, 'command': command,
              'exit_code': result.returncode,
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'status': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
              'sha256': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(files)}}
    if bool(args.data) != bool(args.schema):
        parser.error('--data and --schema must be supplied together')
    if args.data:
        sys.path.insert(0, str(root / 'nextgen/src'))
        import numpy as np
        from longitudinal_impute.data import load_csv, make_mask
        data = load_csv(args.data, args.schema)
        complete = np.isfinite(data.counts).all(axis=1)
        other = data.totals[complete] - data.counts[complete].sum(axis=1)
        records = []
        for kind in ('random_cell', 'whole_feature'):
            for seed in (7, 42):
                hidden, _ = make_mask(data, kind, .05, seed)
                selected = complete & hidden.any(axis=1)
                truth = np.where(hidden[selected], data.counts[selected], 0.).sum()
                visible = np.where(hidden[selected], 0., data.counts[selected]).sum(axis=1)
                budget = (data.totals[selected] - visible).sum()
                records.append({'kind': kind, 'seed': seed, 'complete_affected_rows': int(selected.sum()),
                                'hidden_truth_mass': float(truth), 'r1_budget': float(budget),
                                'budget_to_truth': float(budget / truth) if truth else None})
        report['r1_audit'] = {'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                              'complete_rows': int(complete.sum()),
                              'complete_rows_positive_other': int((other > 0).sum()),
                              'definition': 'Originally complete affected rows only; R1 budget, not a model run',
                              'masks': records}
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(result.stdout + result.stderr)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
