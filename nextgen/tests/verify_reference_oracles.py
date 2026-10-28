"""Reproducible base-R comparisons; no downloads, package installs or cleanup.

Run with --workdir NEW_DIRECTORY --author-source pinned MICoDa_Aux.R.
Requires base Rscript on PATH (or --rscript absolute executable path).
This verifies source/equation parity, not every paper table/benchmark.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from longitudinal_impute.baselines.hron import HronKNN
from longitudinal_impute.baselines.micoda import MICoDa
from longitudinal_impute.baselines.tsagris import TsagrisKNN
from longitudinal_impute.baselines.regression import lts


def verify(workdir, author_source, rscript='Rscript'):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=False)
    fixtures = workdir / 'fixtures'; fixtures.mkdir()
    positive = np.random.default_rng(11).dirichlet(np.ones(4) * 3, size=26)
    mic = positive.copy(); mic[0, [0, 1]] = np.nan; mic[1, [0, 3]] = np.nan; mic[2, [1, 2]] = np.nan
    np.savetxt(fixtures / 'micoda_input.csv', np.column_stack([mic, np.linspace(-1, 1, 26)]),
               delimiter=',', header='a,b,c,d,covariate', comments='')
    hron = positive.copy(); hron[0, [0, 1]] = np.nan; hron[1, 3] = np.nan; hron[2, 2] = np.nan
    np.savetxt(fixtures / 'hron_input.csv', hron, delimiter=',', header='a,b,c,d', comments='')
    query = positive[:3].copy(); query[0, [0, 1]] = np.nan; query[1, [1, 3]] = np.nan; query[2, [0, 2]] = np.nan
    for name, value in [('tsagris_donors', positive), ('tsagris_query', query)]:
        np.savetxt(fixtures / (name + '.csv'), value, delimiter=',', header='a,b,c,d', comments='')
    out = workdir / 'r_output'
    oracle = Path(__file__).parent / 'fixtures/reference_oracles.R'
    process = subprocess.run([str(rscript), str(oracle), str(fixtures), str(author_source), str(out)],
                             capture_output=True, text=True)
    (workdir / 'r_execution.txt').write_text(process.stdout + process.stderr, encoding='utf-8')
    if process.returncode:
        raise RuntimeError('R oracle failed: see ' + str(workdir / 'r_execution.txt'))
    checks = []
    def compare(name, predicted):
        expected = np.genfromtxt(out / (name + '.csv'), delimiter=',', skip_header=1).reshape(predicted.shape)
        error = float(np.abs(predicted - expected).max())
        checks.append({'method': name, 'max_abs_difference': error, 'tolerance': 1e-9, 'pass': error < 1e-9})
    cov = np.linspace(-1, 1, 26)[:, None]
    for version in ['v1', 'v2']:
        compare('micoda_' + version, MICoDa(version=version, reference=3, semantics='author_code',
                                           max_iter=5, fixed_iterations=True).impute(mic, cov))
    compare('hron_knn', HronKNN(3, ties='index').fit(hron).predict(hron))
    for alpha in [-.5, 0., .5, 1.]:
        name = 'tsagris_' + str(int(alpha) if alpha in (0, 1) else alpha)
        compare(name, TsagrisKNN(3, alpha=alpha).fit(positive).predict(query))
    design = np.column_stack([np.ones(7), np.arange(7.)]); y = 2 + 3 * np.arange(7.); y[-1] += 100
    compare('lts', lts(design, y, h=5)[0][None, :])
    (workdir / 'comparison.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    if not all(c['pass'] for c in checks):
        raise AssertionError('Oracle mismatch: ' + repr(checks))
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', required=True, type=Path)
    parser.add_argument('--author-source', required=True, type=Path)
    parser.add_argument('--rscript', default='Rscript')
    args = parser.parse_args()
    print(json.dumps(verify(args.workdir, args.author_source, args.rscript), indent=2))
