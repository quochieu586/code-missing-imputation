"""Generate the audit figures from local harness evidence.

Usage (from a checkout of the PR branch, with its nextgen/src on PYTHONPATH):
  python docs/baseline_audit_2026-10-30/make_figures.py --harness <repo>/.harness

Reads only aggregate metrics and completed CSVs produced by earlier local runs;
writes PNGs next to this script. No row-level values are printed.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'


def masked_scores(harness):
    path = harness / 'micoda-statistical-evaluation/run-final/results.json'
    return json.loads(path.read_text(encoding='utf-8'))


def grouped_bars(ax, kinds, a17, a18, title, ylabel, log=False):
    x = np.arange(len(kinds))
    width = .36
    ax.bar(x - width / 2, a17, width, label='A17/R1', color='#1f77b4')
    ax.bar(x + width / 2, a18, width, label='A18/R2', color='#ff7f0e')
    ax.set_xticks(x, kinds)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale('log')
    ax.legend(frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness', type=Path, required=True)
    args = parser.parse_args()
    harness = args.harness.resolve()
    FIG.mkdir(exist_ok=True)
    data = masked_scores(harness)
    rows = data['masked']
    kinds = ['whole_feature', 'random_cell']
    labels = {'whole_feature': 'whole_feature 5%', 'random_cell': 'random_cell 5%'}
    means = {kind: {name: {} for name in ('A17', 'A18')} for kind in kinds}
    points = {kind: {name: {} for name in ('A17', 'A18')} for kind in kinds}
    for kind in kinds:
        for name in ('A17', 'A18'):
            selected = [r for r in rows if r['mask'] == kind]
            for key in ('mse_count', 'mse_proportion', 'w1_count_macro', 'w1_proportion_macro'):
                values = [r['scores'][name][key] for r in selected]
                means[kind][name][key] = float(np.mean(values))
                points[kind][name][key] = values

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, title, label, log in zip(
            axes, ('mse_count', 'mse_proportion'),
            ('MSE trên ô bị che — count (thang log)', 'MSE trên ô bị che — tỷ lệ count/N'),
            ('MSE count', 'MSE tỷ lệ'), (True, False)):
        grouped_bars(ax, [labels[k] for k in kinds],
                     [means[k]['A17'][key] for k in kinds],
                     [means[k]['A18'][key] for k in kinds], title, label, log)
        for i, kind in enumerate(kinds):
            for j, name in enumerate(('A17', 'A18')):
                offset = -.18 if j == 0 else .18
                ax.scatter([i + offset] * 2, points[kind][name][key], s=18, zorder=3,
                           color='black', alpha=.6)
    fig.suptitle('MICoDa A17/R1 vs A18/R2 — mask 5%, seeds 7 & 42 (chấm = từng seed)', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_mse.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, title, label in zip(
            axes, ('w1_count_macro', 'w1_proportion_macro'),
            ('Wasserstein-1 macro — count', 'Wasserstein-1 macro — tỷ lệ count/N'),
            ('W1 count (trung bình theo target)', 'W1 tỷ lệ')):
        grouped_bars(ax, [labels[k] for k in kinds],
                     [means[k]['A17'][key] for k in kinds],
                     [means[k]['A18'][key] for k in kinds], title, label)
        for i, kind in enumerate(kinds):
            for j, name in enumerate(('A17', 'A18')):
                offset = -.18 if j == 0 else .18
                ax.scatter([i + offset] * 2, points[kind][name][key], s=18, zorder=3,
                           color='black', alpha=.6)
    fig.suptitle('Phân phối các ô bị che: Wasserstein-1 thấp hơn = giống truth hơn', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_wasserstein.png', dpi=160)
    plt.close(fig)

    import sys
    sys.path.insert(0, str(harness.parent / 'nextgen/src'))
    from longitudinal_impute.data import load_csv
    root = HERE.parents[1]
    schema = root / 'nextgen/configs/baselines_covariants.toml'
    source = load_csv(root / 'data/covariants.csv', schema)
    missing = ~np.isfinite(source.counts)
    luts = {}
    for name, directory in (('A17', 'covid-a17-parity'), ('A18', 'a18-fitpredict-whole')):
        completed = load_csv(harness / 'baseline-crosscheck-update' / directory / 'micoda_completed.csv', schema)
        luts[name] = {
            'imputed': (completed.counts / source.totals[:, None])[missing],
            'other': (source.totals - completed.counts.sum(axis=1))[missing.any(axis=1)] / source.totals[missing.any(axis=1)],
        }

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.linspace(0, .25, 80)
    ax.hist(luts['A17']['imputed'], bins=bins, density=True, histtype='step', linewidth=1.6,
            label='A17/R1', color='#1f77b4')
    ax.hist(luts['A18']['imputed'], bins=bins, density=True, histtype='step', linewidth=1.6,
            label='A18/R2', color='#ff7f0e')
    ax.set_yscale('log')
    ax.set_xlabel('Giá trị điền theo tỷ lệ count/N')
    ax.set_ylabel('Mật độ (log)')
    ax.set_title('86.050 NaN tự nhiên được điền — phân phối tỷ lệ (native, không có truth)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_native_imputed.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, color in (('A17', '#1f77b4'), ('A18', '#ff7f0e')):
        values = np.sort(luts[name]['other'])
        ax.plot(values, np.linspace(0, 1, len(values)), label=name, color=color, linewidth=1.8)
    ax.set_xlabel('Tỷ lệ Other ngầm trên dòng có ô thiếu')
    ax.set_ylabel('Tỷ lệ dòng (ECDF)')
    ax.set_title('Other ngầm = (N − tổng 17 target sau điền)/N — 11.155 dòng')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_native_other.png', dpi=160)
    plt.close(fig)
    print('figures written to', FIG)


if __name__ == '__main__':
    main()
