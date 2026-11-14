"""Re-run checklist E0.1 without changing frozen research artifacts."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from audit_e0_extended import ROOT, DATA, sha256, variant_columns

OUT = ROOT / "artifacts" / "e0_checklist_2026-10-05"


def load_data():
    df = pd.read_csv(DATA)
    variants = variant_columns(df)
    if len(variants) != 17:
        raise ValueError(f"Expected 17 variants, got {variants}")
    values = df[variants].to_numpy(float)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("Variant counts must be nonnegative and finite when observed")
    if df['location'].isna().any() or df[['location', 'date']].duplicated().any():
        raise ValueError("Missing locations or duplicate location/date rows")
    return df, variants


def main():
    df, variants = load_data()
    complete = df[variants].notna().all(axis=1)
    sums = df[variants].sum(axis=1)
    total = df['total_sequence']
    valid_total = total.notna() & np.isfinite(total) & (total > 0)
    rows = df.loc[complete, ['location', 'date', 'total_sequence']].copy()
    rows.insert(0, 'row_id', rows.index)
    rows['sum_17'] = sums[complete]
    rows['valid_total'] = valid_total[complete]
    rows['ratio'] = (sums / total.where(valid_total))[complete]
    rows['in_tolerance'] = rows['ratio'].between(0.95, 1.05)
    if rows.empty or not (rows['sum_17'] > 0).any():
        raise ValueError("No complete positive compositions to audit")

    def stats(subset):
        ratios = subset['ratio'].dropna()
        def number(value):
            return float(value) if np.isfinite(value) else None
        return {
            'n': len(subset), 'n_valid_ratios': len(ratios),
            'mean': number(ratios.mean()), 'std': number(ratios.std()),
            'min': number(ratios.min()), 'max': number(ratios.max()),
            'n_in_tolerance': int(subset['in_tolerance'].sum()),
            'fraction_in_tolerance': float(subset['in_tolerance'].mean()),
        }

    all_stats = stats(rows)
    positive_stats = stats(rows.loc[rows['sum_17'] > 0])
    # Revised plan §1.1 is authoritative: >=95% of ALL complete rows.
    # The checklist's positive-only K-weak branch does not authorize JSD.
    confirmed = all_stats['fraction_in_tolerance'] >= 0.95 and rows['valid_total'].all()
    manifest = {
        'task': 'E0.1', 'audit_date': '2026-10-05',
        'input_path': str(DATA.relative_to(ROOT)), 'input_sha256': sha256(DATA),
        'variant_columns': variants, 'n_rows': len(df),
        'n_complete_rows': len(rows), 'n_positive_complete': positive_stats['n'],
        'n_zero_sum_complete': int((rows['sum_17'] == 0).sum()),
        'n_invalid_total_complete': int((~rows['valid_total']).sum()),
        'ratio_all_complete': all_stats, 'ratio_positive_complete': positive_stats,
        'protocol': 'K' if confirmed else 'U', 'jsd_applicable': bool(confirmed),
        'closure_gate': 'PASS' if confirmed else 'FAIL', 'task_status': 'COMPLETE',
        'decision_rule': 'Revised §1.1: >=95% of all complete rows within [0.95,1.05], valid totals required',
        'positive_only_checklist_gate': 'PASS' if positive_stats['fraction_in_tolerance'] >= .95 else ('CAUTION' if positive_stats['fraction_in_tolerance'] >= .8 else 'FAIL'),
        'decision': 'Protocol K: closed composition confirmed' if confirmed else 'Protocol U: closure not established; skip original JSD-complete and known-total restoration',
        'e5_status': 'BLOCKED', 'finalists': [],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT / 'closure_rows.csv', index=False)
    manifest['closure_rows_sha256'] = sha256(OUT / 'closure_rows.csv')
    (OUT / 'data_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
