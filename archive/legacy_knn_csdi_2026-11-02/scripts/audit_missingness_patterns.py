"""Re-run checklist E0.2; zeros remain observed counts, only NaN is missing."""
import json

from audit_composition_closure import OUT, load_data
from audit_e0_extended import ROOT, DATA, sha256


def main():
    df, variants = load_data()
    records = []
    for location, group in df.groupby('location', sort=True):
        observed = group[variants].notna()
        n_patterns = len(observed.drop_duplicates())
        records.append({
            'location': location, 'n_rows': len(group),
            'date_start': str(group['date'].min()), 'date_end': str(group['date'].max()),
            'always_missing': json.dumps([v for v in variants if not observed[v].any()]),
            'always_observed': json.dumps([v for v in variants if observed[v].all()]),
            'sometimes_missing': json.dumps([v for v in variants if observed[v].any() and not observed[v].all()]),
            'n_unique_patterns': n_patterns, 'is_constant_pattern': n_patterns == 1,
            'missing_cells': int((~observed).to_numpy().sum()),
        })
    import pandas as pd
    table = pd.DataFrame(records)
    fraction = float(table['is_constant_pattern'].mean())
    gate = 'PASS' if fraction > .9 else ('INVESTIGATE' if fraction >= .5 else 'RETHINK')
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / 'missingness_patterns.csv', index=False)
    summary = {
        'task': 'E0.2', 'audit_date': '2026-10-05', 'task_status': 'COMPLETE',
        'input_path': str(DATA.relative_to(ROOT)), 'input_sha256': sha256(DATA),
        'variant_columns': variants, 'n_rows': len(df), 'n_locations': len(table),
        'locations_with_constant_pattern': int(table['is_constant_pattern'].sum()),
        'fraction_constant': fraction,
        'mean_unique_patterns': float(table['n_unique_patterns'].mean()),
        'missing_cells': int(table['missing_cells'].sum()),
        'gate': gate, 'primary_scenario': 'whole_variant' if gate == 'PASS' else 'requires_review',
        'secondary_scenario': 'random_cell',
        'mechanism_limit': 'Observed location-variant structure; external assay/reporting cause is unresolved. No MCAR/MAR/MNAR causal claim.',
        'patterns_sha256': sha256(OUT / 'missingness_patterns.csv'),
        'e5_status': 'BLOCKED', 'finalists': [],
    }
    (OUT / 'missingness_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
