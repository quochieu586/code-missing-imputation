import pandas as pd

df = pd.read_csv('artifacts/fused/dataset_0_fused_raw.csv')
arg_rows = df[df['location'] == 'Argentina'].sort_values('date')
print('Argentina rows:', len(arg_rows))
vars_cols = ['recombinant', '20A', '20B', '20C', '20E', 'Beta', 'Alpha', 'Gamma', 'Delta', 'Kappa', 'Epsilon', 'Eta', 'Iota', 'Lambda', 'Mu', 'Omicron', 'S:677']
for i, row in arg_rows.iterrows():
    var_sum = sum(row[c] for c in vars_cols if pd.notna(row[c]))
    print(f"date={row['date']}, total_seq={row['total_sequence']}, variant_sum={var_sum}, other={row['other']}, closure={var_sum + row['other'] - row['total_sequence']}")