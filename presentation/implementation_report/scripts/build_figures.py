"""Create reproducible, editable pgfplots fragments from the frozen CSV source."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIG = ROOT / 'figures'
COLORS = {'hron': 'HronBlue', 'micoda': 'MicodaOrange', 'tsagris': 'TsagrisGreen'}
LABELS = {'hron': 'Hron-kNN', 'micoda': 'MICoDA-V2', 'tsagris': r'Tsagris-$\alpha$'}
SCENARIOS = ['whole_feature', 'random_cell', 'time_block', 'whole_row']
SCENARIO_LABELS = {'whole_feature': 'Feature', 'random_cell': 'Random', 'time_block': 'Time block', 'whole_row': 'Whole row'}

with (DATA / 'primary_metrics_source.csv').open(encoding='utf-8-sig', newline='') as stream:
    rows = list(csv.DictReader(stream))
methods = ['hron', 'micoda', 'tsagris']
by_key = {(r['method'], r['scenario']): r for r in rows}
assert all((m, s) in by_key for m in methods for s in SCENARIOS)

def f(row, key, scale=100):
    return float(row[key]) * scale

def error_point(row, method, y, key='ratio_mae', low='ratio_ci95_low', high='ratio_ci95_high'):
    x = f(row, key)
    e1, e2 = x - f(row, low), f(row, high) - x
    return f'({x:.10f},{y}) += ({e2:.10f},0) -= ({e1:.10f},0)'

def plot_series(scenario, y, key='ratio_mae', low='ratio_ci95_low', high='ratio_ci95_high'):
    out = []
    for y, m in zip([2, 1, 0], methods):
        r = by_key[m, scenario]
        mark = {'hron': 'square*', 'micoda': 'triangle*', 'tsagris': '*'}[m]
        out.append(rf'\addplot+[only marks,mark={mark},mark size=2.6pt,color={COLORS[m]},error bars/.cd,x dir=both,x explicit] coordinates {{ {error_point(r,m,y)} }};')
    return '\n'.join(out)

primary = by_key
parts = [
 r'\begin{tikzpicture}',
 r'\begin{axis}[width=0.82\textwidth,height=4.15cm,xmin=0,xmax=13,xlabel={MAE (điểm phần trăm c/N)},ytick={0,1,2},yticklabels={Tsagris-$\alpha$,MICoDA-V2,Hron-kNN},y dir=normal,axis x line*=bottom,axis y line*=left,grid=major,major grid style={draw=BKLight},tick label style={font=\small},label style={font=\small},clip=false]',
]
for m,y in [('hron',2),('micoda',1),('tsagris',0)]:
    r=by_key[m,'whole_feature']
    mark={'hron':'square*','micoda':'triangle*','tsagris':'*'}[m]
    parts.append(rf'\addplot+[only marks,mark={mark},mark size=3.5pt,mark options={{fill={COLORS[m]}}},color={COLORS[m]},error bars/.cd,x dir=both,x explicit] coordinates {{ {error_point(r,m,y)} }};')
parts += [r'\end{axis}',r'\end{tikzpicture}']
(FIG/'primary_plot.tex').write_text('\n'.join(parts),encoding='utf-8')

parts=[r'\begin{tikzpicture}',
       r'\begin{axis}[width=0.82\textwidth,height=4.8cm,xmin=0,xmax=12.5,ymin=-0.6,ymax=11.6,ytick={0,1,2,3,4,5,6,7,8,9,10,11},yticklabels={Feature / Hron,Feature / MICoDA,Feature / Tsagris,Random / Hron,Random / MICoDA,Random / Tsagris,Time block / Hron,Time block / MICoDA,Time block / Tsagris,Whole row / Hron,Whole row / MICoDA,Whole row / Tsagris},y dir=reverse,axis x line*=bottom,axis y line*=left,xlabel={MAE (pp)},xmajorgrids=true,major grid style={draw=BKLight},tick label style={font=\scriptsize},label style={font=\small},clip=false]']
y=0
for scenario in SCENARIOS:
    for m in methods:
        r=by_key[m,scenario]
        mark={'hron':'square*','micoda':'triangle*','tsagris':'*'}[m]
        parts.append(rf'\addplot+[only marks,mark={mark},mark size=3pt,color={COLORS[m]},error bars/.cd,x dir=both,x explicit] coordinates {{ {error_point(r,m,y)} }};')
        y+=1
parts += [r'\end{axis}',r'\end{tikzpicture}']
(FIG/'scenario_plot.tex').write_text('\n'.join(parts),encoding='utf-8')

with (DATA/'paired_location_differences_source.csv').open(encoding='utf-8-sig',newline='') as stream:
    paired=list(csv.DictReader(stream))
paired=[r for r in paired if r['comparison'].endswith('/whole_feature')]
contrast=[('hron-micoda/whole_feature',2,'Hron $-$ MICoDA'),
          ('hron-tsagris/whole_feature',1,'Hron $-$ Tsagris'),
          ('micoda-tsagris/whole_feature',0,'MICoDA $-$ Tsagris')]
paired_by={r['comparison']:r for r in paired}
parts=[r'\begin{tikzpicture}',
       r'\begin{axis}[width=0.82\textwidth,height=3.9cm,xmin=-0.4,xmax=4.4,xlabel={$\Delta$ MAE theo location (pp)},ytick={0,1,2},yticklabels={MICoDA $-$ Tsagris,Hron $-$ Tsagris,Hron $-$ MICoDA},y dir=normal,axis x line*=bottom,axis y line*=left,xmajorgrids=true,major grid style={draw=BKLight},tick label style={font=\small},label style={font=\small},extra x ticks={0},extra x tick style={grid=major,grid style={black,thick}}]']
for name,y,label in contrast:
    r=paired_by[name]
    x=float(r['location_macro_ratio_mae_difference'])*100
    lo=float(r['ci95_low'])*100
    hi=float(r['ci95_high'])*100
    if y == 2:
        parts.append(r'\draw[Muted,dashed,line width=.7pt] (axis cs:0,0)--(axis cs:0,2);')
    parts.append(rf'\addplot+[only marks,mark=*,mark size=3pt,mark options={{fill=BKTemplate}},color=BKTemplate,error bars/.cd,x dir=both,x explicit] coordinates {{ ({x:.10f},{y}) += ({hi-x:.10f},0) -= ({x-lo:.10f},0) }};')
parts += [r'\end{axis}',r'\end{tikzpicture}']
(FIG/'paired_plot.tex').write_text('\n'.join(parts),encoding='utf-8')

files = ['primary_metrics_source.csv', 'paired_location_differences_source.csv']
manifest = {
    'accepted_run': 'experiments/offline_baselines_2026-10-15_v2',
    'accepted_report': 'docs/benchmark_offline_2026-10-15/report_vi.md',
    'generated_by': 'scripts/build_figures.py',
    'plot_units': 'ratio metrics and differences multiplied by 100 to display percentage points',
    'inputs': {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in files},
    'outputs': {name: hashlib.sha256((FIG/name).read_bytes()).hexdigest()
                for name in ['primary_plot.tex','scenario_plot.tex','paired_plot.tex']},
    'primary_rows': len(rows),
    'paired_whole_feature_comparisons': len(paired),
    'uncertainty': 'Location-bootstrap CI95 from accepted run; fixed fits. No test rerun.',
}
(DATA/'figure_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Wrote three vector pgfplots using {len(rows)} accepted metric rows and {len(paired)} paired whole-feature comparisons.')
