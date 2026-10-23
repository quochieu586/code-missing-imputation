"""Editable Ticket03 panels from saved CSV/summaries only. No model execution."""
from pathlib import Path
import csv,json,hashlib,re,math
D=Path(__file__).resolve().parents[1]; R=D.parents[1]
H=R/'.harness/ticket-03-frames-21-30-2026-10-23'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=read(D/'data/primary_metrics_source.csv')
methods=['hron','micoda','tsagris']; colors=['HronBlue','MicodaOrange','TsagrisGreen']; marks=['square*','triangle*','*']
scenarios=['whole_feature','random_cell','time_block','whole_row']
labels=['Cả thành phần','Ô ngẫu nhiên','Khối thời gian','Cả hàng']
def axis(opts):return '\\begin{axis}['+opts+']\n'
def points(values,color,mark):return rf'\addplot+[only marks,color={color},mark={mark},mark options={{fill={color}}},mark size=2.4pt] coordinates {{ '+ ' '.join(f'({x},{y:.10f})' for x,y in values)+' };\n'
# Reuse corrected exports and compact scenario layout with the same exact endpoints.
by={(r['method'],r['scenario']):r for r in rows}
s='\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.45cm,vertical sep=1.4cm},scale only axis,width=4.2cm,height=1.15cm,xmin=0,xmax=13,ymin=-.5,ymax=2.5,ytick=\\empty,xlabel={MAE (pp)},xmajorgrids=true,major grid style={draw=BKLight},tick label style={font=\\scriptsize},label style={font=\\scriptsize},title style={font=\\small},axis x line*=bottom,axis y line*=left]\n'
for sc,label in zip(scenarios,labels):
    s+=rf'\nextgroupplot[title={{{label}}}]'+'\n'
    for m,c,k,y in zip(methods,colors,marks,[2,1,0]):
        r=by[m,sc]; x=float(r['ratio_mae'])*100;lo=float(r['ratio_ci95_low'])*100;hi=float(r['ratio_ci95_high'])*100
        s+=rf'\addplot+[only marks,color={c},mark={k},mark options={{fill={c}}},mark size=2.3pt,error bars/.cd,x dir=both,x explicit] coordinates {{ ({x:.10f},{y}) += ({hi-x:.10f},0) -= ({x-lo:.10f},0) }};'+'\n'
s+='\\end{groupplot}\n\\end{tikzpicture}\n'
# Metric/unit is stated once under the four panels, avoiding inter-row label collisions.
s=s.replace('xlabel={MAE (pp)},','')
(D/'figures/scenario_plot.tex').write_text(s,encoding='utf-8')
# Verify exported coordinates/endpoints, including duplicated primary points.
records=[]
pattern=r'\(([\d.\-]+),([\d.\-]+)\) \+= \(([\d.\-]+),0\) -= \(([\d.\-]+),0\)'
for file,expected in [('primary_plot.tex',[by[m,'whole_feature'] for m in methods]),('scenario_plot.tex',[by[m,sc] for sc in scenarios for m in methods])]:
    matches=re.findall(pattern,(D/'figures'/file).read_text(encoding='utf-8'))
    assert len(matches)==len(expected)
    for (x,y,plus,minus),r in zip(matches,expected):
        x,plus,minus=map(float,[x,plus,minus]); vals=[x,x-minus,x+plus];truth=[float(r[k])*100 for k in ['ratio_mae','ratio_ci95_low','ratio_ci95_high']]
        assert max(abs(a-b) for a,b in zip(vals,truth))<2e-9
        records.append({'figure':file,'method':r['method'],'scenario':r['scenario'],'exported_estimate_low_high_pp':vals,'source_estimate_low_high_pp':truth})
paired=[r for r in read(D/'data/paired_location_differences_source.csv') if r['comparison'].endswith('/whole_feature')]
matches=re.findall(pattern,(D/'figures/paired_plot.tex').read_text(encoding='utf-8'));assert len(matches)==3
for (x,y,plus,minus),r in zip(matches,paired):
    x,plus,minus=map(float,[x,plus,minus]);vals=[x,x-minus,x+plus];truth=[float(r[k])*100 for k in ['location_macro_ratio_mae_difference','ci95_low','ci95_high']]
    assert max(abs(a-b) for a,b in zip(vals,truth))<2e-9
    records.append({'figure':'paired_plot.tex','comparison':r['comparison'],'exported_estimate_low_high_pp':vals,'source_estimate_low_high_pp':truth})
(H/'ci_endpoint_comparison.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
# Match all CSV metrics to saved per-profile results and collect exact support/auxiliary metrics.
P=R/'experiments/synthetic_baselines_v2_2026-10-23/plots/test_metrics.csv'; syn=read(P);support=[];full=[]
for profile in ['easy','medium','hard']:
    summary=R/f'experiments/synthetic_{profile}_baselines_v2_2026-10-23/summary.json'
    for group in json.loads(summary.read_text(encoding='utf-8'))['results']:
        mask='MCAR' if 'mcar' in group['mask'] else 'MAR'
        for model in group['results']:
            row=next(r for r in syn if r['profile']==profile and r['mask']==mask and r['method']==model['method'])
            overall=model['test']['overall']
            assert int(row['hidden_cells'])==overall['n']
            assert sum(v['n'] for v in model['test']['by_stratum'].values())==overall['n']
            for key in ['count_mae','sample_ratio_mae','latent_p_mae','latent_p_rmse']:assert math.isclose(float(row[key]),overall[key],rel_tol=1e-12)
            for st,v in model['test']['by_stratum'].items():
                val=v['latent_p_mae']; csvval=row[st+'_latent_p_mae'];assert (val is None and not csvval) or math.isclose(float(csvval),val,rel_tol=1e-12)
                support.append({'profile':profile,'mask':mask,'method':model['method'],'stratum':st,**v,'source':str(summary.relative_to(R))})
            full.append({'profile':profile,'mask':mask,'method':model['method'],**overall,'parameters':model['parameters']})
(D/'data/ticket03_synthetic_support.json').write_text(json.dumps(support,indent=2),encoding='utf-8')
(D/'data/ticket03_synthetic_full_metrics.json').write_text(json.dumps(full,indent=2),encoding='utf-8')
# Four readable panels, discrete profiles, shared scales, no fabricated CI/repeated-seed lines.
s='\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.15cm,vertical sep=1.15cm},scale only axis,width=4.2cm,height=1.05cm,xmin=.6,xmax=3.4,ymin=0,ymax=8.5,xtick={1,2,3},xticklabels={easy,medium,hard},ytick={0,4,8},ymajorgrids=true,major grid style={draw=BKLight},tick label style={font=\\scriptsize},label style={font=\\scriptsize},title style={font=\\small},axis x line*=bottom,axis y line*=left]\n'
for metric,label in [('sample_ratio_mae',r'$X/N$ MAE (pp)'),('latent_p_mae',r'$p$ MAE (pp)')]:
    for mask in ['MCAR','MAR']:
        s+=rf'\nextgroupplot[title={{{mask}}},ylabel={{{label}}}]'+'\n'
        for method,color,mark in zip(['hron_knn','micoda_v2','tsagris_alpha'],colors,marks):
            vals=[(i,float(next(r for r in syn if r['profile']==p and r['mask']==mask and r['method']==method)[metric])*100) for i,p in enumerate(['easy','medium','hard'],1)]
            s+=points([(x+[-.10,0,.10][colors.index(color)],y) for x,y in vals],color,mark)
s+='\\end{groupplot}\n\\end{tikzpicture}\n';(D/'figures/ticket03_profiles.tex').write_text(s,encoding='utf-8')
# Strata: explicit fixed subset hard/MCAR, no data-dependent selection.
sub=[r for r in support if r['profile']=='hard' and r['mask']=='MCAR']
s='\\begin{tikzpicture}\n'+axis('width=10.7cm,height=4cm,xmin=.5,xmax=3.5,ymin=0,ymax=30,xtick={1,2,3},xticklabels={True zero,Sampling zero,Positive count},ylabel={Latent-$p$ MAE (pp)},ymajorgrids=true,major grid style={draw=BKLight},tick label style={font=\\small},label style={font=\\small},axis x line*=bottom,axis y line*=left')
for method,color,mark in zip(['hron_knn','micoda_v2','tsagris_alpha'],colors,marks):
    vals=[(i,r['latent_p_mae']*100) for i,st in enumerate(['true_zero','sampling_zero','positive_count'],1) for r in sub if r['method']==method and r['stratum']==st and r['n']>0]
    s+=points([(x+[-.10,0,.10][colors.index(color)],y) for x,y in vals],color,mark)
s+='\\end{axis}\n\\end{tikzpicture}\n';(D/'figures/ticket03_strata.tex').write_text(s,encoding='utf-8')
files=[P]+[R/f'experiments/synthetic_{p}_baselines_v2_2026-10-23/{name}.json' for p in ['easy','medium','hard'] for name in ['summary','run_manifest']]
for p in ['easy','medium','hard']:
    run=json.loads((R/f'experiments/synthetic_{p}_baselines_v2_2026-10-23/run_manifest.json').read_text(encoding='utf-8'))
    assert run['dataset_seed']==2026 and run['scale']=='standard' and run['metric_version']=='dual-truth-v2'
outputs=[D/'figures'/n for n in ['primary_plot.tex','scenario_plot.tex','paired_plot.tex','ticket03_profiles.tex','ticket03_strata.tex']]
manifest={'commands':['py presentation/implementation_report/scripts/build_figures.py','py presentation/implementation_report/scripts/build_ticket03.py'],'source_hashes':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'outputs':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},'ci_points_verified':len(records),'csv_rows_verified':len(syn),'stratum_subset':'hard / MCAR / hidden test cells','null_policy':'n=0 omitted, never replaced by zero','units':'all ratio plots multiplied by100 = percentage points','inference':'synthetic one seed2026/profile, descriptive only'}
(D/'data/ticket03_figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
base_manifest=json.loads((D/'data/figure_manifest.json').read_text(encoding='utf-8'))
base_manifest['outputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs[:3]}
base_manifest['postprocessed_by']='scripts/build_ticket03.py (scenario layout only, same data)'
(D/'data/figure_manifest.json').write_text(json.dumps(base_manifest,indent=2),encoding='utf-8')
print(json.dumps({'ci_points_verified':len(records),'synthetic_rows_verified':len(syn),'support_records':len(support)}))

