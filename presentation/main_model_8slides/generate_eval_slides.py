from pathlib import Path
import csv,json
R=Path(__file__).resolve().parent
S=R.parents[1]/'outputs/eval_meta4_20261120_seed300_309'
def rows(p): return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
a=rows(S/'report/tier1_summary.csv'); b=rows(S/'report/tier2_summary.csv'); z=rows(S/'report/tier0_status.csv')
models=['hron','tsagris','micoda','same_day_mean','main','main_nometa']
names=['Hron','Tsagris','MICoDA A18','Same-day mean','Main','No metadata']
masks=['whole_feature','random_cell','whole_feature_matched']
D=R/'figures/data';D.mkdir(exist_ok=True)
manifest={}
def slide(n,title,intro,body,finding):
 p=R/f'slides/{n}.tex'
 p.write_text('\\begin{frame}{'+title+' --- exploratory}\n\\fontsize{9}{11}\\selectfont\n'+intro+'\n\\par\\medskip\n'+body+'\n\\par\\medskip\n\\textbf{'+finding+'}\n\\end{frame}\n',encoding='utf-8')
def table(head,rs):
 return '\\begin{center}\\begin{tabular}{'+'l'+'r'*(len(head)-1)+'}\\hline\n'+' & '.join(head)+r' \\\hline'+'\n'+'\n'.join(' & '.join(r)+r' \\' for r in rs)+'\n\\hline\\end{tabular}\\end{center}'
slide('09_eval_tier0','Tier 0 checks the saved runs','The checks cover six models, three masks, and seeds 300--309.',table(['The check','The result'],[['The runs completed.','180 of 180'],['The coverage is complete.','100\\%'],['The models keep visible cells.','All runs'],['The checks accept the outputs.','All runs']])+r'We hide 5\% of values. Whole-feature masks hide a feature; random-cell masks hide individual cells. Matched masks hide whole country--variant series, weighted toward the continent and total-count strata of truly missing, model-decided cells.\par\medskip Main uses continent, median age, income classification, and group. No metadata uses the same model without these fields. Hron, Tsagris, and MICoDA A18 are comparison methods. Same-day mean uses the total-weighted mean fraction for that variant in the same continent within $\pm28$ days; sparse windows widen, then fall back to the global mean.\par\medskip The seeds were seen during development. These results are not confirmatory.','All 180 runs pass the output checks.')
rs=[]
for m,n in zip(models,names):
 rr=[next(x for x in a if x['model']==m and x['mask']==k) for k in masks]
 rs.append([n]+[f"{float(x['mae_prop_mean']):.4f}" for x in rr])
tier1_display=[r[:] for r in rs]
slide('10_eval_tier1','Tier 1 measures hidden-value error','The hidden values provide known answers for the error test.',r'Mean absolute error (MAE) is the mean absolute error in the count fraction $c/N$ on cells decided by the model. Here, $c$ is the feature count and $N$ is the total count. Lower values are better.'+table(['The model','Whole feature','Random cell','Matched'],rs)+r'The table shows means over ten seeds. Mean squared error (MSE) is the mean squared error in the same fraction on the same cells.\par\medskip A paired sign test counts which method has lower error for each seed and tests equal chances. The metadata effect is not stable across seeds (MAE sign-test $p=0.754$ on each mask).','Same-day mean has lower error than Main on both whole-feature masks.')
p=R/'slides/10_eval_tier1.tex'
p.write_text(p.read_text(encoding='utf-8').replace(r'\fontsize{9}{11}',r'\fontsize{8.5}{10.3}'),encoding='utf-8')
# Retain source values without recomputing the evaluation.
with (D/'eval_mae.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['x','mean','sd','mask','model'])
 for i,m in enumerate(models):
  for j,k in enumerate(masks):
   x=next(v for v in a if v['model']==m and v['mask']==k)
   w.writerow([i+1+(j-1)*.22,x['mae_prop_mean'],x['mae_prop_sd'],k,m])
for j,k in enumerate(masks):
 with (D/f'eval_mae_{j}.dat').open('w') as f:
  f.write('x mean sd\n')
  for i,m in enumerate(models):
   x=next(v for v in a if v['model']==m and v['mask']==k)
   f.write(f"{i+1+(j-1)*.22} {x['mae_prop_mean']} {x['mae_prop_sd']}\n")
fig=r'''\documentclass[tikz,border=5pt]{standalone}
\input{figure_style.tex}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[width=12cm,height=6cm,axis lines=left,ymin=0,ymax=.23,xmin=.5,xmax=6.5,
 ylabel={Mean absolute error},xtick={1,2,3,4,5,6},xticklabels={Hron,Tsagris,\shortstack{MICoDA\\A18},\shortstack{Same-day\\mean},Main,\shortstack{No\\metadata}},
 tick label style={font=\figsmall},label style={font=\figfont},legend style={font=\figsmall,draw=none,at={(.5,1.03)},anchor=south,legend columns=3},yticklabel={$\mathsf{\pgfmathprintnumber{\tick}}$},yticklabel style={font=\figsmall},ymajorgrids,grid style={draw=Rule}]
'''
for j,(col,label) in enumerate(zip(['Observed','Teal','Orange'],['Whole feature','Random cell','Matched'])):
 fig+=r'\addplot[only marks,mark=*,color='+col+r',error bars/.cd,y dir=both,y explicit] table[x=x,y=mean,y error=sd] {data/eval_mae_'+str(j)+'.dat};\n'+r'\addlegendentry{'+label+'}\n'
fig+='\\end{axis}\n\\end{tikzpicture}\n\\end{document}\n'
(R/'figures/eval_mae.tex').write_text(fig)
slide('11_eval_tier1_plot','Tier 1 compares seed variation','The points show error means, and the bars show one standard deviation over ten seeds.',r'\SlideFigure[.97\linewidth]{eval_mae}','Main has lower mean error than Hron, Tsagris, and MICoDA on all three masks.')
rs=[[n,f"{float(x['local_d1_equal']):+.3f}",f"{float(x['local_d2_equal']):.4f}",x['d3_cells'],f"{float(x['d3_mass']):.3f}"] for m,n in zip(models,names) for x in b if x['model']==m]
slide('12_eval_tier2','Tier 2 describes truly missing cells','These diagnostics describe filled cells; they do not measure true error.',r'D1 subtracts the reference zero fraction from the filled zero fraction. D2 measures the Wasserstein distance between positive count fractions: it measures how far their distributions move. The local reference uses nearby observed data. The table gives equal weight to each variant (target feature).\par\medskip D3 counts positive fills outside the time range with observed data for a feature; its mass sums their count fractions.'+table(['The model','D1','D2','D3 cells','D3 mass'],rs),'Main has a small zero-fraction gap, but its out-of-range mass exceeds every comparison method except No metadata.')
# Exact displayed numbers and their sources are recorded for review.
manifest={'tier0_source':'report/tier0_status.csv','tier1_source':'report/tier1_summary.csv','tier1_display':tier1_display,'tier2_source':'report/tier2_summary.csv','tier2_display':rs,'tier3_source':'report_tier3/verdicts.csv','plot_source':'report/tier1_summary.csv: mae_prop_mean, mae_prop_sd','sign_source':'report/tier1_pairs.csv: main vs main_nometa, mae_prop, sign_p','plot_exact_values':list(csv.DictReader((D/'eval_mae.csv').open()))}
(R/'EVAL_SLIDES.md').write_text('Run `python presentation/main_model_8slides/generate_eval_slides.py` from the repository root. This standard-library script copies reported values; it does not rescore predictions.\n\nTier 1 table and plot use tier1_summary.csv. Tier 2 table uses tier2_summary.csv. Tier 0 uses tier0_status.csv. Tier 3 uses verdicts.csv and frozen thresholds stated in REPORT_TIER3_VI.md.\n\n'+json.dumps(manifest,indent=2))
# Tier 3: copy saved per-seed results, never rescore predictions.
rates=rows(S/'report_tier3/rates.csv')
ranks=rows(S/'report_tier3/rank_agreement_regret.csv')
keys=['u1_under','u1_over','u2','u3']
def values(key,mask):
 source=ranks if key=='u2' else rates
 return sorted([x for x in source if x['mask']==mask and x['case']==key and (key=='u3' or x['reference']=='local')],key=lambda x:int(x['seed']))
means={key:[sum(float(x['gap' if key=='u3' else 'rate']) for x in values(key,m))/10 for m in masks] for key in keys}
summary=[]
for key,label in [('u1_under','U1: D1 under-fill agreement'),('u1_over','U1: D1 over-fill agreement'),('u2','U2: D2 agreement')]:
 summary.append([label]+[f'{v:.2f}' for v in means[key]])
summary += [['U2: regret at most 0.005','7/10','7/10','8/10'],['U3: gap at least 0.2','10/10','10/10','10/10']]
slide('13_eval_tier3','Tier 3 checks warnings and selection','U1 and U2 fail; U3 passes on every mask. The table reports local-reference means over ten seeds.',
 table(['The test','Whole feature','Random cell','Matched'],summary)+
 r'U1 checks each real-model pair: does the D1 difference agree with the true MAE difference? Higher D1 should warn of under-fill on true-positive cells; lower D1 should warn of over-fill on true-zero cells. Each mean must reach 0.8, with no seed below 0.5.\par\medskip U2 compares D2-distance rankings with true MAE rankings. Mean agreement exceeds 0.8, but regret meets 0.005 in only 7--8 seeds; at least 9 of 10 are required. Regret is selected-model MAE minus the lowest MAE.\par\medskip U3 compares true-zero rates among positive fills: D3-flagged cells have a higher rate than unflagged cells. The gap reaches 0.2 in all ten seeds.\par\medskip U4 checks mean fill, zero fill, and a marginal sampler; it is descriptive only.',
 'D3 reliably flags out-of-range fills here; D1 and D2 are not reliable enough for model selection.')
p=R/'slides/13_eval_tier3.tex';p.write_text(p.read_text().replace(r'\fontsize{9}{11}',r'\fontsize{7.6}{9.2}'))
fig=r'''\documentclass[tikz,border=5pt]{standalone}
\input{figure_style.tex}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}\begin{tikzpicture}
'''
for panel,(key,title) in enumerate(zip(keys,['U1: under-fill agreement','U1: over-fill agreement','U2: D2 agreement','U3: true-zero rate gap'])):
 threshold=.2 if key=='u3' else .8
 fig+=r'\begin{axis}[width=6cm,height=3.35cm,at={('+str((panel%2)*6.3)+'cm,'+str(-(panel//2)*3.9)+r'cm)},anchor=north west,title={'+title+r'},title style={font=\figfont},xmin=.5,xmax=3.5,ymin=0,ymax=1.04,xtick={1,2,3},xticklabels={Whole,Random,Matched},tick label style={font=\figsmall},yticklabel={$\mathsf{\pgfmathprintnumber{\tick}}$},yticklabel style={font=\figsmall},axis lines=left,ymajorgrids,grid style={draw=Rule}]'+'\n'
 fig+=r'\addplot[Muted,dashed,domain=.5:3.5] {'+str(threshold)+'};\n'
 for j,(m,col) in enumerate(zip(masks,['Observed','Teal','Orange'])):
  got=values(key,m)
  path=D/f'eval_tier3_{key}_{j}.dat'
  path.write_text('x value\n'+'\n'.join(f"{j+1+(i-4.5)*.035} {x['gap' if key=='u3' else 'rate']}" for i,x in enumerate(got)))
  fig+=r'\addplot[only marks,mark=*,mark size=1.8pt,draw='+col+',fill='+col+r'] table[x=x,y=value] {data/'+path.name+'};\n'
 fig+=r'\end{axis}'+'\n'
fig+=r'\end{tikzpicture}\end{document}'
(R/'figures/eval_tier3.tex').write_text(fig)
slide('14_eval_tier3_plot','Tier 3 separates weak warnings from flags','Each dot is one seed (300--309). Dashed lines show the frozen thresholds: 0.8 for agreement and 0.2 for the gap.',r'\SlideFigure[.64\linewidth]{eval_tier3}'+r'\par\smallskip U2 also requires regret at most 0.005 in 9 of 10 seeds; only 7--8 qualify. Seeds and model set differ from the frozen protocol; scoring and thresholds are unchanged. Tier 4 is not implemented.', 'U1 falls short of the mean-agreement threshold; U3 clears its gap threshold in every seed.')
with (R/'EVAL_SLIDES.md').open('a') as f:
 f.write('\n\nTier 3 revision: per-seed dots copied from report_tier3/rates.csv (local U1; U3 gap) and rank_agreement_regret.csv (local U2). No predictions rescored.\n'+json.dumps({'tier3_means':means,'regret_good_seeds':[7,7,8]},indent=2))

p=R/'slides/14_eval_tier3_plot.tex'
p.write_text(p.read_text().replace(r'\fontsize{9}{11}',r'\fontsize{8}{9.6}'))
