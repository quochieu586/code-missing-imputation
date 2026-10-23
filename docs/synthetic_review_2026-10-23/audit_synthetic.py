"""Independent audit; original sources and published artifacts are read-only."""
import copy,csv,hashlib,importlib.util,json,os,shutil,sys,tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np
from longitudinal_impute.synthetic import GeneratorConfig,generate,load_conditioning,load_evaluation,export_dataset
from longitudinal_impute.synthetic.cli import main
from longitudinal_impute.synthetic.scoring import score
from longitudinal_impute.synthetic.backend import source_provenance,resolve_rscript

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BUNDLES=ROOT/'datasets/synthetic/wave_panel_v1'
R=resolve_rscript()
REPORT={'audit_version':2,'numpy':np.__version__,'source_hashes':{},'profiles':{},'probes':{},'results_recomputed':[]}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def safe(value):
 if isinstance(value,np.generic): return value.item()
 if isinstance(value,np.ndarray): return value.tolist()
 raise TypeError(type(value).__name__)

for folder in ('nextgen/src/longitudinal_impute/synthetic','nextgen/tests'):
 for p in (ROOT/folder).glob('*'):
  if p.is_file(): REPORT['source_hashes'][str(p.relative_to(ROOT))]={'sha256':sha(p),'lines':len(p.read_text(encoding='utf-8-sig').splitlines())}
REPORT['author_source_verified']=source_provenance()['commit']
for profile in ('easy','medium','hard'):
 root=BUNDLES/profile; manifest=json.loads((root/'manifest.json').read_text())
 checks={rel:sha(root/rel)==digest for rel,digest in manifest['files_sha256'].items()}
 generated=generate(GeneratorConfig(profile=profile))
 with np.load(root/'private/ground_truth.npz',allow_pickle=False) as z:
  arrays={k:z[k].copy() for k in z.files}
 matches={k:bool(np.array_equal(v,arrays[k])) for k,v in generated.items()}
 p,x,n=arrays['p'],arrays['counts'],arrays['N']
 clock=np.arange(120)[None,:,None]
 before=clock<arrays['onsets'][:,None,:]
 split_counts={s:int((arrays['split']==s).sum()) for s in ('train','validation','test')}
 truth_checks={'p_finite':bool(np.isfinite(p).all()),'p_nonnegative':bool((p>=0).all()),
 'max_closure_error':float(np.abs(p.sum(axis=2)+arrays['residual_p']-1).max()),
 'count_conservation':bool(np.array_equal(x.sum(axis=2)+arrays['residual_counts'],n)),
 'preonset_zero':bool((p[before]==0).all() and (x[before]==0).all()),
 'postonset_positive':bool((p[~before]>0).all()),
 'strata_correct':bool(np.array_equal(arrays['strata'],np.where(p==0,0,np.where(x==0,1,2)))),
 'entity_splits':split_counts,'generator_hash_matches':sha(ROOT/'nextgen/src/longitudinal_impute/synthetic/generator.py')==manifest['generator_sha256']}
 maskstats={}
 for item in manifest['masks']:
  name=item['name']; model=load_conditioning(root,name); labels,latent=load_evaluation(root,name)
  h=labels.target_mask
  hfile=np.loadtxt(root/'conditioning'/name/'hidden_mask.csv',delimiter=',').astype(bool)
  split_entities=[set(load_conditioning(root,name,split=s).entity_ids) for s in ('train','validation','test')]
  per_entity=h.reshape(60,120,12).mean(axis=(1,2))
  # Actual exposure rates, without adding a probability mechanism.
  row_selected=h.any(axis=1)
  nflat=n.ravel(); trait=np.repeat(arrays['entity_trait_01'],120)
  qn=np.quantile(nflat,[.2,.8]); qt=np.quantile(trait,[.2,.8])
  maskstats[name]={'mask_matches_csv':bool(np.array_equal(h,hfile)),
    'no_hidden_truth_in_conditioning':bool(np.isnan(model.counts[h]).all()),
    'observed_preserved':bool(np.array_equal(model.counts[~h],x.reshape(-1,12)[~h])),
    'metadata_keys':list(model.metadata),'case_rate':float(row_selected.mean()),'cell_rate':float(h.mean()),
    'low_n_case_rate':float(row_selected[nflat<=qn[0]].mean()),'high_n_case_rate':float(row_selected[nflat>=qn[1]].mean()),
    'low_trait_case_rate':float(row_selected[trait<=qt[0]].mean()),'high_trait_case_rate':float(row_selected[trait>=qt[1]].mean()),
    'entity_rate_range':[float(per_entity.min()),float(per_entity.max())],
    'split_disjoint':not any(split_entities[i]&split_entities[j] for i in range(3) for j in range(i)),
    'hidden_strata':{str(code):int((h&(latent['strata']==code)).sum()) for code in (0,1,2)}}
 REPORT['profiles'][profile]={'all_file_hashes_match':all(checks.values()),'file_count':len(checks),
 'replay_exact_all_arrays':all(matches.values()),'replay_array_results':matches,'truth_checks':truth_checks,'masks':maskstats}
 REPORT['profiles'][profile]['residual_zero_count_rows']=int((arrays['residual_counts']==0).sum())
 REPORT['profiles'][profile]['diagnostics']={
  'mean_n':float(n.mean()),'cv_n':float(n.std()/n.mean()),
  'true_zero_cells':int((arrays['strata']==0).sum()),
  'sampling_zero_cells':int((arrays['strata']==1).sum()),
  'positive_cells':int((arrays['strata']==2).sum()),
  'min_residual_p':float(arrays['residual_p'].min()),
  'residual_mean':float(arrays['residual_p'].mean()),'residual_sd':float(arrays['residual_p'].std())}

# Recompute every saved prediction; separate empirical proportions from latent truth.
exp=ROOT/'experiments/synthetic_easy_baselines_2026-10-23'
manifest=json.loads((exp/'run_manifest.json').read_text())
REPORT['easy_benchmark_source_hash_match']={rel:sha(ROOT/rel)==digest for rel,digest in manifest['source_sha256'].items()}
REPORT['easy_benchmark_bundle_hash_matches']=sha(BUNDLES/'easy/manifest.json')==manifest['bundle_manifest_sha256']
for path in sorted(exp.glob('*/*/test_predictions.npz')):
 mask,method=path.parts[-3:-1]
 panel=load_conditioning(BUNDLES/'easy',mask,split='test')
 labels,latent=load_evaluation(BUNDLES/'easy',mask,split='test')
 with np.load(path,allow_pickle=False) as z: pred=z['counts'].copy(); ids=z['row_ids'].copy()
 hidden=labels.target_mask
 saved=json.loads((path.parent/'result.json').read_text())
 ratio=pred/panel.totals[:,None]
 empirical=labels.values/panel.totals[:,None]
 delta=pred[hidden]-labels.values[hidden]
 actual={'count_mae':float(np.abs(delta).mean()),'count_rmse':float(np.sqrt(np.mean(delta**2))),
 'empirical_ratio_mae':float(np.abs(ratio[hidden]-empirical[hidden]).mean()),
 'empirical_ratio_rmse':float(np.sqrt(np.mean((ratio[hidden]-empirical[hidden])**2))),
 'latent_p_mae':float(np.abs(ratio[hidden]-latent['p'][hidden]).mean()),
 'latent_p_rmse':float(np.sqrt(np.mean((ratio[hidden]-latent['p'][hidden])**2))),
 'full_sample_ratio_to_latent_mae':float(np.abs(empirical[hidden]-latent['p'][hidden]).mean())}
 checks={'row_ids_match':tuple(ids)==panel.row_ids,'finite':bool(np.isfinite(pred).all()),
 'nonnegative':bool((pred>=0).all()),'observed_preserved':bool(np.array_equal(pred[~hidden],panel.counts[~hidden])),
 'budget_violations':int((pred.sum(axis=1)>panel.totals+1e-8).sum()),
 'saved_metrics_match':bool(np.isclose(actual['count_mae'],saved['test']['overall']['count_mae']) and np.isclose(actual['empirical_ratio_mae'],saved['test']['overall']['ratio_mae']))}
 diagnostics=json.loads((path.parent/'diagnostics.json').read_text())
 def extract(obj,key):
  result=[]
  if isinstance(obj,dict):
   for k,v in obj.items():
    if k==key: result.append(v)
    result.extend(extract(v,key))
  elif isinstance(obj,list):
   for v in obj: result.extend(extract(v,key))
  return result
 REPORT['results_recomputed'].append({'mask':mask,'method':method,'metrics':actual,'checks':checks,
 'converged_fields':extract(diagnostics,'converged'),'iterations_fields':extract(diagnostics,'iterations'),
 'latent_strata':{name:{'n':int((hidden&(latent['strata']==k)).sum()),
  'latent_p_mae':float(np.abs(ratio[hidden&(latent['strata']==k)]-latent['p'][hidden&(latent['strata']==k)]).mean())}
  for k,name in enumerate(('true_zero','sampling_zero','positive_count'))},
 'zero_predictor':{'count_mae':float(labels.values[hidden].mean()),
  'empirical_ratio_mae':float(empirical[hidden].mean()),'latent_p_mae':float(latent['p'][hidden].mean())}})

# Fault injection on disposable copies only.
work=OUT/'probe_artifacts'; work.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='review-',dir=work) as temp:
 temp=Path(temp); bundle=temp/'copy'; shutil.copytree(BUNDLES/'easy',bundle)
 mask='01_mcar_declared_patterns'
 labels,truth=load_evaluation(bundle,mask,split='test')
 public=load_conditioning(bundle,mask,split='test'); target_row,target_col=np.argwhere(labels.target_mask)[0]
 key=public.row_ids[target_row]; entity,dt=json.loads(key)
 file=bundle/'private/complete_counts.csv'
 with file.open(newline='') as f:
  reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
 target=next(r for r in rows if r['entity_id']==entity and r['date']==dt)
 col=f'component_{target_col+1:02d}'; old=int(target[col]); target[col]=str(old+1)
 with file.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 try:
  changed,_=load_evaluation(bundle,mask,split='test')
  REPORT['probes']['private_csv_mutation']={'accepted':True,'before_label':float(labels.values[target_row,target_col]),'after_label':float(changed.values[target_row,target_col]),'changed_file_hash':sha(file),'original_hash':json.loads((bundle/'manifest.json').read_text())['files_sha256']['private/complete_counts.csv']}
 except Exception as exc: REPORT['probes']['private_csv_mutation']={'accepted':False,'error':repr(exc)}
 shutil.copyfile(BUNDLES/'easy/private/complete_counts.csv',file)
 # Change latent p in a shape-preserving way; this should be rejected before scoring.
 file=bundle/'private/ground_truth.npz'
 with np.load(file,allow_pickle=False) as z: copied={k:z[k].copy() for k in z.files}
 copied['p']=np.roll(copied['p'],1,axis=0)
 np.savez_compressed(file,**copied)
 try:
  _,changed=load_evaluation(bundle,mask,split='test')
  REPORT['probes']['latent_npz_permutation']={'accepted':True,'latent_values_changed':bool(not np.array_equal(changed['p'],truth['p']))}
 except Exception as exc: REPORT['probes']['latent_npz_permutation']={'accepted':False,'error':repr(exc)}
 # Ordinary CSV row reordering breaks latent identity even though counts align by key.
 fresh=temp/'reordered'; shutil.copytree(BUNDLES/'easy',fresh)
 file=fresh/'private/complete_counts.csv'
 with file.open(newline='') as f:
  reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
 with file.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(reversed(rows))
 try:
  changed_labels,changed=load_evaluation(fresh,mask,split='test')
  REPORT['probes']['private_csv_reorder']={'accepted':True,
   'latent_p_still_align':bool(np.array_equal(truth['p'],changed['p']))}
 except ValueError as exc: REPORT['probes']['private_csv_reorder']={'accepted':False,'error':repr(exc)}
 # Wrong scenario from a different capability must not become fixed_wave.
 spec={'scenario':'table1_design','mechanism':'MCAR','case_probability':.3,'seed':2018,'wave_time':'2000-03-01'}
 file=temp/'wrong_scenario.json'; file.write_text(json.dumps(spec)); destination=temp/'wrong_scenario_out'
 code=main(['--rscript',R,'mask','--data',str(BUNDLES/'easy/private/complete_counts.csv'),
 '--schema-config',str(BUNDLES/'easy/schema.toml'),'--spec',str(file),'--output',str(destination)])
 REPORT['probes']['wrong_scenario']={'requested':spec['scenario'],'exit_code':code,
 'actual_scenario':json.loads((destination/'manifest.json').read_text())['diagnostics']['scenario'] if (destination/'manifest.json').exists() else None}
 # One invalid spec after a valid one leaves partially generated artifacts at final path.
 valid=json.loads((ROOT/'nextgen/configs/schouten2018_mcar_patterns.json').read_text())
 invalid={**valid,'case_probability':2}
 dest=temp/'partial'
 try: export_dataset(dest,GeneratorConfig(scale='small'),mask_specs=[valid,invalid],rscript=R)
 except Exception as exc:
  REPORT['probes']['non_atomic_export']={'error':str(exc),'final_path_exists':dest.exists(),'published_manifest_exists':(dest/'manifest.json').exists(),
  'already_written_files':[str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()]}
 try:
  export_dataset(dest,GeneratorConfig(scale='small'),mask_specs=[valid],rscript=R)
  REPORT['probes']['non_atomic_export']['retry_succeeded']=True
 except Exception as exc: REPORT['probes']['non_atomic_export']['retry_error']=repr(exc)
 # Programmatic controls accept a negative standard deviation.
 cfg=GeneratorConfig(scale='small',residual_sd=-.03)
 try:
  cfg.validate(); REPORT['probes']['negative_residual_sd']={'accepted':True}
 except Exception as exc: REPORT['probes']['negative_residual_sd']={'accepted':False,'error':repr(exc)}

table=ROOT/'experiments/schouten2018_table1_2026-10-23'
tm=json.loads((table/'manifest.json').read_text())
with (table/'table1_design_runs.csv').open(newline='') as f: rows=list(csv.DictReader(f))
pattern_checks=[]
for row in rows:
 h=np.loadtxt(table/f'{row["key"]}_hidden.csv',delimiter=',').astype(bool)
 pattern_checks.append(not h[:,2].any() and np.array_equal(h[:,0],h[:,1]))
REPORT['table1']={'all_files_hash_match':all(sha(table/rel)==digest for rel,digest in tm['design_files_sha256'].items()),
 'runs':len(rows),'all_joint_patterns_correct':all(pattern_checks),'replicates':tm['replicates'],
 'post_imputation_inference_tables':tm['post_imputation_inference_tables']}
# Verify the new format against the immutable v1 numerical snapshot.
REPORT['v2_comparison']={}
for profile in ('easy','medium','hard'):
 old=BUNDLES/profile
 new=ROOT/'datasets/synthetic/wave_panel_v2'/profile
 if not new.exists(): continue
 manifest2=json.loads((new/'manifest.json').read_text())
 with np.load(old/'private/ground_truth.npz') as z1, np.load(new/'private/ground_truth.npz') as z2:
  same=all(np.array_equal(z1[k],z2[k]) for k in z1.files)
 masks=[]
 for item in manifest2['masks']:
  name=item['name']
  l1,p1=load_evaluation(old,name); l2,p2=load_evaluation(new,name)
  masks.append(np.array_equal(l1.target_mask,l2.target_mask) and np.array_equal(l1.values,l2.values,equal_nan=True)
               and np.array_equal(p1['p'],p2['p']))
 REPORT['v2_comparison'][profile]={'all_original_arrays_equal':same,'all_masks_labels_p_equal':all(masks),
  'all_file_hashes_match':all(sha(new/rel)==digest for rel,digest in manifest2['files_sha256'].items()),
  'format_version':manifest2['format_version']}
REPORT['expected_probes_pass'] = (
 all(not REPORT['probes'][key]['accepted'] for key in
     ('private_csv_mutation','latent_npz_permutation','private_csv_reorder','negative_residual_sd'))
 and REPORT['probes']['wrong_scenario']['exit_code'] != 0
 and not REPORT['probes']['non_atomic_export']['final_path_exists']
 and REPORT['probes']['non_atomic_export'].get('retry_succeeded',False))
REPORT['review_tests']={'note':'Run pytest separately; see FIX_COMPLETION_VI.md for observed results.'}
# Additive rescore: preserve original six predictions, selections, reports and v1 bundles.
rescored=[]
for path in sorted(exp.glob('*/*/test_predictions.npz')):
 mask,method=path.parts[-3:-1]
 panel=load_conditioning(BUNDLES/'easy',mask,split='test')
 labels,latent=load_evaluation(BUNDLES/'easy',mask,split='test')
 with np.load(path,allow_pickle=False) as z:
  if tuple(z['row_ids']) != panel.row_ids: raise ValueError('Saved prediction identity mismatch')
  metrics=score(labels,latent,panel,z['counts'])
 rescored.append({'mask':mask,'method':method,'prediction_sha256':sha(path),'metrics':metrics})
(OUT/'dual_truth_metrics_v2.json').write_text(json.dumps({
 'metric_version':'dual-truth-v2','bundle_version':'synthetic-panel-v1',
 'protocol':'Additive rescore of saved test predictions; original validation X/N MAE selections unchanged; no test retuning.',
 'results':rescored},indent=2,default=safe),encoding='utf-8')
(OUT/'audit_results_post_fix.json').write_text(json.dumps(REPORT,indent=2,default=safe),encoding='utf-8')
print(json.dumps({'audit_output':str(OUT/'audit_results_post_fix.json'),'profiles':{k:{'hashes':v['all_file_hashes_match'],'replay':v['replay_exact_all_arrays']} for k,v in REPORT['profiles'].items()},'probes':REPORT['probes'],'expected_probes_pass':REPORT['expected_probes_pass']},indent=2,default=safe))
checks = [REPORT['expected_probes_pass'], REPORT['table1']['all_files_hash_match'],REPORT['table1']['all_joint_patterns_correct']]
checks += [v['all_file_hashes_match'] and v['replay_exact_all_arrays'] for v in REPORT['profiles'].values()]
checks += [all(v[k] for k in ('all_original_arrays_equal','all_masks_labels_p_equal','all_file_hashes_match')) for v in REPORT['v2_comparison'].values()]
checks += [all(item['checks'][key] for key in ('row_ids_match','finite','nonnegative','observed_preserved','saved_metrics_match')) and item['checks']['budget_violations']==0 for item in REPORT['results_recomputed']]
if not all(checks): raise SystemExit('Repair audit checks failed')
