"""Read-only experiment/code audit; all outputs go to reports, no training."""
from __future__ import annotations
import rework_runtime
from rework_runtime import ROOT
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import pytest

from scripts import run_e5_rework as R
from src.evaluation import rework_gates as RG
from scripts.diagnose_e5_rework import frozen_hashes, independent_metrics
from src.evaluation.research_pilot import sha256, write_json

B = ROOT/'artifacts/e3e4_rework_2026-10-05_b'
OUT = ROOT/'reports/e5_codebase_audit_2026-10-06'
SCENARIOS = (*RG.PRIMARY, 'random_cell_bank0')


def main():
    if (OUT/'audit.json').exists(): raise FileExistsError('Completed audit exists')
    OUT.mkdir(parents=True, exist_ok=True); start=time.perf_counter()
    all_inputs={str(p.relative_to(ROOT)):sha256(p) for p in B.rglob('*') if p.is_file()}
    baseline=json.loads((ROOT/'artifacts/e3e4_rework_2026-10-05/frozen_hashes_before.json').read_text())
    old_ok=baseline==frozen_hashes()
    cfg=json.loads((B/'config.json').read_text()); stored=json.loads((B/'validation_scores.json').read_text())
    manifest=json.loads((ROOT/'artifacts/e0_extended/manifest.json').read_text())
    df=pd.read_csv(ROOT/'data/covariants.csv'); names=manifest['variant_columns']
    fold=json.loads((ROOT/'artifacts/e0_extended/cv_splits.json').read_text())[0]
    outer=df.iloc[fold['train_rows']]; pool=outer[outer.location!='Denmark']; val=outer[outer.location=='Denmark']
    ids=val.index.to_numpy(); truth=val[names].to_numpy(float)
    wvmask=dict(np.load(B/'validation_whole_variant_masks.npz'))
    scenarios={s:wvmask[s] for s in RG.PRIMARY}
    scenarios['random_cell_bank0']=np.load(B/'validation_random_cell_bank0.npz')['bank0']
    regenerated,info=R.whole_variant_masks(pool,names,len(val))
    assert all(np.array_equal(regenerated[k],v) for k,v in wvmask.items())
    assert np.array_equal(R.make_random_cell_mask(len(val)),scenarios['random_cell_bank0'])
    raw=pool[names].to_numpy(float); ok=np.isfinite(raw).all(1)&(np.nansum(raw,1)>0)
    full=pool.loc[ok]; rawfull=full[names].to_numpy(float); tm=R.load_transform_module()
    bins,_=R.prevalence_bins(raw,names); delta_inner=tm.fit_positivity_delta(raw)
    rows=[]; mismatches=[]; contract_failures=[]; independent_scores={}; checkpoint_checks=[]
    verified=0

    def check(key,scenario,pred,score,branch):
        nonlocal verified
        hidden=scenarios[scenario]; visible=~hidden
        if not np.isfinite(pred).all() or (pred<0).any() or not np.array_equal(pred[visible],truth[visible]):
            contract_failures.append(f'{branch}/{key}/{scenario}')
        calc,_,eligible=independent_metrics(truth,pred,hidden)
        for attr,value in calc.items():
            if abs(value-score[attr])>1e-12: mismatches.append([branch,key,scenario,attr])
        assert score['eligible_row_positions']==np.flatnonzero(eligible).tolist()
        method=key.rsplit('__seed',1)[0]
        diag=R.numerical_diagnostics(pred,np.where(hidden,np.nan,truth))
        assert diag==score['numerical_diagnostics']
        rows.append({'branch':branch,'method':method,'key':key,'scenario':scenario,
            **calc,'n_eligible':int(eligible.sum()),'n_nonzero_hidden':int((hidden&eligible[:,None]&(truth>0)).sum()),
            'n_zero_hidden':int((hidden&eligible[:,None]&(truth==0)).sum()),
            'scale_flags':diag['n_scale_guard_rows'],'max_scale_ratio':diag['count_sum_over_visible_sum_max'],
            'best_epoch':score.get('best_epoch'),'channels':score.get('channels'),
            'latent_dim':score.get('latent_dim'),'parameter_count':score.get('parameter_count')})
        verified+=1
        result={**score,**calc}; return result

    for branch,cache in [('main',B/'cache_validation'),('cap8',B/'cache_cap8')]:
        scn_keys=SCENARIOS if branch=='main' else ('random_cell_bank0',)
        files=sorted(cache.glob('*__seed*.json')); assert len(files)==96
        for jp in files:
            blob=json.loads(jp.read_text()); key=blob['key']
            assert blob['config_sha256']==sha256(B/'config.json') and blob['mc_samples']==8 and blob['max_epochs']==40
            assert len(blob['trace'])==40
            best=min(blob['trace'],key=lambda r:r['validation_fixture_loss'])
            assert best['epoch']==blob['best_epoch'] and best['validation_fixture_loss']==blob['best_loss']
            arrays=np.load(jp.with_suffix('.npz'))
            state=torch.load(jp.with_suffix('.pt'),weights_only=True,map_location='cpu')
            channel_width=state['input_projection.weight'].shape[0]
            latent_dim=state['feature_embedding.weight'].shape[0]
            assert channel_width==blob['channels']
            refs=R.hkglr_refs(key.split('__')[1],pool,names)
            transform=tm.make_transform(key.split('__')[1],17,.5,refs if key.split('__')[1].startswith('hk') else None)
            assert transform.forward(rawfull).shape[1]==latent_dim
            for scn in scn_keys:
                lat=arrays[f'latent__{scn}']; assert lat.shape==(8,len(val),latent_dim) and np.isfinite(lat).all()
                pred=arrays[f'pred__{scn}']; saved=blob['scores'][scn]
                assert saved['latent_dim']==latent_dim and saved['channels']==channel_width
                recalced,_=tm.restore_observed_counts(transform.inverse(lat.mean(0)),np.where(scenarios[scn],np.nan,truth))
                assert np.max(np.abs(recalced-pred))<1e-12
                sc=check(key,scn,pred,saved,branch)
                if branch=='main':
                    assert saved==stored[key][scn]
                    independent_scores.setdefault(key,{})[scn]=sc
            checkpoint_checks.append({'branch':branch,'key':key,'channels':channel_width,'latent_dim':latent_dim,
                'best_epoch':blob['best_epoch'],'parameter_count':sum(v.numel() for k,v in state.items() if k!='step_table'),
                'has_saved_mean_or_scale':bool('mean' in state or 'scalar_scale' in state)})
        controls=np.load(cache/'controls_predictions.npz')
        assert len(controls.files)==4*len(scn_keys)
        for key in controls.files:
            scn=next(s for s in scn_keys if key.endswith('__'+s)); method=key[:-len(scn)-2]
            # Controls are identical across main/cap8, calculated independently of the model.
            sc=check(method,scn,controls[key],stored[key],branch)
            if branch=='main': independent_scores[key]=sc

    # Full gate structures, not only final booleans.
    gate=RG.validation_gates(independent_scores,R.ALL_TRANSFORMS,R.INIT_MODES,R.ALL_SEEDS)
    saved_gates=json.loads((B/'validation_gates.json').read_text()); assert gate==saved_gates
    finalists,diagnostic=RG.select_finalists(independent_scores,gate,R.ALL_TRANSFORMS,R.INIT_MODES,R.ALL_SEEDS)
    saved_finalists=json.loads((B/'validation_finalists.json').read_text())
    assert finalists==saved_finalists['finalists'] and diagnostic==saved_finalists['diagnostic_top3_not_finalists']
    assert not (B/'canada_evaluated.lock').exists() and not (B/'cache_test').exists()

    # Recompute current initializer/baseline paths, and poison real held-out inputs.
    control_checks=[]
    for scn,hidden in scenarios.items():
        q=np.where(hidden,np.nan,truth)
        init,fb,ek,prov=R.initialize_crossfit(q,ids,val.location.to_numpy(),pool,names)
        donor_ids={d for rr in prov.values() for r in rr for c in r['cells'] for d in c['donor_ids']}
        assert donor_ids<=set(pool.index) and not donor_ids & set(ids)
        for mode in R.INIT_MODES:
            pred=np.load(B/'cache_validation/controls_predictions.npz')[f'init_only__{mode}__{scn}']
            assert np.array_equal(init[mode],pred)
        poisoned=truth.copy(); poisoned[hidden]=1e200; safe=np.where(hidden,np.nan,poisoned)
        init2,fb2,ek2,_=R.initialize_crossfit(safe,ids,val.location.to_numpy(),pool,names)
        for mode in R.ALL_MODES:
            a=R.condition_features(q,~hidden,'init' if mode in init else mode,R.raw_scale_for(raw),init.get(mode),fb.get(mode),ek.get(mode))
            b=R.condition_features(safe,~hidden,'init' if mode in init2 else mode,R.raw_scale_for(raw),init2.get(mode),fb2.get(mode),ek2.get(mode))
            assert np.array_equal(a,b)
        for name in ('linear','locf_nocb'):
            pred,score=R.evaluate_baseline_on_scenario(val,names,hidden,R.time_encode(val.date,pd.to_datetime(pool.date).min().toordinal()),full,.5,bins,name)
            cached=np.load(B/'cache_validation/controls_predictions.npz')[f'{name}__{scn}']
            assert np.array_equal(pred,cached)
            poison_frame=val.copy(); poison_frame[names]=poisoned
            poisoned_pred,_=R.evaluate_baseline_on_scenario(poison_frame,names,hidden,R.time_encode(val.date,pd.to_datetime(pool.date).min().toordinal()),full,.5,bins,name)
            assert np.array_equal(pred,poisoned_pred)
        control_checks.append({'scenario':scn,'actual_initializers_and_baselines_match_cache':True,'poison_condition_all_modes':True,
            'no_positive_visible_rows':int((np.nansum(q,1)==0).sum()),
            'no_positive_visible_eligible_rows':int(((np.nansum(q,1)==0)&(truth.sum(1)>0)).sum())})

    # Replay several saved models from current code and train-only fitted stats.
    replays=[]
    fixture=R.build_val_fixture(pool,val,names,8,wvmask,scenarios['random_cell_bank0'])
    for mode,name in [('hron_2a','clr16'),('no_init','ilr16'),('mask_only','hkglr16_actual')]:
        key=f'{mode}__{name}__seed42'; refs=R.hkglr_refs(name,pool,names)
        trans=tm.make_transform(name,17,.5,refs if name.startswith('hk') else None)
        target=trans.forward(rawfull); mean=target.mean(0); scale=float(np.sqrt(np.mean((target-mean)**2)))
        model=R.CSDICoreRework(target.shape[1],channels=R.MODEL_CHANNELS[name],heads=1)
        model.load_state_dict(torch.load(B/'cache_validation'/f'{key}.pt',map_location='cpu',weights_only=True))
        times=R.time_encode(val.date,pd.to_datetime(pool.date).min().toordinal()); windows=R.window_indices(val,ids,8)
        fn=R.make_fixture(fixture['query'],fixture['visible'],fixture['times'],fixture['window'][0],fixture['init'],fixture['fb'],fixture['ek'],fixture['raw_scale'],trans,mean,scale,mode)
        loss=fn(model); blob=json.loads((B/'cache_validation'/f'{key}.json').read_text())
        assert abs(loss-blob['best_loss'])<1e-12
        scn='whole_common_n2'; hidden=scenarios[scn]; q=np.where(hidden,np.nan,truth)
        init,fb,ek,_=R.initialize_crossfit(q,ids,val.location.to_numpy(),pool,names)
        pred,_,lat=R.evaluate_csdi_on_scenario(model,trans,name,refs,mean,scale,val,names,hidden,truth,.5,bins,tm,
            R.raw_scale_for(raw),mode,init.get(mode),fb.get(mode),ek.get(mode),windows,times,42)
        cache=np.load(B/'cache_validation'/f'{key}.npz')
        assert np.max(np.abs(lat-cache[f'latent__{scn}']))<1e-12
        assert np.max(np.abs(pred-cache[f'pred__{scn}']))<1e-12
        replays.append({'key':key,'scenario':scn,'best_loss_replayed':loss,'latent_and_prediction_match':True})

    # Counterexample: current scale flags do not block an otherwise passing gate.
    synthetic={}; s='whole_rare_n2'
    for seed in R.ALL_SEEDS:
        for mode,m2 in [('no_init',10.),('mask_only',11.),('hron_2a',8.)]:
            synthetic[f'{mode}__clr16__seed{seed}']={s:{'m2':m2,'clr_mae_nonzero_hidden':1.,'status':'OK',
                'nonfinite_cells':0,'negative_cells':0,'observed_restoration_exact':True,
                'numerical_diagnostics':{'n_scale_guard_rows':9}}}
    synthetic[f'init_only__hron_2a__{s}']={'m2':10.,'clr_mae_nonzero_hidden':1.,'status':'OK',
        'nonfinite_cells':0,'negative_cells':0,'observed_restoration_exact':True}
    weakened_gate=RG.validation_gates(synthetic,['clr16'],['hron_2a'],R.ALL_SEEDS,[s])
    assert weakened_gate['passed']

    data=pd.DataFrame(rows); data.to_csv(OUT/'method_scenarios.csv',index=False)
    grouped=data.groupby(['branch','method','scenario']).agg(m2_mean=('m2','mean'),m2_sd=('m2','std'),
        mae_nonzero_mean=('clr_mae_nonzero_hidden','mean'),mae_zero_mean=('clr_mae_zero_hidden','mean'),
        flags_mean=('scale_flags','mean'),flags_max=('scale_flags','max')).reset_index()
    grouped.to_csv(OUT/'method_summary.csv',index=False)
    pd.DataFrame(checkpoint_checks).to_csv(OUT/'checkpoint_audit.csv',index=False)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(buf):
        test_code=int(pytest.main([str(ROOT/'tests/unit'),'-q','-p','no:cacheprovider',f'--basetemp={OUT/"pytest_tmp"}']))
    (OUT/'tests.txt').write_text(buf.getvalue(),encoding='utf-8')
    latest={str(p.relative_to(ROOT)):sha256(p) for p in B.rglob('*') if p.is_file()}
    assert all_inputs==latest and baseline==frozen_hashes()
    original_hashes=json.loads((B/'file_hashes.json').read_text())
    art_changes=[rel for rel,want in original_hashes['artifact_hashes'].items() if sha256(B/rel)!=want]
    code_changes=[rel for rel,want in original_hashes['code_hashes'].items() if sha256(ROOT/rel)!=want]
    main=data[data.branch=='main']; cap=data[data.branch=='cap8']
    findings={
        'scale_guard_not_gated_counterexample':weakened_gate['passed'],
        'registered_clr16_latent_dim':next(t['latent_dim'] for t in cfg['transforms'] if t['name']=='clr16'),
        'actual_clr16_latent_dims':sorted(main[main.method.str.endswith('__clr16')].latent_dim.dropna().unique().tolist()),
        'main_entries_with_scale_flags':int((main.scale_flags>0).sum()),'main_total_entries':len(main),
        'main_flagged_method_row_occurrences':int(main.scale_flags.sum()),
        'main_max_scale_ratio':float(main.max_scale_ratio.max()),
        'cap8_entries_with_scale_flags':int((cap.scale_flags>0).sum()),
        'checkpoint_metadata_not_saved':all(not x['has_saved_mean_or_scale'] for x in checkpoint_checks),
        'fixture_row_ids':fixture['window'][0].tolist(),
        'fixture_stable_row_ids':val[val[names].sum(1)>0].index.to_numpy()[fixture['window'][0]].tolist(),
        'fixture_full_positive_rows':len(fixture['query']),
        'fixture_vs_eval_random_masks_differing_eligible_rows':int((fixture['visible']!=~scenarios['random_cell_bank0'][truth.sum(1)>0]).any(1).sum()),
        'inner_positivity_delta':delta_inner,
        'capacity8_clr16_clr17_identical_metrics':bool(np.array_equal(
            cap[cap.method.str.endswith('__clr16')].sort_values('key').m2.to_numpy(),
            cap[cap.method.str.endswith('__clr17')].sort_values('key').m2.to_numpy()))}
    result={'audit_completed':True,'independent_metrics_match':not mismatches,'metric_mismatches':mismatches,
        'verified_method_scenarios':verified,'checkpoint_count':len(checkpoint_checks),'contract_failures':contract_failures,
        'gates_recomputed_full_match':True,'conditioning_passed_cells':sum(v['passed'] for v in gate['conditioning_gate']['pairs'].values()),
        'conditioning_total_cells':len(gate['conditioning_gate']['pairs']),
        'efficacy_passed_pairs':sum(v['passed'] for v in gate['efficacy_gate']['pairs'].values()),
        'efficacy_total_pairs':len(gate['efficacy_gate']['pairs']),
        'canada_seal_verified':True,'new_training':False,'input_artifact_hashes_match':not art_changes,
        'upstream_code_hashes_match':not code_changes,'old_frozen_files_match':old_ok,
        'artifact_hash_mismatches':art_changes,'code_hash_mismatches':code_changes,
        'poison_and_controls':control_checks,'checkpoint_replays':replays,'implementation_findings':findings,
        'legacy_and_rework_test_exit_code':test_code,'tests_output_tail':buf.getvalue()[-500:],
        'original_inputs_unchanged':all_inputs==latest,'e5_status':'BLOCKED','finalists':[],
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'data_sha256':sha256(ROOT/'data/covariants.csv'),'python':sys.executable,'torch':torch.__version__,
        'runtime_seconds':time.perf_counter()-start,
        'source_code_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'scripts/run_e5_rework.py',
            ROOT/'scripts/audit_e5_rework.py',ROOT/'src/stage_b/research_csdi_rework.py',ROOT/'src/evaluation/rework_gates.py']}}
    write_json(OUT/'audit.json',result)
    print(json.dumps({k:result[k] for k in ('audit_completed','verified_method_scenarios','checkpoint_count','independent_metrics_match',
        'legacy_and_rework_test_exit_code','implementation_findings','runtime_seconds')},indent=2))


if __name__=='__main__':main()
