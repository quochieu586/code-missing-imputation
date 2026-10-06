"""Phase A: read-only legacy audit, held-out inference/oracles, no model fitting."""
from __future__ import annotations
import rework_runtime
from rework_runtime import ROOT
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.run_e3_revised import load_transform_module, window_indices, initialize_crossfit
from src.stage_b.research_csdi import CSDICore, condition_features, schedule, forward_noise
from src.evaluation.research_pilot import sha256, write_json, write_hashes, evaluate, numerical_diagnostics

OUT = ROOT / 'artifacts/e3e4_rework_2026-10-05'
OLD = ROOT / 'artifacts/e3_csdi_revised'
FROZEN = ('e0_extended','e1_pilot','e2_transforms','e3_csdi','e4_pilot','e3_csdi_revised','e4_pilot_revised','e0_checklist_2026-10-05')


def frozen_hashes():
    files = [p for folder in FROZEN for p in (ROOT/'artifacts'/folder).rglob('*') if p.is_file()]
    files += list((ROOT/'directive').glob('*.pdf')) + [ROOT/'data/covariants.csv']
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def summary(x):
    a = np.asarray(x, dtype=float).ravel()
    return {'n': len(a), 'min': float(a.min()), 'q01':float(np.quantile(a,.01)),
            'median':float(np.median(a)), 'q99':float(np.quantile(a,.99)),
            'max':float(a.max()), 'rms':float(np.sqrt(np.mean(a*a)))}


def independent_metrics(truth, pred, hidden):
    eligible = np.isfinite(truth).all(1) & (truth.sum(1)>0) & hidden.any(1)
    def clr(x):
        z = np.log(np.where(x == 0, .5, x))
        return z - z.mean(1, keepdims=True)
    z, p = clr(truth), clr(pred)
    squared = np.square(z-p).sum(1)
    cells=hidden & eligible[:,None]
    return {'m2':float(squared[eligible].mean()),
        'm3':float(np.linalg.norm(np.cov(z[eligible].T,ddof=1)-np.cov(p[eligible].T,ddof=1),'fro')/16),
        'clr_mae_all_hidden':float(np.abs(z-p)[cells].mean()),
        'clr_mae_nonzero_hidden':float(np.abs(z-p)[cells & (truth>0)].mean()),
        'clr_mae_zero_hidden':float(np.abs(z-p)[cells & (truth==0)].mean())}, squared, eligible


def main():
    start=time.perf_counter()
    if (OUT/'phase_a.json').exists():
        raise FileExistsError('Completed phase A exists; preserve it')
    OUT.mkdir(parents=True, exist_ok=True)
    before=frozen_hashes(); write_json(OUT/'frozen_hashes_before.json',before)
    torch.set_num_threads(2)
    df=pd.read_csv(ROOT/'data/covariants.csv'); names=list(df.columns[3:]); tm=load_transform_module()
    old_manifest=json.loads((OLD/'manifest.json').read_text())
    traces=[]; losses=[]; latents=[]; row_records=[]; contributions=[]; oracles=[]
    metrics_verified=0; failure_count=0; checkpoints=0
    validation_context=None
    for partition in ('validation','test'):
        metrics=json.loads((OLD/f'e3_metrics_{partition}.json').read_text())
        predictions=json.loads((OLD/f'predictions_{partition}.json').read_text())
        meta=metrics['metadata']; ids=np.array(predictions['row_ids']); hidden=np.array(predictions['hidden_eval_mask'],bool)
        truth=df.loc[ids,names].to_numpy(float); query=np.where(hidden,np.nan,truth)
        train=df[df.location.isin(meta['train_locations'])]
        raw=train[names].to_numpy(float); eligible_train=np.isfinite(raw).all(1)&(np.nansum(raw,axis=1)>0)
        train_full=train.loc[eligible_train]; raw_full=train_full[names].to_numpy(float)
        failure_count+=len(metrics['failures'])
        if partition=='validation': validation_context=(meta,ids,hidden,truth,query,train)
        for key,score in metrics['methods'].items():
            pred=np.array([r['prediction_raw'] for r in predictions['methods'][key]],float)
            independent, row_m2, eligible=independent_metrics(truth,pred,hidden)
            for attr,value in independent.items():
                if abs(value-score[attr])>1e-12: raise AssertionError((partition,key,attr,value,score[attr]))
            metrics_verified+=1
            flags=np.zeros(len(ids),bool); flags[score['numerical_diagnostics']['scale_guard_row_positions']]=True
            affected=eligible & flags
            positive=(query>0)&np.isfinite(query); visible_sum=np.nansum(query,1)
            contributions.append({'partition':partition,'method':key,'m2':independent['m2'],
                'n_eligible':int(eligible.sum()),'n_flagged_eligible':int(affected.sum()),
                'flagged_fraction_of_total_m2':float(row_m2[affected].sum()/row_m2[eligible].sum()),
                'm2_median_diagnostic':float(np.median(row_m2[eligible])),
                'm2_90pct_trimmed_diagnostic':float(np.sort(row_m2[eligible])[:max(1,int(.9*eligible.sum()))].mean()),
                'm2_nonflagged_diagnostic':float(row_m2[eligible&~flags].mean()),
                'positive_visible_min_flagged':int(positive.sum(1)[flags].min()) if flags.any() else None})
            comp=None
            if 'transform' in score:
                name=score['transform']; refs=tuple(names.index(n) for n in meta['references'])
                trans=tm.make_transform(name,17,.5,refs if name.startswith('hk') else None)
                target=trans.forward(raw_full); mean=np.array(score['target_mean']); scale=score['target_scalar_scale']
                samples=np.load(OLD/f'{partition}_{key}_latent_samples.npz')['latent_samples']
                comp=trans.inverse(samples.mean(0))
                latents.append({'partition':partition,'method':key,'scalar_scale':scale,
                    'training_latent':summary(target),'training_standardized':summary((target-mean)/scale),
                    'sample_latent':summary(samples),'sample_standardized':summary((samples-mean)/scale),
                    'mean_latent_sample_disagreement_rms':float(np.sqrt(np.mean((samples[0]-samples[1])**2)))})
                trace=score['training_trace']; y=np.array([t['epsilon_mse'] for t in trace])
                for record in trace: traces.append({'partition':partition,'method':key,**record})
                last_slope=float(np.polyfit(np.arange(5),y[-5:],1)[0])
                entry={'partition':partition,'method':key,'train_first':float(y[0]),'train_last':float(y[-1]),
                    'last5_slope_per_epoch':last_slope,'historical_validation_trace_available':False}
                ck=torch.load(OLD/f'{partition}_{key}.pt',map_location='cpu',weights_only=False)
                checkpoints+=1
                assert ck['train_row_ids']==train_full.index.tolist()
                np.testing.assert_allclose(ck['mean'],target.mean(0),atol=1e-12,rtol=0)
                assert abs(float(np.sqrt(np.mean((target-target.mean(0))**2)))-ck['scalar_scale'])<1e-12
                if partition=='validation':
                    # Evaluate final frozen checkpoint, not a history of validation losses.
                    model=CSDICore(target.shape[1]);model.load_state_dict(ck['state_dict']);model.eval()
                    full=np.isfinite(truth).all(1)&(truth.sum(1)>0)
                    idx=np.flatnonzero(full)
                    z0=torch.from_numpy(((trans.forward(truth[idx])-mean)/scale)[:,None,:])
                    initkey='init_only__'+ck['mode']
                    init=np.array([r['prediction_raw'] for r in predictions['methods'][initkey]],float) if initkey in predictions['methods'] else None
                    provenance=json.loads(gzip.decompress((OLD/'query_initializer_provenance_validation.json.gz').read_bytes()))
                    records=provenance.get(ck['mode'])
                    fb=np.array([[c['fallback'] for c in r['cells']] for r in records],float) if records else None
                    ek=np.array([[c['effective_k'] for c in r['cells']] for r in records],float) if records else None
                    cond=condition_features(query,np.isfinite(query),'init' if init is not None else ck['mode'],ck['raw_log_scale'],init,fb,ek)
                    c=torch.from_numpy(cond[idx,None,:]); t=torch.from_numpy(((pd.to_datetime(df.loc[ids[idx],'date']).map(pd.Timestamp.toordinal).to_numpy()-ck['time_origin'])/14.)[:,None])
                    beta,ab=schedule(); vals=[]
                    with torch.no_grad():
                        rng=torch.Generator().manual_seed(80042)
                        for step in range(20):
                            noise=torch.randn(z0.shape,dtype=torch.float64,generator=rng)
                            noisy=forward_noise(z0,noise,ab[step]); p=model(noisy,c,t,torch.full((len(idx),),step,dtype=torch.int64))
                            vals.append(float(((p-noise)**2).mean()))
                    entry['final_checkpoint_validation_epsilon_mse_row_fixture']=float(np.mean(vals))
                    entry['validation_fixture_note']='Length-1 rows; retrospective diagnostic only, not comparable to stochastic training-window loss or a validation history.'
                losses.append(entry)
            for i,rid in enumerate(ids):
                rec={'partition':partition,'method':key,'row_id':int(rid),'location':str(df.loc[rid,'location']),
                    'date':str(df.loc[rid,'date']),'eligible':bool(eligible[i]),'flagged':bool(flags[i]),
                    'row_m2':float(row_m2[i]),'truth_sum':float(truth[i].sum()),
                    'visible_sum':float(visible_sum[i]),'n_positive_visible':int(positive[i].sum()),
                    'sum_ratio':float(pred[i].sum()/max(visible_sum[i],1.)),
                    'min_pred_visible_probability':float(comp[i,positive[i]].min()) if comp is not None and positive[i].any() else None}
                row_records.append(rec)
        # Oracle only on eligible inner train and Denmark; never Canada.
        if partition=='validation':
            for label,r,q,h in [('validation',truth,query,hidden),('train',raw_full,np.where(np.array([[(i+j)%17<5 for j in range(17)] for i in range(len(raw_full))]),np.nan,raw_full),None)]:
                hidden_oracle=~np.isfinite(q)
                trans=tm.make_transform('clr16',17,.5)
                p=trans.inverse(trans.forward(r)); recovered,info=tm.restore_observed_counts(p,q)
                score=evaluate(r,recovered,hidden_oracle,q,.5,{'rare':list(range(5)),'middle':list(range(5,11)),'common':list(range(11,17))})
                anchored=(np.isfinite(q)&(q>0)).any(1); eligible=r.sum(1)>0
                _,row_m2,_=independent_metrics(r,recovered,hidden_oracle)
                # Isolate pseudocount roundtrip: hidden raw zeros become delta, by design.
                expected=np.where(r==0,.5,r)
                visible=np.isfinite(q); expected[visible]=q[visible]
                oracles.append({'partition':label,'m2':score['m2'],
                    'm2_anchored':float(row_m2[anchored&eligible].mean()),
                    'n_no_positive_visible_anchor':info['n_no_positive_visible_anchor'],
                    'max_anchored_raw_error_vs_positive_truth_view':float(np.abs(recovered-expected)[anchored].max()),
                    'guard':numerical_diagnostics(recovered,q),'observed_exact':score['observed_restoration_exact']})
    meta,ids,hidden,truth,query,pool=validation_context
    locations=df.loc[ids,'location'].to_numpy(); pool_raw=pool[names].to_numpy(float)
    init,fb,ek,prov=initialize_crossfit(query,ids,locations,pool,names)
    poison_checks=[]
    for poison_value in (np.nan,1e200,-1e200):
        poisoned=truth.copy(); poisoned[hidden]=poison_value
        safe=np.where(hidden,np.nan,poisoned)
        init2,fb2,ek2,prov2=initialize_crossfit(safe,ids,locations,pool,names)
        for mode in ('hron_2a','aitchison_complete','no_init','mask_only'):
            a=condition_features(query,np.isfinite(query),'init' if mode in init else mode,meta['raw_log_scale'],init.get(mode),fb.get(mode),ek.get(mode))
            b=condition_features(safe,np.isfinite(safe),'init' if mode in init else mode,meta['raw_log_scale'],init2.get(mode),fb2.get(mode),ek2.get(mode))
            assert np.array_equal(a,b)
            if mode in init: assert np.array_equal(init[mode],init2[mode])
        # Held-out poison cannot enter fitted statistics: fit selection never includes Canada/Denmark.
        assert not set(pool.index)&set(ids)
        poison_checks.append({'poison':str(poison_value),'all_four_conditions_and_initializers_identical':True})
    used={d for records in prov.values() for record in records for c in record['cells'] for d in c['donor_ids']}
    assert used<=set(pool.index) and not used&set(ids)
    # Audit all training-bank provenance, not only inference queries.
    bank_trace=json.loads(gzip.decompress((OLD/'initializer_provenance_validation.json.gz').read_bytes()))
    for bank in bank_trace.values():
        for records in bank['methods'].values():
            for record in records:
                for c in record['cells']:
                    for donor in c['donor_ids']:
                        assert donor in pool.index and df.loc[donor,'location']!=record['location']
    beta,ab=schedule(); preservation=before==frozen_hashes(); assert preservation
    for filename,rows in [('training_traces.csv',traces),('loss_summary.csv',losses),('row_diagnostics.csv',row_records),('m2_contributions.csv',contributions)]:
        pd.DataFrame(rows).to_csv(OUT/filename,index=False)
    result={'status':'COMPLETE','phase':'A','no_training':True,'metrics_verified':metrics_verified,
        'checkpoint_count':checkpoints,'failure_records':failure_count,
        'environment':{'python':sys.executable,'version':sys.version,'torch':torch.__version__,
            'compatible_wheels_path':str(rework_runtime.SITE),'broken_venv_base_bypassed':True},
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'data_sha256':sha256(ROOT/'data/covariants.csv'),'frozen_files_unchanged':preservation,
        'schedule':{'terminal_alpha_bar':float(ab[-1]),'beta_max':float(beta.max()),'zero_epsilon_reverse_amplification':float(1/ab[-1].sqrt())},
        'latent_distributions':latents,'oracle_restoration':oracles,'leakage_poisoning':poison_checks,
        'validation_donors_and_all_training_banks_location_excluded':True,
        'validation_train_windows':38,'outer_train_windows':51,
        'no_historical_validation_loss_traces':True,'e5_status':'BLOCKED','finalists':[],
        'runtime_seconds':time.perf_counter()-start}
    write_json(OUT/'phase_a.json',result)
    write_hashes(OUT,[Path(__file__),ROOT/'scripts/rework_runtime.py'])
    print(json.dumps({k:result[k] for k in ('status','metrics_verified','checkpoint_count','failure_records','schedule','oracle_restoration','frozen_files_unchanged')},indent=2))


if __name__=='__main__': main()
