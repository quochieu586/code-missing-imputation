"""Repair E3 protocol; never overwrite legacy E0--E4 artifacts.

Fixed settings are declared before test evaluation. Inner validation location
is alphabetically first complete outer-train location (Denmark). All controls
share seed, masks, DDPM schedule, windows, training and sampling randomness.
"""
from __future__ import annotations
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True  # frozen E2 code must not receive new bytecode
from src.stage_b.research_csdi import (CSDICore,condition_features,forward_noise,
    epsilon_loss,schedule,sample_latents)
from src.stage_a.initializers_e1 import build_cache,materialize,train_fallback_values
from src.evaluation.research_pilot import (evaluate,numerical_diagnostics,prevalence_bins,
    sha256,write_json,write_hashes,failure_records)

OUT=ROOT/'artifacts/e3_csdi_revised'
MASK_ID='fold0_random_r0.30_seed42'
DATA_SHA='bbb1aacdf46ebef345ca818f086d5586f4aded45475b00b6ee9bd89fc94a90ab'
CONFIG={'seed':42,'mask_seed':42,'channels':16,'layers':2,'epochs':20,
    'steps':20,'samples':2,'batch_windows':8,'window_length':8,'mask_banks':4,
    'masked_parts_per_row':5,'learning_rate':.001,'clipping':'disabled',
    'projection':'final_only','outer_passes':1,'scale_guard_threshold':1000.,
    'schedule':'quad_1e-4_to_0.5','scalar_latent_scaling':'train_RMS_std_all_coordinates',
    'checkpoint':'fixed_final_epoch_no_test_selection'}
TRANSFORMS=('clr16','ilr16','hkglr16_actual')


def load_transform_module():
    path=ROOT/'artifacts/e2_transforms/transforms.py'
    spec=importlib.util.spec_from_file_location('frozen_e2_revised',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def window_indices(frame,row_ids,length=8):
    """Nonoverlapping location windows; rows appear once, no padding.

    Break a segment when a full-label row is removed. Date gaps are retained
    with their actual date embedding; no synthetic dates are inserted.
    """
    ids=np.asarray(row_ids,dtype=np.int64);pos={int(v):i for i,v in enumerate(ids)}
    windows=[]
    for _,group in frame.groupby('location',sort=True):
        ordered=group.sort_values('date').index.to_numpy(dtype=np.int64)
        segment=[]
        for rid in ordered:
            if int(rid) not in pos:
                windows.extend([np.asarray(segment[i:i+length],dtype=np.int64)
                                for i in range(0,len(segment),length)])
                segment=[]
            else: segment.append(pos[int(rid)])
        windows.extend([np.asarray(segment[i:i+length],dtype=np.int64)
                        for i in range(0,len(segment),length)])
    if not windows or sorted(np.concatenate(windows).tolist())!=list(range(len(ids))):
        raise AssertionError('window rows must be scored exactly once')
    return windows


def masks(n,bank):
    rng=np.random.default_rng(CONFIG['mask_seed']+bank)
    visible=np.ones((n,17),dtype=bool)
    for i in range(n):visible[i,rng.choice(17,5,replace=False)]=False
    return visible


def initialize_crossfit(query,query_ids,query_locations,pool_frame,names):
    """Actual E1 algorithms; entire query location excluded from donors."""
    outputs={m:np.empty_like(query) for m in ('hron_2a','aitchison_complete')}
    fb={m:np.zeros_like(query) for m in outputs};ek={m:np.zeros_like(query) for m in outputs}
    trace={m:[] for m in outputs}
    for location in sorted(set(query_locations)):
        positions=np.flatnonzero(np.asarray(query_locations)==location)
        donors=pool_frame[pool_frame.location!=location]
        donor_ids=donors.index.to_numpy(dtype=np.int64)
        train=donors[names].to_numpy(dtype=np.float64)
        fallback=train_fallback_values(train)
        for method in outputs:
            cache,fit=build_cache(query[positions],train,donor_ids,method,4,
                                  query_ids=query_ids[positions],delta=.5)
            values,prov=materialize(query[positions],cache,fallback,4)
            outputs[method][positions]=values
            for i,p in enumerate(positions):
                fb[method][p]=[int(c['fallback']) for c in prov[i]]
                ek[method][p]=[c['effective_k'] for c in prov[i]]
                trace[method].append({'row_id':int(query_ids[p]),'location':str(location),
                    'cells':prov[i]})
                used={v for c in prov[i] for v in c['donor_ids']}
                if not used.issubset(set(donor_ids.tolist())):
                    raise AssertionError('initializer donor leak')
    visible=np.isfinite(query)
    for m,values in outputs.items():
        if not np.array_equal(query[visible],values[visible]):raise AssertionError('initializer restoration')
    return outputs,fb,ek,trace


def conditions_for_train(frame,all_frame,names,raw_scale,label):
    raw=frame[names].to_numpy(dtype=np.float64)
    ids=frame.index.to_numpy(dtype=np.int64)
    conditions={};traces={}
    for bank in range(CONFIG['mask_banks']):
        visible=masks(len(frame),bank);query=np.where(visible,raw,np.nan)
        init,fb,ek,trace=initialize_crossfit(query,ids,frame.location.to_numpy(),all_frame,names)
        for mode in ('hron_2a','aitchison_complete','no_init','mask_only'):
            condition_features_mode='init' if mode in init else mode
            conditions[(mode,bank)]=condition_features(query,visible,condition_features_mode,raw_scale,
                init.get(mode),fb.get(mode),ek.get(mode))
        traces[str(bank)]={'visibility':visible.tolist(),'methods':trace}
        print(f'{label}: actual initializer cache bank {bank+1}/4',flush=True)
    serial=json.dumps(traces,allow_nan=False).encode('utf-8')
    (OUT/f'initializer_provenance_{label}.json.gz').write_bytes(gzip.compress(serial,mtime=0))
    np.savez_compressed(OUT/f'conditions_{label}.npz',row_ids=ids,
                       **{f'{m}_bank{b}':v for (m,b),v in conditions.items()})
    return conditions


def groups_of_windows(windows,rng=None):
    groups={}
    for ids in windows:groups.setdefault(len(ids),[]).append(ids)
    batches=[]
    for length in sorted(groups):
        group=groups[length]
        order=np.arange(len(group)) if rng is None else rng.permutation(len(group))
        for i in range(0,len(order),CONFIG['batch_windows']):
            batches.append(np.stack([group[k] for k in order[i:i+CONFIG['batch_windows']]]))
    if rng is not None:rng.shuffle(batches)
    return batches


def fit_model(target,condition_banks,mode,windows,times):
    torch.manual_seed(CONFIG['seed'])
    model=CSDICore(target.shape[1],channels=16,steps=20,layers=2)
    mean=target.mean(0)
    # One scalar per transform; no coordinate whitening or rank-deficient masks.
    scale=max(float(np.sqrt(np.mean((target-mean)**2))),1e-8)
    normalized=(target-mean)/scale
    beta,alpha_bar=schedule()
    optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'])
    rng=np.random.default_rng(CONFIG['seed']);noise_rng=torch.Generator().manual_seed(CONFIG['seed'])
    trace=[];start=time.perf_counter()
    for epoch in range(CONFIG['epochs']):
        model.train();losses=[]
        cond=condition_banks[(mode,epoch%CONFIG['mask_banks'])]
        for idx in groups_of_windows(windows,rng):
            z0=torch.from_numpy(normalized[idx]);c=torch.from_numpy(cond[idx]);t=torch.from_numpy(times[idx])
            steps=torch.randint(20,(len(idx),),generator=noise_rng)
            eps=torch.randn(z0.shape,dtype=torch.float64,generator=noise_rng)
            noisy=forward_noise(z0,eps,alpha_bar[steps,None,None])
            pred=model(noisy,c,t,steps)
            # Every full LR coordinate is a supervised target; full labels are
            # known only on complete positive rows. No raw mask slicing here.
            loss=epsilon_loss(pred,eps,torch.ones_like(z0,dtype=torch.bool))
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite train loss')
            optimizer.zero_grad();loss.backward();optimizer.step()
            losses.append(float(loss.detach()))
        trace.append({'epoch':epoch+1,'epsilon_mse':float(np.mean(losses))})
    return model,mean,scale,trace,time.perf_counter()-start


def infer(model,mean,scale,condition,times,windows,transform,name):
    beta,alpha_bar=schedule();samples=np.empty((2,len(times),model.latent_dim),dtype=np.float64)
    diagnostics=[];start=time.perf_counter()
    for batch_id,idx in enumerate(groups_of_windows(windows)):
        result,diag=sample_latents(model,torch.from_numpy(condition[idx]),torch.from_numpy(times[idx]),
            beta,alpha_bar,torch.from_numpy(mean),scale,name,
            getattr(transform.spec,'references',()) or (),samples=2,seed=CONFIG['seed']+10000+batch_id)
        for b,ids in enumerate(idx):samples[:,ids,:]=result[:,b,:,:].numpy()
        diagnostics.append(diag)
    return samples,diagnostics,time.perf_counter()-start


def gate_pairs(methods):
    pairs={}
    for name in TRANSFORMS:
        no=methods[f'no_init__{name}'];mask=methods[f'mask_only__{name}']
        valid=no['status']=='OK' and mask['status']=='OK'
        pairs[name]={'no_init_m2':no.get('m2'),'mask_only_m2':mask.get('m2'),
            'passed':bool(valid and mask['m2']>no['m2'])}
    return {'criterion':'matched transform, seed, masks, rows; mask-only M2 > no-init M2',
            'pairs':pairs,'passed':all(v['passed'] for v in pairs.values())}


def run_partition(label,pool,query_frame,query,hidden,ids,names,tm,delta,test_e1=None):
    raw_all=pool[names].to_numpy(dtype=np.float64)
    train_ok=np.isfinite(raw_all).all(1)&(np.nansum(raw_all,axis=1)>0)
    eligible=pool.iloc[np.flatnonzero(train_ok)]
    if set(pool.location)&set(query_frame.location):raise AssertionError('train/heldout locations overlap')
    raw=eligible[names].to_numpy(dtype=np.float64)
    windows=window_indices(pool,eligible.index.to_numpy(),8)
    query_windows=window_indices(query_frame,ids,8)
    raw_scale=float(np.log1p(np.nanmax(raw_all)))
    origin=pd.to_datetime(pool.date).min().toordinal()
    times=(pd.to_datetime(eligible.date).map(pd.Timestamp.toordinal).to_numpy(dtype=np.float64)-origin)/14.
    qt=(pd.to_datetime(query_frame.date).map(pd.Timestamp.toordinal).to_numpy(dtype=np.float64)-origin)/14.
    cond_banks=conditions_for_train(eligible,pool,names,raw_scale,label)
    if test_e1 is None:
        init,fb,ek,prov=initialize_crossfit(query,ids,query_frame.location.to_numpy(),pool,names)
    else:
        init={};fb={};ek={};prov={}
        for m in ('hron_2a','aitchison_complete'):
            rows=test_e1['methods'][m]
            if [r['row_id'] for r in rows]!=ids.tolist():raise AssertionError('E1 row alignment')
            if not np.array_equal(np.asarray([r['hidden_eval_mask'] for r in rows]),hidden):raise AssertionError('E1 mask alignment')
            init[m]=np.asarray([r['prediction_raw'] for r in rows],dtype=np.float64)
            fb[m]=np.asarray([[c['fallback'] for c in r['cell_provenance']] for r in rows],dtype=np.float64)
            ek[m]=np.asarray([[c['effective_k'] for c in r['cell_provenance']] for r in rows],dtype=np.float64)
            prov[m]=rows
    (OUT/f'query_initializer_provenance_{label}.json.gz').write_bytes(gzip.compress(json.dumps(prov,allow_nan=False).encode(),mtime=0))
    rates=pool[names].isna().mean();refs=sorted(names,key=lambda n:(float(rates[n]),n))[:5]
    if label=='test':
        frozen_refs=json.loads((ROOT/'artifacts/e2_transforms/e2_metrics_fold0.json').read_text())['reference_sets_all_folds']['0']['actual_lowest_missing']
        if refs!=frozen_refs:raise AssertionError('HK references changed from E2')
    ref_ids=tuple(names.index(n) for n in refs)
    bins,prevalence=prevalence_bins(raw_all,names)
    truth=query_frame[names].to_numpy(dtype=np.float64)
    methods={};predictions={};failures=[]
    for m in init:
        methods[f'init_only__{m}']=evaluate(truth,init[m],hidden,query,delta,bins)
        methods[f'init_only__{m}'].update(method_type='init_only',initializer=m,source='E1_frozen' if label=='test' else 'inner_train_only_actual_initializer',numerical_diagnostics=numerical_diagnostics(init[m],query))
        predictions[f'init_only__{m}']=init[m]
    for name in TRANSFORMS:
        transform=tm.make_transform(name,17,delta,ref_ids if name.startswith('hk') else None)
        target=transform.forward(raw)
        for mode in ('hron_2a','aitchison_complete','no_init','mask_only'):
            key=f'{mode}__{name}'
            print(f'{label}: fit {key} (20 epochs)',flush=True)
            model,mean,scale,trace,train_seconds=fit_model(target,cond_banks,mode,windows,times)
            query_cond=condition_features(query,np.isfinite(query),'init' if mode in init else mode,
                raw_scale,init.get(mode),fb.get(mode),ek.get(mode))
            sample,diags,sample_seconds=infer(model,mean,scale,query_cond,qt,query_windows,transform,name)
            composition=transform.inverse(sample.mean(0))
            pred,restoration=tm.restore_observed_counts(composition,query)
            if not np.array_equal(pred[np.isfinite(query)],query[np.isfinite(query)]):raise AssertionError('observed hard assertion')
            score=evaluate(truth,pred,hidden,query,delta,bins)
            score.update(method_type='init_csdi' if mode in init else mode+'_csdi',
                initializer=mode if mode in init else None,transform=name,
                model_seed=42,sampling_seed=10042,train_seconds=train_seconds,sampling_seconds=sample_seconds,
                parameter_count=sum(p.numel() for p in model.parameters()),latent_dim=target.shape[1],
                training_trace=trace,numerical_diagnostics=numerical_diagnostics(pred,query),
                sample_diagnostics=diags,restoration={'n_no_positive_visible_anchor':restoration['n_no_positive_visible_anchor'],
                    'scales':restoration['scales'].tolist()},target_scalar_scale=scale,target_mean=mean.tolist())
            methods[key]=score;predictions[key]=pred
            torch.save({'state_dict':model.state_dict(),'mean':mean,'scalar_scale':scale,
                'config':CONFIG,'feature_order':names,'references':refs,'train_row_ids':eligible.index.to_list(),
                'raw_log_scale':raw_scale,'time_origin':origin,'transform':name,'mode':mode},OUT/f'{label}_{key}.pt')
            np.savez_compressed(OUT/f'{label}_{key}_latent_samples.npz',row_ids=ids,latent_samples=sample)
            print(f'{label}: {key} M2={score["m2"]:.6g}',flush=True)
    pred_json={'row_ids':ids.tolist(),'hidden_eval_mask':hidden.tolist(),'methods':{
        k:[{'row_id':int(rid),'prediction_raw':p[i].tolist()} for i,rid in enumerate(ids)]
        for k,p in predictions.items()}}
    write_json(OUT/f'predictions_{label}.json',pred_json)
    initialization_pairs={}
    for name in TRANSFORMS:
        no=methods[f'no_init__{name}']['m2']
        initialization_pairs[name]={m:{'m2_improvement_fraction':1-methods[f'{m}__{name}']['m2']/no,
            'exceeds_5pct':bool(methods[f'{m}__{name}']['m2']<.95*no)}
            for m in ('hron_2a','aitchison_complete')}
    metadata={'partition':label,'train_locations':sorted(set(pool.location)),
        'heldout_locations':sorted(set(query_frame.location)),
        'complete_positive_train_rows':len(raw),'excluded_complete_zero_sum_train_rows':int((np.isfinite(raw_all).all(1)&(np.nansum(raw_all,axis=1)==0)).sum()),
        'train_windows':len(windows),'query_windows':len(query_windows),'padding_rows':0,
        'overlap_rows':0,'references':refs,'prevalence':dict(zip(names,prevalence.tolist())),
        'prevalence_bins':{k:[names[j] for j in v] for k,v in bins.items()},
        'query_row_ids':ids.tolist(),'raw_log_scale':raw_scale,'time_origin':origin}
    failures=failure_records(methods,ids,label)
    result={'metadata':metadata,'methods':methods,'method_count':len(methods),'failures':failures,
        'mask_only_gate':gate_pairs(methods),'initialization_value_pairs':initialization_pairs}
    write_json(OUT/f'e3_metrics_{label}.json',result)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'e3_metrics_test.json').exists():raise FileExistsError('completed revised run exists; preserve it')
    data=ROOT/'data/covariants.csv'
    if sha256(data)!=DATA_SHA:raise AssertionError('data changed')
    frozen_paths=[data]+[p for d in ('e0_extended','e1_pilot','e2_transforms','e3_csdi','e4_pilot')
        for p in (ROOT/'artifacts'/d).glob('*') if p.is_file()]
    before={str(p.relative_to(ROOT)):sha256(p) for p in frozen_paths}
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    tm=load_transform_module();df=pd.read_csv(data)
    manifest=json.loads((ROOT/'artifacts/e0_extended/manifest.json').read_text())
    names=manifest['variant_columns'];fold=json.loads((ROOT/'artifacts/e0_extended/cv_splits.json').read_text())[0]
    pool=df.iloc[fold['train_rows']];query_frame=df.iloc[fold['test_rows']]
    delta=tm.fit_positivity_delta(pool[names].to_numpy(dtype=np.float64))
    if delta!=.5:raise AssertionError('frozen positivity delta')
    complete=pool[np.isfinite(pool[names].to_numpy()).all(1)]
    validation_location=sorted(set(complete.location))[0]
    inner_pool=pool[pool.location!=validation_location];val_frame=pool[pool.location==validation_location]
    # Refit positivity on inner train. Equality .5 is verified, not assumed.
    inner_delta=tm.fit_positivity_delta(inner_pool[names].to_numpy(dtype=np.float64))
    run={'status':'RUNNING','config':CONFIG,'mask_id':MASK_ID,'feature_order':names,
        'validation_location_rule':'alphabetically_first_complete_outer_train_location',
        'validation_location':validation_location,'frozen_input_hashes':before,
        'reference_snapshot':json.loads((ROOT/'ref/csdi_official/snapshot.json').read_text(encoding='utf-8-sig')),
        'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__},
        'legacy_runs_superseded_for_inference':['e3_csdi','e4_pilot'],'legacy_files_preserved':True}
    write_json(OUT/'manifest.json',run)
    ids=val_frame.index.to_numpy(dtype=np.int64);v=masks(len(val_frame),0)
    q=np.where(v,val_frame[names].to_numpy(dtype=np.float64),np.nan)
    validation=run_partition('validation',inner_pool,val_frame,q,~v,ids,names,tm,inner_delta)
    # Freeze decision based on inner validation before any outer-test fitting.
    write_json(OUT/'validation_decision.json',{'conditioning_gate':validation['mask_only_gate'],
        'initializer_value':validation['initialization_value_pairs'],
        'hyperparameter_selection':'none; fixed pilot config',
        'continue_test_as_exploratory_even_if_gate_fails':True})
    ids=query_frame.index.to_numpy(dtype=np.int64)
    hidden=np.load(ROOT/'artifacts/e0_extended/masks_random_cell.npz')[MASK_ID]
    q=np.where(~hidden,query_frame[names].to_numpy(dtype=np.float64),np.nan)
    e1=json.loads((ROOT/'artifacts/e1_pilot/predictions_fold0.json').read_text())
    result=run_partition('test',pool,query_frame,q,hidden,ids,names,tm,delta,e1)
    after={str(p.relative_to(ROOT)):sha256(p) for p in frozen_paths}
    if before!=after:raise AssertionError('frozen inputs modified')
    numerical_ok=all(v['status']=='OK' and v.get('numerical_diagnostics',{}).get('n_scale_guard_rows',0)==0
        for partition in (validation,result) for v in partition['methods'].values())
    run.update(status='COMPLETE',conditioning_gate_validation=validation['mask_only_gate']['passed'],
        conditioning_gate_test=result['mask_only_gate']['passed'],numerical_gate=numerical_ok,
        e5_eligible=bool(validation['mask_only_gate']['passed'] and result['mask_only_gate']['passed'] and numerical_ok),
        frozen_inputs_unchanged=True)
    write_json(OUT/'manifest.json',run)
    lines=['# E3 revised audit and pilot','',
        '**Implementation:** PyTorch float64 CSDI adaptation, temporal and feature attention, gated residual/skip blocks.',
        'Legacy E3 used a row MLP and median training context; its conditioning PASS is withdrawn for research claims.',
        '',f'Validation: {validation_location}; full positive training rows: {validation["metadata"]["complete_positive_train_rows"]}.',
        f'Outer train full positive rows: {result["metadata"]["complete_positive_train_rows"]}; windows: {result["metadata"]["train_windows"]}.',
        'Condition encoder has the same 85 input slots/16 channels for CLR, ILR and HKGLR. No raw mask is used as a latent mask.',
        'Hron/Aitchison are recomputed after each of four fixed training mask banks with the entire query location excluded from donors.',
        'Twenty epochs/20 reverse steps/two samples; same seed 42 for every branch. Checkpoint fixed at epoch 20; no test tuning.',
        'Noise is IID Gaussian; no intermediate projection or clipping; latent samples are averaged before inverse and exact raw restoration.',
        'M2/M3 use the same 101 truth-positive affected test rows, including zero-sum predictions after positivity adjustment.',
        '', '| Method | M2 | Nonzero CLR MAE | Scale guard rows |','|---|---:|---:|---:|']
    for k,v in result['methods'].items():lines.append(f'| {k} | {v["m2"]:.6g} | {v["clr_mae_nonzero_hidden"]:.6g} | {v.get("numerical_diagnostics",{}).get("n_scale_guard_rows",0)} |')
    lines += ['',f'Conditioning gate (all matched transforms): validation **{validation["mask_only_gate"]["passed"]}**, test **{result["mask_only_gate"]["passed"]}**.',
        f'Numerical gate: **{numerical_ok}**. E5 eligible on E3 prerequisites: **{run["e5_eligible"]}**.',
        'A failed efficacy gate is a negative experimental result; it is not repaired by changing a threshold or picking another test seed.',
        'Protocol U raw-count strata and scales are diagnostics only. Scale guard 1000 × visible sum (or 1) is an engineering flag, not biological validity.',
        'No confidence intervals or significance claims are made from one location. Two samples only test the sampling path.']
    (OUT/'e3_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_hashes(OUT,[Path(__file__),ROOT/'src/stage_b/research_csdi.py',ROOT/'src/evaluation/research_pilot.py'])
    print(json.dumps({k:run[k] for k in ('status','conditioning_gate_validation','conditioning_gate_test','numerical_gate','e5_eligible')},indent=2))


if __name__=='__main__':
    try: main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        write_json(OUT/'run_failure.json',{'status':'BLOCKED_IMPLEMENTATION','type':type(exc).__name__,'reason':str(exc)})
        raise
