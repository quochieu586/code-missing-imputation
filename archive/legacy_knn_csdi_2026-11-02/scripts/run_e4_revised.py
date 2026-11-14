"""E4 comparison consumes only verified revised E3; nested schema v2."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation.research_pilot import (evaluate,prevalence_bins,numerical_diagnostics,
    sha256,write_json,write_hashes,failure_records)
E3=ROOT/'artifacts/e3_csdi_revised';OUT=ROOT/'artifacts/e4_pilot_revised'


def temporal_baseline(query,times,fallback,method):
    q=np.asarray(query,dtype=np.float64);t=np.asarray(times,dtype=np.float64)
    if np.any(np.diff(t)<=0):raise ValueError('temporal baseline dates must be strictly increasing')
    result=q.copy();fallback_cells=0;leading_nocb=0;negative_extrapolations=0
    for j in range(q.shape[1]):
        obs=np.isfinite(q[:,j]);idx=np.flatnonzero(obs)
        if not len(idx):
            result[:,j]=fallback[j];fallback_cells+=len(q);continue
        if method=='linear':
            result[~obs,j]=np.interp(t[~obs],t[idx],q[idx,j])
            if len(idx)>1:
                left=(~obs)&(t<t[idx[0]]);right=(~obs)&(t>t[idx[-1]])
                slope0=(q[idx[1],j]-q[idx[0],j])/(t[idx[1]]-t[idx[0]])
                slope1=(q[idx[-1],j]-q[idx[-2],j])/(t[idx[-1]]-t[idx[-2]])
                result[left,j]=q[idx[0],j]+slope0*(t[left]-t[idx[0]])
                result[right,j]=q[idx[-1],j]+slope1*(t[right]-t[idx[-1]])
        elif method=='locf_nocb':
            last=q[idx[0],j]
            leading_nocb+=int(idx[0])
            for i in range(len(q)):
                if obs[i]:last=q[i,j]
                else:result[i,j]=last
        else:raise ValueError(method)
    negative_extrapolations=int((result<0).sum())
    result=np.maximum(result,0.)
    result[np.isfinite(q)]=q[np.isfinite(q)]
    return result,{'fallback_cells':fallback_cells,'leading_nocb_cells':leading_nocb,
        'negative_extrapolations_clamped':negative_extrapolations,
        'clamped_fraction':negative_extrapolations/max(int((~np.isfinite(q)).sum()),1),
        'scenario':'random_cell_only; whole_variant uses train fallback if no trajectory exists'}


def mean_baseline(query,train_mean,method):
    q=np.asarray(query,dtype=np.float64);visible=np.isfinite(q);out=q.copy()
    if method=='train_mean_scaled':
        # Preserve a raw zero mean; positivity is confined to LR metric view.
        composition=train_mean/train_mean.sum()
        for i in range(len(q)):
            pos=visible[i]&(q[i]>0)&(composition>0)
            scale=float(np.median(q[i,pos]/composition[pos])) if pos.any() else 1.
            out[i,~visible[i]]=composition[~visible[i]]*scale
    elif method=='row_positive_mean_sensitivity':
        for i in range(len(q)):
            pos=visible[i]&(q[i]>0)
            value=float(q[i,pos].mean()) if pos.any() else float(train_mean.mean())
            out[i,~visible[i]]=value
    else:raise ValueError(method)
    return out


def efficacy_gate(methods):
    pairs={}
    for key,score in methods.items():
        if score.get('method_type')!='init_csdi':continue
        initializer=score['initializer'];base=methods[f'init_only__{initializer}']
        if score['eligible_row_positions']!=base['eligible_row_positions']:
            raise AssertionError('gate compared different truth rows')
        valid=score['status']=='OK' and base['status']=='OK'
        gain=1-score['m2']/base['m2'] if valid and base['m2']>0 else None
        fidelity=bool(valid and score['clr_mae_nonzero_hidden']<=base['clr_mae_nonzero_hidden'])
        pairs[key]={'initializer_m2':base['m2'],'diffusion_m2':score['m2'],
            'm2_improvement_fraction':gain,'m2_over_10pct':bool(gain is not None and gain>.10),
            'nonzero_mae_no_deterioration':fidelity,
            'passed':bool(gain is not None and gain>.10 and fidelity)}
    return {'criterion':'matched Init+CSDI improves M2 strictly >10% without nonzero CLR MAE deterioration',
        'pairs':pairs,'passed':any(v['passed'] for v in pairs.values())}


def verify_e3_signature(manifest,predictions,fold_ids,mask,names):
    if manifest.get('status')!='COMPLETE':raise ValueError('E3 run incomplete')
    if manifest['mask_id']!='fold0_random_r0.30_seed42' or manifest['feature_order']!=names:
        raise ValueError('E3 evaluation signature mismatch')
    if predictions['row_ids']!=list(fold_ids) or not np.array_equal(predictions['hidden_eval_mask'],mask):
        raise ValueError('E3 prediction row/mask mismatch')
    for key,rows in predictions['methods'].items():
        if [row['row_id'] for row in rows]!=list(fold_ids):raise ValueError('method row order mismatch '+key)


def validate_contract(metrics,predictions):
    if not metrics['methods'] or metrics['method_count']!=len(metrics['methods']):
        raise AssertionError('empty or miscounted methods container')
    if set(metrics['methods'])!=set(predictions['methods']):
        raise AssertionError('metrics/predictions method mismatch')
    if any('method_type' in value for key,value in metrics.items()
           if key!='methods' and isinstance(value,dict)):
        raise AssertionError('method written at top level')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'e4_metrics_fold0.json').exists():raise FileExistsError('completed revised E4 exists; preserve it')
    run=json.loads((E3/'manifest.json').read_text())
    scores=json.loads((E3/'e3_metrics_test.json').read_text())
    predictions=json.loads((E3/'predictions_test.json').read_text())
    validation=json.loads((E3/'e3_metrics_validation.json').read_text())
    df=pd.read_csv(ROOT/'data/covariants.csv')
    fold=json.loads((ROOT/'artifacts/e0_extended/cv_splits.json').read_text())[0]
    names=run['feature_order'];ids=fold['test_rows'];train=df.iloc[fold['train_rows']]
    test=df.iloc[ids];truth=test[names].to_numpy(dtype=np.float64)
    mask=np.load(ROOT/'artifacts/e0_extended/masks_random_cell.npz')[run['mask_id']]
    verify_e3_signature(run,predictions,ids,mask,names)
    for rel,want in run['frozen_input_hashes'].items():
        if sha256(ROOT/rel)!=want:raise AssertionError('frozen input hash changed '+rel)
    verification=json.loads((E3/'verification.json').read_text())
    if not verification['passed']:raise AssertionError('E3 implementation verification failed')
    hashes=json.loads((E3/'file_hashes.json').read_text())['artifact_hashes']
    for rel,want in hashes.items():
        if sha256(E3/rel)!=want:raise AssertionError('E3 hash mismatch '+rel)
    for rel,want in json.loads((E3/'file_hashes.json').read_text())['code_hashes'].items():
        if sha256(Path(rel))!=want:raise AssertionError('E3 audited code hash mismatch '+rel)
    frozen_files=[p for d in ('e0_extended','e1_pilot','e2_transforms','e3_csdi','e4_pilot','e3_csdi_revised')
        for p in (ROOT/'artifacts'/d).glob('*') if p.is_file()]
    before={str(p.relative_to(ROOT)):sha256(p) for p in frozen_files}
    q=np.where(~mask,truth,np.nan);times=pd.to_datetime(test.date).map(pd.Timestamp.toordinal).to_numpy(dtype=np.float64)
    train_raw=train[names].to_numpy(dtype=np.float64);bins,prevalence=prevalence_bins(train_raw,names)
    methods={};pred_out={}
    for key,rows in predictions['methods'].items():
        p=np.asarray([r['prediction_raw'] for r in rows],dtype=np.float64)
        score=evaluate(truth,p,mask,q,.5,bins)
        for attr in ('method_type','initializer','transform','train_seconds','sampling_seconds','parameter_count','model_seed','numerical_diagnostics'):
            if attr in scores['methods'][key]:score[attr]=scores['methods'][key][attr]
        score['source']='revised_E3_verified_signature'
        # Guard a stale or differently implemented E3 metric contract.
        if abs(score['m2']-scores['methods'][key]['m2'])>1e-10:raise AssertionError('metric mismatch '+key)
        methods[key]=score;pred_out[key]=rows
    means=np.nanmean(train_raw,axis=0)
    for key in ('linear','locf_nocb','train_mean_scaled','row_positive_mean_sensitivity'):
        start=time.perf_counter()
        if key in ('linear','locf_nocb'):
            p,diag=temporal_baseline(q,times,means,key)
        else:
            p=mean_baseline(q,means,key);diag={}
        elapsed=time.perf_counter()-start
        if not np.array_equal(p[~mask],q[~mask]):raise AssertionError('baseline exact restoration '+key)
        score=evaluate(truth,p,mask,q,.5,bins)
        score.update(method_type='classical_baseline',source=key,runtime_seconds=elapsed,
            baseline_diagnostics=diag,numerical_diagnostics=numerical_diagnostics(p,q),
            baseline_family='Mean' if 'mean' in key else key)
        methods[key]=score
        pred_out[key]=[{'row_id':int(rid),'prediction_raw':p[i].tolist()} for i,rid in enumerate(ids)]
    # A common denominator is asserted for every scored method.
    row_sets={tuple(v['eligible_row_positions']) for v in methods.values()}
    if len(row_sets)!=1:raise AssertionError('different scoring rows')
    efficacy=efficacy_gate(methods)
    conditioning={'validation':validation['mask_only_gate'],'test':scores['mask_only_gate'],
        'passed':bool(validation['mask_only_gate']['passed'] and scores['mask_only_gate']['passed'])}
    restoration=all(v['observed_restoration_exact'] for v in methods.values())
    blockers=[]
    if not conditioning['passed']:blockers.append('E3_CONDITIONING_GATE_FAILED')
    if not run['numerical_gate']:blockers.append('E3_SCALE_OR_NUMERICAL_GATE_FAILED')
    if not restoration:blockers.append('OBSERVED_RESTORATION_FAILED')
    # Finalists can only come from inner validation and all prerequisite gates.
    val_efficacy=efficacy_gate(validation['methods'])
    candidates=[k for k,v in val_efficacy['pairs'].items() if v['passed'] and
        validation['initialization_value_pairs'][k.split('__')[1]][k.split('__')[0]]['exceeds_5pct']]
    finalists=sorted(candidates,key=lambda k:validation['methods'][k]['m2'])[:3] if not blockers and efficacy['passed'] else []
    result={'schema_version':2,'metadata':{'fold_id':0,'test_location':'Canada','protocol':'U',
        'mask_id':run['mask_id'],'feature_order':names,'delta':.5,'model_seed':42,'mask_seed':42,
        'test_row_ids':ids,'eligible_row_ids':[ids[i] for i in next(iter(row_sets))],
        'n_hidden_cells':int(mask.sum()),'n_scored_hidden_cells':next(iter(methods.values()))['n_scored_hidden_cells'],
        'e3_manifest_hash':sha256(E3/'manifest.json'),'e3_predictions_hash':sha256(E3/'predictions_test.json'),
        'prevalence':dict(zip(names,prevalence.tolist())),
        'prevalence_bins':{k:[names[j] for j in v] for k,v in bins.items()},
        'legacy_outputs_preserved':True,'runtime_note':'reused init-only fitting time unavailable, not invented'},
        'methods':methods,'method_count':len(methods),
        'gates':{'conditioning':conditioning,'observed_restoration':{'passed':restoration},
                 'diffusion_efficacy_test':efficacy,'diffusion_efficacy_validation':val_efficacy},
        'blockers':blockers,'e5_status':'BLOCKED' if blockers else ('FAIL_EFFICACY' if not efficacy['passed'] else ('NO_VALIDATION_FINALIST' if not finalists else 'READY')),
        'finalists':finalists,'failures':validation['failures']+failure_records(methods,ids,'test')}
    predictions_output={'metadata':result['metadata'],'row_ids':ids,
               'hidden_eval_mask':mask.tolist(),'methods':pred_out}
    validate_contract(result,predictions_output)
    write_json(OUT/'e4_metrics_fold0.json',result)
    write_json(OUT/'predictions_fold0.json',predictions_output)
    after={str(p.relative_to(ROOT)):sha256(p) for p in frozen_files}
    if before!=after:raise AssertionError('upstream artifacts modified')
    lines=['# E4 revised comparison','',f'**E5 status: {result["e5_status"]}**. Blockers: {", ".join(blockers) or "none"}.',
        '', 'Fold 0 Canada; frozen random-cell r=0.30/seed 42; same 101 truth-positive rows for all methods.',
        'Old E3 MLP outputs are preserved for audit, excluded from this revised experiment.',
        'Three matched mask-only controls are required because transform-specific noise/learning can differ; this gives 18 concrete methods.',
        'All schema entries are under `methods`; `method_count == len(methods)` is enforced.',
        '', '| Method | M2 | Nonzero CLR MAE | M3 | Exact observed |','|---|---:|---:|---:|---|']
    for k,v in methods.items():lines.append(f'| {k} | {v["m2"]:.6g} | {v["clr_mae_nonzero_hidden"]:.6g} | {v["m3"]:.6g} | {v["observed_restoration_exact"]} |')
    lines += ['',f'Diffusion efficacy >10% + no nonzero deterioration: **{efficacy["passed"]}**.',
        f'Conditioning gate (every matched transform, validation and test): **{conditioning["passed"]}**.',
        f'Exact restoration: **{restoration}**.',f'Finalists: **{finalists}**.',
        '', 'Raw-count MAE and scale diagnostics under Protocol U are not selection criteria. No true total is used as a scale anchor.',
        'Linear/LOCF can use neighboring visible dates of the same variant in this random-cell scenario. Their advantage does not establish performance for whole-variant missingness.',
        'LOCF is explicitly labelled LOCF+NOCB for leading gaps. Linear negative edge extrapolation clamp counts are recorded.',
        'Train prevalence uses observed nonmissing cells as denominator; rare/common sets are frozen from outer train.',
        'Native JSD is a secondary diagnostic; zero-sum predictions produce an explicit undefined status rather than a changed denominator.',
        'Finalists are selected on inner validation only, subject to E3/E4 prerequisites. No significance claim follows from one test location.']
    (OUT/'e4_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(OUT/'upstream_integrity.json',{'unchanged':True,'hashes':before})
    write_hashes(OUT,[Path(__file__),ROOT/'src/evaluation/research_pilot.py'])
    print(json.dumps({'methods':len(methods),'e5_status':result['e5_status'],
                      'blockers':blockers,'efficacy_passed':efficacy['passed'],'finalists':finalists},indent=2))


if __name__=='__main__':main()
