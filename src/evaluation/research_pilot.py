"""Shared, truth-defined E3/E4 evaluation and provenance contracts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def default(v):
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,np.generic): return v.item()
        raise TypeError(type(v))
    Path(path).write_text(json.dumps(value,default=default,indent=2,
                                   allow_nan=False),encoding='utf-8')


def metric_clr(x, delta=.5):
    a = np.asarray(x,dtype=np.float64)
    if not np.isfinite(a).all() or (a<0).any():
        raise ValueError('metric input must be finite nonnegative')
    logs = np.log(np.where(a==0,delta,a))
    return logs-logs.mean(axis=-1,keepdims=True)


def prevalence_bins(train, names):
    train = np.asarray(train,dtype=np.float64)
    observed = np.isfinite(train)
    rates = np.divide(((train>0)&observed).sum(0),observed.sum(0),
                      out=np.zeros(len(names),dtype=np.float64),where=observed.sum(0)>0)
    order = sorted(range(len(names)),key=lambda j:(rates[j],names[j]))
    return {k:order[a:b] for k,a,b in [('rare',0,5),('middle',5,11),('common',11,17)]},rates


def evaluate(truth,pred,hidden,query,delta,bins):
    truth=np.asarray(truth,dtype=np.float64); pred=np.asarray(pred,dtype=np.float64)
    hidden=np.asarray(hidden,dtype=bool); query=np.asarray(query,dtype=np.float64)
    if not (truth.shape==pred.shape==hidden.shape==query.shape):
        raise ValueError('evaluation shape mismatch')
    visible=np.isfinite(query)
    if np.any(hidden&visible): raise ValueError('hidden truth was passed as visible')
    # Fixed by ground truth only, NEVER exclude a row for bad/zero predictions.
    eligible=np.isfinite(truth).all(1)&(truth.sum(1)>0)&hidden.any(1)
    cells=hidden&eligible[:,None]
    exact=np.array_equal(pred[visible],query[visible])
    result={'n_hidden_cells':int(hidden.sum()),'n_scored_hidden_cells':int(cells.sum()),
        'n_affected_rows':int(eligible.sum()),
        'eligible_row_positions':np.flatnonzero(eligible).tolist(),
        'n_truth_zero_sum_excluded':int((np.isfinite(truth).all(1)&(truth.sum(1)==0)).sum()),
        'n_prediction_zero_sum_on_eligible':int((pred[eligible].sum(1)==0).sum()),
        'observed_restoration_exact':bool(exact),
        'observed_restoration_max_abs_error':(float(np.max(np.abs(pred[visible]-query[visible]))) if visible.any() else 0.) if exact else None,
        'nonfinite_cells':int((~np.isfinite(pred)).sum()),'negative_cells':int((pred<0).sum()),
        'raw_count_metrics_role':'diagnostic_only_under_protocol_U'}
    if not exact or not np.isfinite(pred).all() or (pred<0).any():
        result.update(status='FAIL_OUTPUT_CONTRACT',m2=None,clr_mae_nonzero_hidden=None,m3=None)
        return result
    z=metric_clr(truth,delta); zp=metric_clr(pred,delta)
    error=np.abs(z-zp)
    def avg(mask): return float(error[mask].mean()) if mask.any() else None
    result.update(status='OK',m2=float(((z-zp)**2).sum(1)[eligible].mean()),
        clr_mae_all_hidden=avg(cells),clr_mae_nonzero_hidden=avg(cells&(truth>0)),
        clr_mae_zero_hidden=avg(cells&(truth==0)),
        n_hidden_nonzero=int((cells&(truth>0)).sum()),n_hidden_zero=int((cells&(truth==0)).sum()),
        m3=float(np.linalg.norm(np.cov(z[eligible].T,ddof=1)-np.cov(zp[eligible].T,ddof=1),'fro')/(truth.shape[1]-1)))
    # Native J_T undefined for zero-sum predictions. Explicitly fail this
    # secondary metric instead of changing its denominator across methods.
    if (pred[eligible].sum(1)>0).all():
        p=truth[eligible]/truth[eligible].sum(1,keepdims=True)
        q=pred[eligible]/pred[eligible].sum(1,keepdims=True)
        m=.5*(p+q)
        a=np.zeros_like(p);b=np.zeros_like(q)
        pm=p>0;qm=q>0
        a[pm]=p[pm]*np.log(p[pm]/m[pm]);b[qm]=q[qm]*np.log(q[qm]/m[qm])
        result['native_jsd_diagnostic']=float((a+b).sum(1).mean())
        result['native_jsd_status']='OK'
    else:
        result.update(native_jsd_diagnostic=None,native_jsd_status='UNDEFINED_ZERO_PREDICTION_MASS')
    strata={}
    for key,ids in bins.items():
        mask=cells&np.isin(np.arange(truth.shape[1])[None,:],ids)
        nonzero=mask&(truth>0)
        strata[key]={'n_cells':int(mask.sum()),'clr_mae':avg(mask),
            'nonzero_clr_mae':avg(nonzero),
            'raw_mae_diagnostic':float(np.abs(pred-truth)[mask].mean()) if mask.any() else None}
    result['train_prevalence_strata']=strata
    return result


def numerical_diagnostics(pred,query,threshold=1000.):
    """Registered engineering scale guard; no true mass/total_sequence read."""
    pred=np.asarray(pred,dtype=np.float64);q=np.asarray(query,dtype=np.float64)
    visible=np.isfinite(q)
    anchor=np.where(visible,q,0.).sum(1)
    ratio=pred.sum(1)/np.maximum(anchor,1.)
    return {'guard_threshold':threshold,'guard_basis':'visible_sum_or_one',
        'count_sum_over_visible_sum_max':float(ratio.max()),
        'count_sum_over_visible_sum_median':float(np.median(ratio)),
        'scale_guard_row_positions':np.flatnonzero(ratio>threshold).tolist(),
        'n_scale_guard_rows':int((ratio>threshold).sum()),
        'no_positive_visible_anchor_positions':np.flatnonzero(anchor==0).tolist(),
        'scale_guard_role':'engineering_failure_flag_not_biological_validity'}


def failure_records(methods, row_ids, partition):
    """Record flags without removing rows, changing metrics or capping predictions."""
    records=[]
    for key,score in methods.items():
        diag=score.get('numerical_diagnostics',{})
        positions=diag.get('scale_guard_row_positions',[])
        if positions:
            records.append({'partition':partition,'method':key,'code':'SCALE_GUARD_EXCEEDED',
                'row_ids':[int(row_ids[i]) for i in positions],
                'threshold':diag['guard_threshold'],'basis':diag['guard_basis'],
                'role':'engineering_flag_under_Protocol_U','predictions_modified':False})
        if score['status']!='OK':
            records.append({'partition':partition,'method':key,'code':'INVALID_PREDICTION',
                'status':score['status']})
    return records


def write_hashes(out,code_paths=()):
    out=Path(out)
    files={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='file_hashes.json' and '__pycache__' not in p.parts}
    write_json(out/'file_hashes.json',{'artifact_hashes':files,
        'code_hashes':{str(p).replace('\\','/'):sha256(p) for p in code_paths}})
