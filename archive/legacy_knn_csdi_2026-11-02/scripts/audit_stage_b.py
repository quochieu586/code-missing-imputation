"""Recompute metrics and verify exported single-pass artifacts independently."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.core.transforms import clr
from src.evaluation.metrics import evaluate


def main():
    folder = Path(sys.argv[1] if len(sys.argv)>1 else 'artifacts/stage_b_single_pass_final')
    record = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    arrays = np.load(folder/'audit_arrays.npz')
    raw = pd.read_csv('data/covariants.csv')
    stage_a = pd.read_csv('data/processed/X1_knn_imputed.csv')
    final = pd.read_csv(folder/'X2_diffusion_imputed.csv')
    cols = [c for c in raw if c not in ['location','date','total_sequence']]
    x = final[cols].to_numpy(float)
    m0 = arrays['M0']
    assert np.array_equal(m0,raw[cols].notna().to_numpy())
    assert np.array_equal(x[m0],stage_a[cols].to_numpy(float)[m0])
    assert np.isfinite(x).all() and (x>0).all()
    assert np.max(np.abs(clr(x).sum(-1)))<1e-8
    train, test = arrays['train_rows'], arrays['test_rows']
    assert not (train & test).any()
    assert set(raw.loc[train,'location']).isdisjoint(raw.loc[test,'location'])
    val, target = arrays['validation_mask'],arrays['test_mask']
    assert not (val & target).any()
    assert not (arrays['available'] & (val | target)).any()
    assert not val[test].any() and not target[train].any()
    assert np.array_equal(arrays['evaluation_X1'][arrays['available']],arrays['truth'][arrays['available']])
    assert np.array_equal(arrays['evaluation_X2'][arrays['available']],arrays['truth'][arrays['available']])
    evaluated = test & m0.all(1)
    counts = raw[cols].to_numpy(float)
    total = raw.total_sequence.to_numpy(float)[:,None]
    zero = (counts==0) | ((counts>=0) & (total>0) & (counts/np.maximum(total,1)<1e-4))
    for name, key in [('knn_aitchison','evaluation_X1'),('knn_csdi_one_pass','evaluation_X2')]:
        metrics = evaluate(arrays['truth'][evaluated],arrays[key][evaluated],target[evaluated],zero[evaluated])
        for metric, value in metrics.items():
            saved = record['metrics'][name][metric]
            assert (value is None and saved is None) or np.isclose(value,saved,rtol=1e-12,atol=1e-12),metric
    assert record['n_iter']==record['max_iter']==1 and not record['converged']
    for stem, key in [('Z1_clr','production_Z1'),('Z2_clr','production_Z2')]:
        z = pd.read_csv(folder/f'{stem}.csv')[cols].to_numpy(float)
        assert np.allclose(z,arrays[key],atol=1e-12,rtol=1e-12)
    audit = dict(passed=True,rows=len(x),features=len(cols),
                 original_missing_cells=int((~m0).sum()),observed_locked=int(m0.sum()),
                 evaluated_subjects=raw.loc[evaluated,'location'].unique().tolist(),
                 evaluated_rows=int(evaluated.sum()),evaluated_cells=int(target.sum()),
                 max_abs_clr_row_sum=float(np.max(np.abs(clr(x).sum(-1)))),
                 note='Metrics recomputed from saved arrays; not a full handoff acceptance or paper replication.')
    (folder/'audit_verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
