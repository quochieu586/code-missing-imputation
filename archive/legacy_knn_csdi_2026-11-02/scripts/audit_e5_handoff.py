"""Verify the read-only Phase A handoff, without rewriting legacy artifacts."""
import rework_runtime
from rework_runtime import ROOT
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_e5_rework import OUT, frozen_hashes, independent_metrics
from src.evaluation.research_pilot import sha256, write_json, write_hashes


def main():
    if (OUT/'verification.json').exists():
        raise FileExistsError('Completed handoff verification exists; preserve it')
    before=json.loads((OUT/'frozen_hashes_before.json').read_text())
    assert before==frozen_hashes(), 'Frozen artifacts/data/PDF changed'
    df=pd.read_csv(ROOT/'data/covariants.csv'); names=list(df.columns[3:])
    verified=[]
    for folder,metric_file,pred_file in [
        ('e3_csdi_revised','e3_metrics_validation.json','predictions_validation.json'),
        ('e3_csdi_revised','e3_metrics_test.json','predictions_test.json'),
        ('e4_pilot_revised','e4_metrics_fold0.json','predictions_fold0.json'),
    ]:
        base=ROOT/'artifacts'/folder
        scores=json.loads((base/metric_file).read_text())
        preds=json.loads((base/pred_file).read_text())
        ids=preds['row_ids']; truth=df.loc[ids,names].to_numpy(float)
        hidden=np.array(preds['hidden_eval_mask'],bool); visible=~hidden
        assert set(preds['methods'])==set(scores['methods'])
        for key,records in preds['methods'].items():
            assert [r['row_id'] for r in records]==ids
            pred=np.array([r['prediction_raw'] for r in records],float)
            assert np.isfinite(pred).all() and (pred>=0).all()
            assert np.array_equal(truth[visible],pred[visible])
            calc,_,eligible=independent_metrics(truth,pred,hidden)
            assert np.flatnonzero(eligible).tolist()==scores['methods'][key]['eligible_row_positions']
            for attr,value in calc.items():
                assert abs(value-scores['methods'][key][attr])<1e-12,(folder,key,attr)
            verified.append(f'{folder}/{metric_file}:{key}')
    # Run all preexisting tests; no cache files in protected artifacts.
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code=int(pytest.main([str(ROOT/'tests/unit'),'-q','-p','no:cacheprovider']))
    (OUT/'verification_tests.txt').write_text(buf.getvalue(),encoding='utf-8')
    assert code==0, buf.getvalue()
    assert before==frozen_hashes()
    result={'handoff_audit_passed':True,'legacy_test_exit_code':code,
        'legacy_test_output':buf.getvalue().strip(),'independently_verified_entries':len(verified),
        'metric_attributes':['m2','m3','clr_mae_all_hidden','clr_mae_nonzero_hidden','clr_mae_zero_hidden'],
        'absolute_tolerance':1e-12,'finite_nonnegative_and_observed_locking':True,
        'truth_only_common_eligibility':True,'frozen_artifacts_data_and_pdfs_unchanged':True,
        'new_diagnostic_script_unit_tests':'PENDING; must add before Phase C',
        'phase_a':'DIAGNOSTICS_EXECUTED_WITH_DOCUMENTED_LIMITS',
        'phase_b':'NOT_STARTED','phase_c':'NOT_STARTED','phase_d':'NOT_STARTED',
        'e5_status':'BLOCKED','finalists':[],
        'training_code_changes':False,'new_training_runs':0,
        'python':sys.executable,'data_sha256':sha256(ROOT/'data/covariants.csv'),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    write_json(OUT/'verification.json',result)
    write_hashes(OUT,[Path(__file__),ROOT/'scripts/diagnose_e5_rework.py',ROOT/'scripts/rework_runtime.py'])
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
