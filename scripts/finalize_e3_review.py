"""Add audit evidence to a finished E3 run; NEVER retrain or change predictions."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from src.evaluation.research_pilot import (sha256,write_json,write_hashes,
    numerical_diagnostics,failure_records)
OUT=ROOT/'artifacts/e3_csdi_revised'


def main():
    old_hashes=json.loads((OUT/'file_hashes.json').read_text())
    for rel,want in old_hashes['artifact_hashes'].items():
        assert sha256(OUT/rel)==want,rel
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['status']=='COMPLETE'
    for rel,want in manifest['frozen_input_hashes'].items():assert sha256(ROOT/rel)==want,rel
    for record in manifest['reference_snapshot']['files']:
        assert sha256(record['Path'])==record['Hash'].lower(),record['Path']
    completed=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','tests'],
        cwd=ROOT,capture_output=True,text=True)
    (OUT/'verification_tests.txt').write_text(completed.stdout+completed.stderr,encoding='utf-8')
    if completed.returncode:raise AssertionError('tests failed; see verification_tests.txt')
    df=pd.read_csv(ROOT/'data/covariants.csv');names=manifest['feature_order']
    immutable={rel:want for rel,want in old_hashes['artifact_hashes'].items()
        if rel.startswith('predictions_') or rel.endswith(('.pt','.npz','.gz'))}
    all_metrics=[]
    for partition in ('validation','test'):
        path=OUT/f'e3_metrics_{partition}.json';metrics=json.loads(path.read_text())
        prediction=json.loads((OUT/f'predictions_{partition}.json').read_text())
        ids=prediction['row_ids'];hidden=np.array(prediction['hidden_eval_mask'],dtype=bool)
        query=np.where(hidden,np.nan,df.iloc[ids][names].to_numpy(dtype=np.float64))
        for key,score in metrics['methods'].items():
            pred=np.array([row['prediction_raw'] for row in prediction['methods'][key]])
            assert np.array_equal(pred[np.isfinite(query)],query[np.isfinite(query)]),key
            assert [row['row_id'] for row in prediction['methods'][key]]==ids,key
            score['numerical_diagnostics']=numerical_diagnostics(pred,query)
        metrics['failures']=failure_records(metrics['methods'],ids,partition)
        write_json(path,metrics);all_metrics.append(metrics)
    numerical_ok=all(not item['failures'] for item in all_metrics)
    manifest['numerical_gate']=numerical_ok
    manifest['e5_eligible']=bool(numerical_ok and manifest['conditioning_gate_validation'] and manifest['conditioning_gate_test'])
    manifest.setdefault('training_code_hashes_before_report_finalization',old_hashes['code_hashes'])
    manifest['postrun_finalization']={'prediction_or_checkpoint_changes':False,
        'changes':['explicit failure logs','init-only scale diagnostics','verification evidence'],
        'script':'scripts/finalize_e3_review.py'}
    write_json(OUT/'manifest.json',manifest)
    for rel,want in immutable.items():assert sha256(OUT/rel)==want,rel
    write_json(OUT/'verification.json',{'passed':True,'test_exit_code':completed.returncode,
        'test_output':completed.stdout.strip(),'frozen_legacy_inputs_unchanged':True,
        'official_snapshot_hashes_verified':True,'predictions_checkpoints_caches_unchanged':True,
        'exact_observed_restoration_all_methods':True,'stable_row_ids_all_methods':True,
        'training_and_report_code_hashes_separate':True,
        'scientific_gates_passed':False,'meaning':'implementation checks pass; experiment gates can fail'})
    report=OUT/'e3_report.md'
    lines=report.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        for key,score in all_metrics[1]['methods'].items():
            if line.startswith('| '+key+' |'):
                lines[i]=f'| {key} | {score["m2"]:.6g} | {score["clr_mae_nonzero_hidden"]:.6g} | {score["numerical_diagnostics"]["n_scale_guard_rows"]} |'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with report.open('a',encoding='utf-8') as f:
        f.write('\nAudit finalization: explicit per-method scale failure logs include original CSV row IDs, in validation and test JSON. All init-only outputs also receive scale diagnostics. Verification details: `verification.json` and `verification_tests.txt`. Predictions, samples, checkpoints and cached initializers remain byte-for-byte unchanged.\n')
    write_hashes(OUT,[ROOT/'scripts/run_e3_revised.py',ROOT/'src/stage_b/research_csdi.py',
        ROOT/'src/evaluation/research_pilot.py',Path(__file__),
        ROOT/'tests/unit/stage_b/test_e3_revised.py',ROOT/'tests/unit/stage_b/test_e4_revised.py'])
    print(completed.stdout.strip())
    print(json.dumps({'implementation_verified':True,'e5_eligible':manifest['e5_eligible'],
        'failure_records':sum(len(m['failures']) for m in all_metrics)},indent=2))


if __name__=='__main__':main()
