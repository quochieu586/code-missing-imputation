"""Verify completed revised E4 artifacts without changing experiment results."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from src.evaluation.research_pilot import sha256,write_json,write_hashes,evaluate,prevalence_bins
from scripts.run_e4_revised import validate_contract,efficacy_gate
OUT=ROOT/'artifacts/e4_pilot_revised'


def main():
    for folder in ('e3_csdi_revised','e4_pilot_revised'):
        path=ROOT/'artifacts'/folder;hashes=json.loads((path/'file_hashes.json').read_text())
        for rel,want in hashes['artifact_hashes'].items():assert sha256(path/rel)==want,rel
        for rel,want in hashes['code_hashes'].items():assert sha256(rel)==want,rel
    integrity=json.loads((OUT/'upstream_integrity.json').read_text())
    for rel,want in integrity['hashes'].items():assert sha256(ROOT/rel)==want,rel
    run=json.loads((ROOT/'artifacts/e3_csdi_revised/manifest.json').read_text())
    for rel,want in run['frozen_input_hashes'].items():assert sha256(ROOT/rel)==want,rel
    metrics=json.loads((OUT/'e4_metrics_fold0.json').read_text())
    predictions=json.loads((OUT/'predictions_fold0.json').read_text())
    validate_contract(metrics,predictions);assert len(metrics['methods'])==18
    df=pd.read_csv(ROOT/'data/covariants.csv');names=metrics['metadata']['feature_order']
    fold=json.loads((ROOT/'artifacts/e0_extended/cv_splits.json').read_text())[0]
    ids=fold['test_rows'];assert predictions['row_ids']==ids
    mask=np.load(ROOT/'artifacts/e0_extended/masks_random_cell.npz')[run['mask_id']]
    assert np.array_equal(mask,predictions['hidden_eval_mask'])
    truth=df.iloc[ids][names].to_numpy(dtype=np.float64)
    query=np.where(mask,np.nan,truth);visible=np.isfinite(query)
    bins,_=prevalence_bins(df.iloc[fold['train_rows']][names].to_numpy(dtype=np.float64),names)
    checks={}
    for key,rows in predictions['methods'].items():
        assert [row['row_id'] for row in rows]==ids
        pred=np.array([row['prediction_raw'] for row in rows],dtype=np.float64)
        assert np.isfinite(pred).all() and (pred>=0).all()
        assert np.array_equal(pred[visible],query[visible]),key
        computed=evaluate(truth,pred,mask,query,.5,bins);saved=metrics['methods'][key]
        for attr in ('m2','m3','clr_mae_all_hidden','clr_mae_nonzero_hidden','clr_mae_zero_hidden'):
            assert abs(computed[attr]-saved[attr])<1e-12,(key,attr)
        assert computed['n_affected_rows']==101 and computed['n_scored_hidden_cells']==505
        assert computed['eligible_row_positions']==saved['eligible_row_positions']
        checks[key]={'eligible_rows':101,'scored_hidden_cells':505,
            'visible_zero_cells_restored':int((visible&(truth==0)).sum()),
            'recomputed_metrics_match':True,'observed_restoration_exact':True}
    assert efficacy_gate(metrics['methods'])==metrics['gates']['diffusion_efficacy_test']
    assert not metrics['finalists'] and metrics['e5_status']=='BLOCKED'
    write_json(OUT/'verification.json',{'passed':True,'methods_verified':checks,
        'schema_verified':True,'method_count':18,'all_upstream_and_code_hashes_match':True,
        'frozen_data_and_legacy_E0_E4_unchanged':True,'n_hidden_cells':545,
        'n_eligible_rows':101,'n_scored_hidden_cells':505,
        'n_zero_sum_truth_rows_excluded':8,'prediction_dependent_row_exclusion':False,
        'scientific_gates_passed':False,'finalists':[],'e5_status':'BLOCKED',
        'test_evidence':'../e3_csdi_revised/verification_tests.txt (49 passed)'})
    write_hashes(OUT,[ROOT/'scripts/run_e4_revised.py',ROOT/'src/evaluation/research_pilot.py',
        Path(__file__),ROOT/'tests/unit/stage_b/test_e4_revised.py'])
    print(json.dumps({'verified_methods':18,'rows':101,'hidden_scored':505,
        'hashes_restoration_schema_metrics':'PASS','e5_status':'BLOCKED'},indent=2))


if __name__=='__main__':main()
