"""Downstream contracts and baseline leakage tests for the revised pilot."""
import copy
import numpy as np
import pytest
from scripts.run_e4_revised import (efficacy_gate,verify_e3_signature,
    validate_contract,temporal_baseline,mean_baseline)
from src.evaluation.research_pilot import failure_records


def score(m2,mae,kind='init_csdi'):
    return dict(m2=m2,clr_mae_nonzero_hidden=mae,status='OK',
        method_type=kind,initializer='hron_2a',eligible_row_positions=[0,1])


def test_efficacy_gate_is_strict_and_checks_fidelity_and_method_type():
    methods={'init_only__hron_2a':score(100.,2.,'init_only'),
        'exact_10pct':score(90.,1.),'deteriorated':score(80.,3.),
        'no_init':score(1.,1.,'no_init_csdi')}
    gate=efficacy_gate(methods)
    assert not gate['passed'] and len(gate['pairs'])==2
    methods['valid']=score(89.,2.)
    assert efficacy_gate(methods)['pairs']['valid']['passed']
    methods['valid']['eligible_row_positions']=[1]
    with pytest.raises(AssertionError,match='different truth rows'):efficacy_gate(methods)


@pytest.mark.parametrize('change',['mask','rows','feature_order','unfinished'])
def test_e3_signature_rejects_stale_evaluation(change):
    manifest={'status':'COMPLETE','mask_id':'fold0_random_r0.30_seed42','feature_order':['a']}
    prediction={'row_ids':[12,15],'hidden_eval_mask':[[True],[False]],
        'methods':{'example':[{'row_id':12},{'row_id':15}]}}
    if change=='mask':prediction['hidden_eval_mask'][0][0]=False
    if change=='rows':prediction['methods']['example'].reverse()
    if change=='feature_order':manifest['feature_order']=['b']
    if change=='unfinished':manifest['status']='STARTED'
    with pytest.raises(ValueError):
        verify_e3_signature(manifest,prediction,[12,15],np.array([[True],[False]]),['a'])


def test_nested_output_contract_rejects_original_empty_container_bug():
    methods={'example':score(1.,1.)}
    metrics={'method_count':1,'methods':methods};predictions={'methods':{'example':[]}}
    validate_contract(metrics,predictions)
    broken=copy.deepcopy(metrics);broken['example']=broken['methods'].pop('example')
    with pytest.raises(AssertionError):validate_contract(broken,predictions)
    with pytest.raises(AssertionError):validate_contract(metrics,{'methods':{}})


def test_baselines_use_visible_values_only_and_label_leading_nocb():
    raw=np.array([[999.,0.],[2.,0.],[999.,0.],[6.,0.]])
    hidden=np.array([[True,False],[False,False],[True,False],[False,False]])
    fallback=np.array([3.,4.]);times=np.arange(4,dtype=np.float64)
    for method in ('linear','locf_nocb','train_mean_scaled','row_positive_mean_sensitivity'):
        def apply(truth):
            query=np.where(hidden,np.nan,truth)
            if method in ('linear','locf_nocb'):
                return temporal_baseline(query,times,fallback,method)
            return mean_baseline(query,fallback,method),{}
        a,diag=apply(raw);poison=raw.copy();poison[hidden]=1e12;b,_=apply(poison)
        assert np.array_equal(a,b) and np.array_equal(a[~hidden],raw[~hidden])
        if method=='locf_nocb':assert diag['leading_nocb_cells']==1 and a[0,0]==2.
    q=np.full((4,2),np.nan)
    p,diag=temporal_baseline(q,times,fallback,'linear')
    assert np.array_equal(p,np.broadcast_to(fallback,(4,2))) and diag['fallback_cells']==8
    with pytest.raises(ValueError,match='strictly increasing'):
        temporal_baseline(q,np.array([0.,0.,1.,2.]),fallback,'linear')


def test_failure_log_uses_stable_original_ids_without_dropping_rows():
    methods={'example':score(1.,1.)}
    methods['example']['numerical_diagnostics']={'scale_guard_row_positions':[1],
        'guard_threshold':1000.,'guard_basis':'visible_sum_or_one'}
    records=failure_records(methods,[14,103],'test')
    assert records[0]['row_ids']==[103] and not records[0]['predictions_modified']
    assert methods['example']['eligible_row_positions']==[0,1]
