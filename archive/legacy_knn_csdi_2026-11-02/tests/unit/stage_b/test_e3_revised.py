"""Research gates: official numerical core, leakage, representation and rows."""
import ast
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.stage_b.research_csdi import (schedule,forward_noise,reverse_step,epsilon_loss,
    CSDICore,condition_features,project_final,sample_latents)
from src.evaluation.research_pilot import evaluate,prevalence_bins

ROOT=Path(__file__).resolve().parents[3]


def test_official_csdi_forward_loss_and_reverse_parity():
    # Execute the ORIGINAL methods, extracted unmodified from the pinned
    # snapshot. The denoiser is shared; parity concerns DDPM/loss, not weights.
    source=(ROOT/'ref/csdi_official/main_model.py').read_text()
    tree=ast.parse(source)
    base=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='CSDI_base')
    funcs=[n for n in base.body if isinstance(n,ast.FunctionDef) and n.name in
           ('calc_loss','impute','set_input_to_diffmodel')]
    cls=ast.ClassDef(name='Reference',bases=[ast.Attribute(ast.Name('torch',ast.Load()),'nn',ast.Load())],keywords=[],body=funcs,decorator_list=[])
    # Build class with nn.Module base; no dependency stub/reference edits.
    cls.bases=[ast.Attribute(ast.Attribute(ast.Name('torch',ast.Load()),'nn',ast.Load()),'Module',ast.Load())]
    module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    ns={'torch':torch,'np':np};exec(compile(module,'official_csdi_methods','exec'),ns)
    ref=ns['Reference']();ref.device='cpu';ref.is_unconditional=False
    beta,ab=schedule(4);ref.beta=beta.numpy();ref.alpha=ab.numpy();ref.alpha_hat=(1-beta).numpy()
    ref.alpha_torch=ab[:,None,None];ref.num_steps=4
    class Oracle(torch.nn.Module):
        def forward(self,inp,side,step):return .17*inp[:,1]+.03
    ref.diffmodel=Oracle()
    x=torch.arange(24,dtype=torch.float64).reshape(2,3,4)/7.
    cond=torch.zeros_like(x);observed=torch.ones_like(x);side=torch.zeros(2,1,3,4,dtype=torch.float64)
    old_dtype=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(17);noise=torch.randn_like(x)
        noisy=forward_noise(x,noise,ab[2]);pred=.17*noisy+.03
        expected=epsilon_loss(pred,noise,torch.ones_like(x,dtype=torch.bool))
        torch.manual_seed(17);actual=ref.calc_loss(x,cond,observed,side,0,set_t=2)
        assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
        torch.manual_seed(23);current=torch.randn_like(x)
        for t in reversed(range(4)):
            eps=.17*current+.03
            eta=torch.randn_like(current) if t else None
            current=reverse_step(current,eps,t,beta,ab,eta)
        torch.manual_seed(23);official=ref.impute(x,cond,side,1)[:,0]
        assert torch.allclose(current,official,atol=1e-12,rtol=1e-12)
    finally:torch.set_default_dtype(old_dtype)


def test_terminal_step_ignores_noise():
    beta,ab=schedule();x=torch.tensor([1.,-2.],dtype=torch.float64);eps=x*.1
    a=reverse_step(x,eps,0,beta,ab,torch.full_like(x,float('nan')))
    b=reverse_step(x,eps,0,beta,ab)
    assert torch.equal(a,b)
    assert ab[-1]<.02  # old schedule ended at .633 and was not Gaussian


@pytest.mark.parametrize('mode',['init','no_init','mask_only'])
def test_hidden_poison_does_not_change_condition(mode):
    raw=np.arange(34,dtype=np.float64).reshape(2,17);mask=np.ones_like(raw,dtype=bool);mask[:,[1,4,7]]=False
    init=np.where(mask,raw,3.);fb=np.zeros_like(raw);ek=np.where(mask,0.,4.)
    expected=condition_features(raw,mask,mode,10.,init,fb,ek)
    for val in (float('nan'),1e200,-1e200):
        poisoned=raw.copy();poisoned[~mask]=val
        actual=condition_features(poisoned,mask,mode,10.,init,fb,ek)
        assert np.array_equal(expected,actual)
    if mode=='mask_only':
        changed=raw*100+2
        assert np.array_equal(expected,condition_features(changed,mask,mode,10.,init*100,fb,ek))


def test_actual_initializer_excludes_query_location_even_after_poison():
    import pandas as pd
    from scripts.run_e3_revised import initialize_crossfit
    names=[f'v{i}' for i in range(17)]
    pool=pd.DataFrame(np.arange(1,103,dtype=np.float64).reshape(6,17),columns=names)
    pool['location']=['A','A','B','B','C','C']
    query=pool.loc[[0],names].to_numpy();query[:,[2,5]]=np.nan
    a=initialize_crossfit(query,np.array([0]),np.array(['A']),pool,names)
    pool.loc[pool.location=='A',names]=1e9
    b=initialize_crossfit(query,np.array([0]),np.array(['A']),pool,names)
    for method in a[0]:
        assert np.array_equal(a[0][method],b[0][method])
        for record in a[3][method]:
            for cell in record['cells']:assert not set(cell['donor_ids'])&{0,1}


def test_loss_unknown_values_are_not_arithmetic_operands():
    noise=torch.tensor([1.,2.,float('nan')],dtype=torch.float64)
    pred=torch.tensor([1.,1.,float('nan')],dtype=torch.float64)
    assert epsilon_loss(pred,noise,torch.tensor([True,True,False])).item()==.5


@pytest.mark.parametrize('k',[16,17])
def test_float64_temporal_model_and_gradients(k):
    torch.set_num_threads(2);torch.manual_seed(42)
    m=CSDICore(k);z=torch.randn(2,3,k,dtype=torch.float64)
    c=torch.randn(2,3,85,dtype=torch.float64);time=torch.arange(3,dtype=torch.float64).expand(2,-1)
    pred=m(z,c,time,torch.tensor([1,2]));pred.square().mean().backward()
    assert pred.shape==z.shape and pred.dtype==torch.float64
    assert all(p.dtype==torch.float64 for p in m.parameters())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_raw_mask_is_not_latent_mask_and_final_projection():
    z=torch.tensor([[1.,2.,4.,8.]],dtype=torch.float64)
    clr=project_final(z,'clr16');hk=project_final(z,'hkglr16_actual',(0,2))
    assert abs(float(clr.sum()))<1e-12
    assert abs(float(hk[:,[0,2]].mean()))<1e-12
    assert torch.equal(project_final(z,'ilr16'),z)
    # Moving the hidden fourth raw part changes ALL CLR coordinates; an
    # observed raw index is not an observed CLR coordinate.
    a=np.log([[1.,2.,3.,4.]]);b=np.log([[1.,2.,3.,40.]])
    assert np.all((a-a.mean(1,keepdims=True))!=(b-b.mean(1,keepdims=True)))


def test_fixed_truth_rows_include_zero_sum_predictions():
    truth=np.array([[1.,2.,3.],[2.,3.,4.],[0.,0.,0.]])
    hidden=np.ones_like(truth,dtype=bool);query=np.full_like(truth,np.nan)
    pred=truth.copy();pred[0]=0
    a=evaluate(truth,pred,hidden,query,.5,{'rare':[0],'common':[1,2]})
    b=evaluate(truth,truth,hidden,query,.5,{'rare':[0],'common':[1,2]})
    assert a['eligible_row_positions']==b['eligible_row_positions']==[0,1]
    assert a['m2']>0 and a['n_prediction_zero_sum_on_eligible']==1
    assert a['native_jsd_diagnostic'] is None


def test_prevalence_denominator_excludes_natural_missing():
    _,rate=prevalence_bins(np.array([[1.,0.],[np.nan,1.],[np.nan,0.]]),['a','b'])
    assert rate[0]==1 and rate[1]==1/3


def test_restoration_locks_observed_zeros():
    from scripts.run_e3_revised import load_transform_module
    tm=load_transform_module();raw=np.arange(17,dtype=np.float64)[None,:]
    q=raw.copy();q[:,[3,5]]=np.nan
    for name in ('clr16','ilr16','hkglr16_actual'):
        transform=tm.make_transform(name,17,.5,(0,1,2,3,4) if name.startswith('hk') else None)
        p=transform.inverse(transform.forward(raw));out,info=tm.restore_observed_counts(p,q)
        visible=np.isfinite(q)
        assert np.array_equal(out[visible],q[visible]) and out[0,0]==0.


def test_windows_do_not_mix_locations_or_duplicate_rows():
    import pandas as pd
    from scripts.run_e3_revised import window_indices
    df=pd.DataFrame({'location':['A']*10+['B']*4,'date':list(range(10))+list(range(4))})
    ids=np.array([0,1,2,4,5,6,7,8,9,10,11,12,13])
    windows=window_indices(df,ids,8)
    assert sorted(np.concatenate(windows).tolist())==list(range(len(ids)))
    for w in windows: assert df.loc[ids[w],'location'].nunique()==1
