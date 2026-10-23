"""Generator invariants and end-to-end dataset/model contract checks."""
import json
from dataclasses import replace
from pathlib import Path
import shutil
import numpy as np
import pytest
from longitudinal_impute.synthetic import GeneratorConfig, generate, export_dataset, load_conditioning, load_evaluation, NotReadyError
from longitudinal_impute.synthetic.backend import resolve_rscript
from longitudinal_impute.baselines.panels import subset
from longitudinal_impute.contracts import FitScope

@pytest.mark.parametrize('profile', ['easy','medium','hard'])
def test_generator_truth_invariants(profile):
    cfg=GeneratorConfig(profile=profile,scale='small')
    a=generate(cfg); b=generate(cfg)
    for key in a: np.testing.assert_array_equal(a[key],b[key])
    p,x=a['p'],a['counts']
    assert p.shape == (30,120,12)
    np.testing.assert_allclose(p.sum(axis=2)+a['residual_p'],1,atol=1e-14)
    np.testing.assert_array_equal(x.sum(axis=2)+a['residual_counts'],a['N'])
    assert (a['N']>0).all() and (x>=0).all()
    before=np.arange(120)[None,:,None] < a['onsets'][:,None,:]
    assert (p[before]==0).all() and (x[before]==0).all()
    after=~before
    assert (p[after]>0).all()
    assert all((a['strata']==i).any() for i in (0,1,2))
    assert set(a['split'])=={'train','validation','test'}
    assert dict(zip(*np.unique(a['split'],return_counts=True))) == {'train':18,'validation':6,'test':6}
    different=generate(replace(cfg,seed=cfg.seed+1))
    assert not np.array_equal(a['counts'],different['counts'])


def test_profiles_share_splits_trait_and_offsets_latent_stream():
    easy=generate(GeneratorConfig(profile='easy',scale='small'))
    hard=generate(GeneratorConfig(profile='hard',scale='small'))
    np.testing.assert_array_equal(easy['split'],hard['split'])
    np.testing.assert_array_equal(easy['entity_trait_01'],hard['entity_trait_01'])
    np.testing.assert_array_equal(easy['residual_p'],hard['residual_p'])
    np.testing.assert_allclose(easy['entity_shifts']*8,hard['entity_shifts'])
    assert easy['N'].mean()>hard['N'].mean()*10

@pytest.mark.parametrize('scale,entities',[('small',30),('standard',60),('large',120)])
def test_scale_and_invalid_config(scale,entities):
    assert generate(GeneratorConfig(scale=scale))['counts'].shape==(entities,120,12)
    with pytest.raises(ValueError): generate(GeneratorConfig(temporal_correlation=1))
    with pytest.raises(ValueError): generate(GeneratorConfig(residual_sd=1))


def test_not_ready_creates_no_dataset(tmp_path):
    output=tmp_path/'blocked'
    with pytest.raises(NotReadyError):
        export_dataset(output,GeneratorConfig(scale='small'),mask_specs=[{'mechanism':'MNAR'}])
    assert not output.exists()


def test_bundle_and_current_baselines(tmp_path):
    try: rscript=resolve_rscript()
    except RuntimeError: pytest.skip('Rscript required for mask integration')
    patterns=json.loads(Path('nextgen/configs/schouten2018_mar_patterns.json').read_text())
    mcar={**patterns,'mechanism':'MCAR'}
    bundle=tmp_path/'dataset'
    report=export_dataset(bundle,GeneratorConfig(scale='small'),mask_specs=[mcar,patterns],rscript=rscript)
    assert report['status']=='COMPLETE'
    for file,digest in report['files_sha256'].items():
        import hashlib
        assert hashlib.sha256((bundle/file).read_bytes()).hexdigest()==digest
    name=report['masks'][0]['name']
    expected=load_conditioning(bundle,name,split='test')
    labels,latent=load_evaluation(bundle,name,split='test')
    assert labels.row_ids==expected.row_ids
    np.testing.assert_array_equal(labels.target_mask,~expected.conditioning_mask)
    assert np.isnan(labels.values[~labels.target_mask]).all()
    assert latent['p'].shape==expected.counts.shape
    assert (latent['p'][latent['strata']==0]==0).all()
    # Loader must work even when evaluator-only directory is inaccessible/absent.
    shutil.move(str(bundle/'private'),str(tmp_path/'evaluator'))
    train=load_conditioning(bundle,name,split='train')
    test=load_conditioning(bundle,name,split='test')
    assert not set(train.entity_ids)&set(test.entity_ids)
    assert np.isnan(train.counts[~train.conditioning_mask]).all()
    assert set(train.metadata)=={'entity_trait_01'}
    from longitudinal_impute.baselines.adapters import CountBaselineAdapter
    from longitudinal_impute.baselines.hron import HronKNN
    from longitudinal_impute.baselines.micoda import MICoDa
    from longitudinal_impute.baselines.tsagris import TsagrisKNN
    # Spread donor rows across all waves, keeping the ordinary fit/predict API.
    donors=subset(train,np.arange(0,len(train.row_ids),3))
    query=subset(test,np.arange(0,120,10))
    for core in (HronKNN(2),TsagrisKNN(2),MICoDa(W=1e-5,max_iter=3)):
        scope=FitScope('inductive',donors.row_ids,'synthetic_entity_split',donors.mask_id)
        model=CountBaselineAdapter(core).fit(donors,scope)
        result=model.predict(query)
        assert np.isfinite(result).all() and (result>=0).all()
        np.testing.assert_array_equal(result[query.conditioning_mask],query.counts[query.conditioning_mask])
        assert (result.sum(axis=1)<=query.totals+1e-8).all()
