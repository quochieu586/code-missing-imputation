"""Regression probes for the synthetic repair ticket F1--F5."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import numpy as np
import pytest
from longitudinal_impute.synthetic import GeneratorConfig, export_dataset, load_conditioning, load_evaluation
from longitudinal_impute.synthetic.cli import main
from longitudinal_impute.synthetic.service import MaskSpec
from longitudinal_impute.synthetic.scoring import score

SOURCE = Path('datasets/synthetic/wave_panel_v1/easy')
MASK = '01_mcar_declared_patterns'


@pytest.fixture
def bundle(tmp_path):
    dest = tmp_path/'bundle'
    shutil.copytree(SOURCE,dest)
    return dest


def rehash(bundle, relative):
    file = bundle/'manifest.json'
    manifest = json.loads(file.read_text())
    manifest['files_sha256'][relative] = hashlib.sha256((bundle/relative).read_bytes()).hexdigest()
    file.write_text(json.dumps(manifest))


def test_private_csv_mutation_rejected(bundle):
    path = bundle/'private/complete_counts.csv'
    text = path.read_text(); path.write_text(text.replace(',0,',',1,',1))
    with pytest.raises(ValueError,match='checksum'):
        load_evaluation(bundle,MASK)


def test_private_npz_permutation_rejected(bundle):
    path = bundle/'private/ground_truth.npz'
    with np.load(path) as z: truth = {k:z[k] for k in z.files}
    truth['p'] = np.roll(truth['p'],1,axis=0)
    np.savez_compressed(path,**truth)
    with pytest.raises(ValueError,match='checksum'):
        load_evaluation(bundle,MASK)


def test_private_reorder_never_silently_changes_p(bundle):
    path = bundle/'private/complete_counts.csv'
    lines = path.read_text().splitlines()
    path.write_text('\n'.join([lines[0],*reversed(lines[1:])])+'\n')
    with pytest.raises(ValueError,match='checksum'):
        load_evaluation(bundle,MASK)
    rehash(bundle,'private/complete_counts.csv')
    with pytest.raises(ValueError,match='identity contract'):
        load_evaluation(bundle,MASK)


@pytest.mark.parametrize('field', ['p','counts','N','strata'])
def test_private_internal_consistency_after_rehash(bundle,field):
    path = bundle/'private/ground_truth.npz'
    with np.load(path) as z: truth = {k:z[k] for k in z.files}
    truth[field].flat[0] += 1
    np.savez_compressed(path,**truth); rehash(bundle,'private/ground_truth.npz')
    with pytest.raises(ValueError,match='Inconsistent'):
        load_evaluation(bundle,MASK)


def test_private_shape_keys_checked(bundle):
    path = bundle/'private/ground_truth.npz'
    with np.load(path) as z: truth = {k:z[k] for k in z.files}
    truth['p'] = truth['p'].ravel()
    np.savez_compressed(path,**truth); rehash(bundle,'private/ground_truth.npz')
    with pytest.raises(ValueError,match='keys/shapes'):
        load_evaluation(bundle,MASK)


def test_public_checksum_schema_and_private_absence(bundle):
    shutil.rmtree(bundle/'private')
    assert load_conditioning(bundle,MASK).counts.shape == (7200,12)
    path = bundle/'schema.toml'
    path.write_text(path.read_text().replace('component_12','component_11'))
    with pytest.raises(ValueError,match='checksum'):
        load_conditioning(bundle,MASK)
    rehash(bundle,'schema.toml')
    with pytest.raises(ValueError):
        load_conditioning(bundle,MASK)


def test_scenario_rejected_before_data_io(tmp_path):
    spec = dict(scenario='table1_design',mechanism='MCAR',seed=2018,case_probability=.3,wave_time='2000-03-01')
    path = tmp_path/'spec.json'; path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='scenario'):
        MaskSpec.parse(spec)
    out = tmp_path/'out'
    assert main(['mask','--data','absent','--schema-config','absent','--spec',str(path),'--output',str(out)]) == 1
    assert not out.exists()


def test_export_validates_all_specs_and_retry(tmp_path):
    valid = json.loads(Path('nextgen/configs/schouten2018_mcar_patterns.json').read_text())
    out = tmp_path/'out'
    with pytest.raises(ValueError):
        export_dataset(out,GeneratorConfig(scale='small'),mask_specs=[valid,{**valid,'case_probability':2}])
    assert not out.exists()
    # Retry the same destination; no mask requires no R subprocess.
    assert export_dataset(out,GeneratorConfig(scale='small'))['format_version'] == 'synthetic-panel-v2'


def test_runtime_export_failure_cleans_staging_and_retry(tmp_path,monkeypatch):
    from longitudinal_impute.synthetic import service
    valid = json.loads(Path('nextgen/configs/schouten2018_mcar_patterns.json').read_text())
    out = tmp_path/'out'
    calls = []
    original_apply = service.apply_mask
    def fail(*args,**kwargs):
        calls.append(1)
        if len(calls) == 1:
            return original_apply(*args,**kwargs)
        raise RuntimeError('injected backend failure')
    with monkeypatch.context() as context:
        context.setattr(service,'apply_mask',fail)
        with pytest.raises(RuntimeError,match='injected'):
            export_dataset(out,GeneratorConfig(scale='small'),mask_specs=[valid,valid])
    assert len(calls) == 2 and not out.exists() and not list(tmp_path.glob('.out-*'))
    export_dataset(out,GeneratorConfig(scale='small'))


@pytest.mark.parametrize('mechanism', ['mcar','mar'])
def test_v2_identity_and_mask_replay(tmp_path,mechanism):
    from longitudinal_impute.synthetic.backend import resolve_rscript
    valid = json.loads(Path(f'nextgen/configs/schouten2018_{mechanism}_patterns.json').read_text())
    out = tmp_path/'v2'
    export_dataset(out,GeneratorConfig(),mask_specs=[valid],rscript=resolve_rscript())
    name = f'01_{mechanism}_declared_patterns'
    old_name = f"{'01' if mechanism == 'mcar' else '02'}_{mechanism}_declared_patterns"
    original,latent = load_evaluation(out,name,split='test')
    old,old_latent = load_evaluation(SOURCE,old_name,split='test')
    np.testing.assert_array_equal(original.target_mask,old.target_mask)
    np.testing.assert_array_equal(latent['p'],old_latent['p'])
    path = out/'private/complete_counts.csv'
    lines = path.read_text().splitlines(); path.write_text('\n'.join([lines[0],*reversed(lines[1:])])+'\n')
    rehash(out,'private/complete_counts.csv')
    reordered,truth = load_evaluation(out,name,split='test')
    np.testing.assert_array_equal(reordered.values,original.values)
    np.testing.assert_array_equal(truth['p'],latent['p'])


def test_score_distinguishes_sample_ratio_from_latent_and_hidden_only():
    panel = SimpleNamespace(row_ids=('a',), counts=np.array([[np.nan,7.]]), totals=np.array([10.]),conditioning_mask=np.array([[False,True]]))
    labels = SimpleNamespace(row_ids=('a',),values=np.array([[0.,np.nan]]),target_mask=~panel.conditioning_mask)
    latent = {'p':np.array([[.2,.5]]),'strata':np.array([[1,2]])}
    metrics = score(labels,latent,panel,np.array([[0.,7.]]))
    assert metrics['overall']['count_mae'] == metrics['overall']['sample_ratio_mae'] == 0
    assert metrics['overall']['latent_p_mae'] == .2
    assert metrics['overall']['n'] == 1
    assert metrics['by_stratum']['sampling_zero']['latent_p_rmse'] == .2
    assert metrics['by_stratum']['positive_count']['n'] == 0
    assert metrics['by_stratum']['positive_count']['latent_p_mae'] is None


def test_residual_contract():
    from longitudinal_impute.synthetic import generate
    truth = generate(GeneratorConfig(profile='hard'))
    assert (truth['residual_counts']==0).sum() == 178
    assert (truth['counts'].sum(axis=2) <= truth['N']).all()
    assert (truth['p'].sum(axis=2) < 1).all()
    with pytest.raises(ValueError):
        GeneratorConfig(residual_sd=-.03).validate()
