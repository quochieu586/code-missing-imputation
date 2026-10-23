"""Synthetic hierarchy tests: no real dataset or accepted run is used."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from longitudinal_impute.benchmark import engine, driver
from longitudinal_impute.benchmark.artifacts import file_hash, publish_json, prediction_artifact_path
from longitudinal_impute.benchmark.candidates import supplementary_candidate, final_candidate
from longitudinal_impute.benchmark.completion import RunArtifacts


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding='utf-8')


@pytest.fixture
def prepared(tmp_path):
    lines = ['entity,date,total,a,b']
    for location in range(6):
        for day in (1, 2):
            b = '' if location == 5 and day == 2 else str(20 + location)
            lines.append(f'L{location},2020-01-0{day},100,{10+day+location},{b}')
    (tmp_path/'data.csv').write_text('\n'.join(lines)+'\n')
    (tmp_path/'schema.toml').write_text('[dataset]\nentity="entity"\ntime="date"\ntotal="total"\ntargets=["a","b"]\n')
    config = {'data':'data.csv','schema':'schema.toml','folds':3,'split_seed':3,
        'scenarios':['whole_feature'],'mask_seeds':[7],'mask_rate':.3,
        'numeric_metadata':[],'fallback':'train_mean','degenerate_policies':True,
        'candidates':{'tsagris':[{'method':'tsagris_alpha','parameters':{'k':1,'alpha':.5}}]},
        'supplementary_methods':['tsagris_basic'],'supplementary_scenarios':['whole_feature'],
        'supplementary_wall_seconds_per_fold_method':30}
    write(tmp_path/'config.json',config)
    output = tmp_path/'run'
    engine.prepare(tmp_path,tmp_path/'config.json',output)
    manifest,raw,_,_ = engine.load_run(tmp_path,output)
    return SimpleNamespace(root=tmp_path,output=output,manifest=manifest,raw=raw,
                           store=RunArtifacts(output,manifest,raw))


def complete(fixture, fold=0):
    for stage in ('validation','test'):
        engine.run_fold(fixture.root,fixture.output,fold,stage)
    engine.supplementary_one(fixture.root,fixture.output,fold,'tsagris_basic')
    publish_json(fixture.output/f'fold{fold}/completion.json',{
        'status':'done',**fixture.store.identity(fold),'seconds':1.,
        'artifacts':fixture.store.fold_inventory(fold)})


@pytest.fixture
def completed(prepared):
    complete(prepared)
    return prepared


def prediction(fixture, supplementary=False):
    directory = fixture.output/'fold0'
    directory /= 'supplementary/tsagris_basic' if supplementary else 'test/tsagris'
    job = directory/'whole_feature_7'
    return prediction_artifact_path(job,json.loads((job/'result.json').read_text()))


def test_valid_fold_and_supplementary_are_reused_without_fitting(completed):
    completed.store.fold(0)
    with patch.object(engine,'make_adapter',side_effect=AssertionError('must not refit')):
        engine.run_fold(completed.root,completed.output,0,'all')
        engine.supplementary_one(completed.root,completed.output,0,'tsagris_basic')


def test_driver_checks_children_before_accepting_marker(prepared):
    publish_json(prepared.output/'fold0/completion.json',{
        'status':'done',**prepared.store.identity(0),'seconds':1.,'artifacts':{}})
    with patch('sys.argv',['driver','--root',str(prepared.root),'--output',str(prepared.output)]), \
         patch.object(driver.subprocess,'Popen',side_effect=AssertionError('must not launch')):
        with pytest.raises(ValueError,match='artifact'):
            driver.main()
    assert (prepared.output/'fold0/completion.json').exists()


def test_driver_reuses_only_fully_valid_folds(completed,capsys):
    complete(completed,1)
    complete(completed,2)
    with patch('sys.argv',['driver','--root',str(completed.root),'--output',str(completed.output)]), \
         patch.object(driver.subprocess,'Popen',side_effect=AssertionError('must not launch')):
        driver.main()
    assert 'all_folds_complete' in capsys.readouterr().out


def test_worker_exit_zero_does_not_publish_completion_without_children(prepared):
    fake = SimpleNamespace(pid=123,poll=lambda:0)
    with patch('sys.argv',['driver','--root',str(prepared.root),'--output',str(prepared.output),'--workers','1']), \
         patch.object(driver.subprocess,'Popen',return_value=fake):
        with pytest.raises(ValueError,match='artifact'):
            driver.main()
    assert not (prepared.output/'fold0/completion.json').exists()


@pytest.mark.parametrize('relative',[
    'fold0/selection.json',
    'fold0/validation/tsagris/candidate0/candidate.json',
    'fold0/validation/tsagris/candidate0/seed7/result.json',
    'fold0/test/tsagris/whole_feature_7/result.json',
    'fold0/supplementary/tsagris_basic/finished.json',
    'fold0/supplementary/tsagris_basic/whole_feature_7/result.json'])
def test_missing_child_rejected_and_retained_in_fixture(completed,relative):
    path = completed.output/relative
    retained = path.with_suffix('.retained')
    assert path.resolve().is_relative_to(completed.output.resolve())
    path.rename(retained)
    with pytest.raises(ValueError):
        completed.store.fold(0)
    assert retained.is_file()


@pytest.mark.parametrize('supplementary',[False,True])
def test_corrupt_prediction_is_rejected_through_fold_marker(completed,supplementary):
    path = prediction(completed,supplementary)
    with path.open('ab') as stream:
        stream.write(b'corrupt-review-fixture')
    with pytest.raises(ValueError,match='corrupt'):
        completed.store.fold(0)


def test_modified_child_metadata_invalidates_sealed_fold(completed):
    marker = completed.output/'fold0/test/tsagris/whole_feature_7/result.json'
    record = json.loads(marker.read_text())
    record['seconds'] += 1
    write(marker,record)
    with pytest.raises(ValueError,match='inventory'):
        completed.store.fold(0)


def test_direct_engine_entrypoints_cannot_bypass_a_fold_marker(completed):
    marker = completed.output/'fold0/test/tsagris/whole_feature_7/result.json'
    record = json.loads(marker.read_text())
    record['seconds'] += 1
    write(marker,record)
    with pytest.raises(ValueError,match='inventory'):
        engine.run_fold(completed.root,completed.output,0,'all')
    with pytest.raises(ValueError,match='inventory'):
        engine.supplementary_one(completed.root,completed.output,0,'tsagris_basic')


def test_selection_cannot_reuse_changed_candidate_or_score(completed):
    marker = completed.output/'fold0/validation/tsagris/candidate0/candidate.json'
    record = json.loads(marker.read_text())
    record['ratio_mae'] += 1
    write(marker,record)
    with pytest.raises(ValueError,match='score'):
        completed.store.selection(0)


def test_successful_supplementary_marker_without_children_is_rejected(prepared):
    engine.run_fold(prepared.root,prepared.output,0,'validation')
    selection = prepared.store.selection(0)['selected']
    write(prepared.output/'fold0/supplementary/tsagris_basic/finished.json',{
        'status':'ok',**prepared.store.identity(0,'tsagris_basic'),'seconds':1.,'artifacts':{},
        'candidate':supplementary_candidate(prepared.manifest['config'],selection,'tsagris_basic')})
    with patch.object(engine.subprocess,'run',side_effect=AssertionError('must not run')):
        with pytest.raises(ValueError,match='evaluation'):
            engine.run_fold(prepared.root,prepared.output,0,'supplementary')


@pytest.mark.parametrize('status',['not_applicable','numerical_failure','resource_limit','failed_runtime'])
def test_terminal_failures_remain_distinct_and_require_their_evidence(completed,status):
    marker = completed.output/'fold0/supplementary/tsagris_basic/finished.json'
    record = json.loads(marker.read_text())
    record.update(status=status,error='synthetic failure',artifacts={})
    if status in ('resource_limit','failed_runtime'):
        log = marker.parent/'retained_execution.txt'
        log.write_text('retained execution log')
        record.update(execution_log=log.relative_to(completed.output).as_posix(),execution_log_hash=file_hash(log))
    record['artifacts'] = completed.store.supplementary_inventory(0,'tsagris_basic',record)
    write(marker,record)
    assert completed.store.supplementary(0,'tsagris_basic')['status'] == status
    if status in ('resource_limit','failed_runtime'):
        log.write_text('corrupt log')
        with pytest.raises(ValueError,match='log'):
            completed.store.supplementary(0,'tsagris_basic')


def test_final_export_cache_validates_identity_selection_and_csv(completed):
    for fold in (1,2):
        engine.run_fold(completed.root,completed.output,fold,'validation')
    result = engine.final_refit(completed.root,completed.output)
    with patch.object(engine,'make_adapter',side_effect=AssertionError('must not refit')):
        assert engine.final_refit(completed.root,completed.output) == result
    record = result[0]
    csv = completed.output/'final/tsagris'/record['csv_artifact']
    with csv.open('ab') as stream:
        stream.write(b'corrupt CSV')
    selections = [completed.store.selection(fold) for fold in range(3)]
    candidate = final_candidate(completed.manifest['config'],selections,'tsagris')
    with pytest.raises(ValueError,match='CSV'):
        completed.store.final('tsagris',candidate)


def test_incomplete_evaluation_retry_keeps_attempt_and_validates_cached_file(prepared):
    train,scope = engine.train_panel(prepared.raw,prepared.manifest['config'],[0,1,2,3],0)
    adapter = engine.make_adapter(prepared.manifest['config'],prepared.manifest['config']['candidates']['tsagris'][0]).fit(train,scope)
    job = prepared.output/'isolated_retry'
    args = (adapter,prepared.raw,prepared.manifest['config'],prepared.output,[4,5],'whole_feature',7,job)
    with patch.object(adapter,'predict_result',side_effect=RuntimeError('injected interruption')):
        with pytest.raises(RuntimeError,match='injected'):
            engine.evaluate(*args)
    result = engine.evaluate(*args)
    assert len(list((job/'attempts').iterdir())) == 2
    with patch.object(adapter,'predict_result',side_effect=AssertionError('must not predict')):
        assert engine.evaluate(*args) == result


@pytest.mark.parametrize('outcome',['timeout','exit_error','exit_zero_without_marker'])
def test_parent_publishes_valid_terminal_failure_and_retained_log(prepared,outcome):
    engine.run_fold(prepared.root,prepared.output,0,'validation')
    effect = subprocess.TimeoutExpired('synthetic',30) if outcome == 'timeout' else None
    code = 3 if outcome == 'exit_error' else 0
    with patch.object(engine.subprocess,'run',side_effect=effect,return_value=SimpleNamespace(returncode=code)):
        engine.run_fold(prepared.root,prepared.output,0,'supplementary')
    record = prepared.store.supplementary(0,'tsagris_basic')
    assert record['status'] == ('resource_limit' if outcome == 'timeout' else 'failed_runtime')
    assert (prepared.output/record['execution_log']).exists()
    assert record['artifacts']


def test_marker_path_escape_cannot_be_accepted_by_ancestor(completed):
    marker = completed.output/'fold0/test/tsagris/whole_feature_7/result.json'
    record = json.loads(marker.read_text())
    record['predictions_artifact'] = '../../outside.npz'
    write(marker,record)
    with pytest.raises(ValueError,match='path'):
        completed.store.fold(0)
