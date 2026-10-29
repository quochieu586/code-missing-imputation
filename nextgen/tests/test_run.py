"""run.py end to end on the small COVID-like and fake microbiome fixtures."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from longitudinal_impute import run
from longitudinal_impute.baselines.hron import HronKNN
from longitudinal_impute.data import load_csv

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
COVID_SCHEMA = '[dataset]\nentity="location"\ntime="date"\ntotal="total_sequence"\ntargets=["v1","v2","v3","v4"]\n'
MICROBIOME_SCHEMA = ('[dataset]\nentity="subject"\ntime="visit"\ntargets=["Bacteroides","Prevotella",'
                     '"Faecalibacterium","Akkermansia","Bifidobacterium"]\n')


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def call(tmp_path, data, schema, models, *extra):
    output = tmp_path / 'out'
    code = run.main(['--data', str(data), '--schema', str(write(tmp_path, 'schema.toml', schema)),
                     '--models', str(write(tmp_path, 'models.toml', models)), '--output', str(output), *extra])
    return code, output


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def test_covid_like_run_writes_every_output_and_survives_a_failing_model(tmp_path, monkeypatch):
    fitted = []

    class RecordingHron(HronKNN):
        def fit(self, values):
            fitted.append(values.copy())
            return super().fit(values)

    monkeypatch.setitem(run.MODELS, 'hron', RecordingHron)
    source = FIXTURES / 'covid_small.csv'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    # MICoDa defaults (W = 0) reject the zero counts, exactly as on the real COVID table.
    code, out = call(tmp_path, source, COVID_SCHEMA, '[hron]\nk = 3\n[tsagris]\nk = 2\n[micoda]\n',
                     '--mask', 'whole_feature', '--mask-rate', '.2', '--seeds', '1', '2')
    assert code == 1
    names = {p.name for p in out.iterdir()}
    assert names == {'config.json', 'hron_completed.csv', 'tsagris_completed.csv', 'mask_whole_feature_seed1.npy',
                     'mask_whole_feature_seed2.npy', 'summary.csv', 'summary.json', 'errors.log'}
    log = (out / 'errors.log').read_text(encoding='utf-8').splitlines()
    assert [line.split(' | ')[:4] for line in log] == [['micoda', 'native', '-', 'BaselineError'],
                                                      ['micoda', 'masked', '1', 'BaselineError'],
                                                      ['micoda', 'masked', '2', 'BaselineError']]
    assert 'MICoDa ALR needs positive parts' in log[0]

    rows = read_rows(out / 'summary.csv')
    assert [(r['model'], r['pass'], r['seed'], r['status']) for r in rows] == [
        ('hron', 'native', '', 'ok'), ('tsagris', 'native', '', 'ok'), ('micoda', 'native', '', 'failed'),
        *[(m, 'masked', s, st) for s in '12' for m, st in (('hron', 'ok'), ('tsagris', 'ok'), ('micoda', 'failed'))]]
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert summary['native']['tsagris']['info']['zero_anchor_global_rows'] == 1
    assert set(summary['native']['hron']['wasserstein']) == {'v1', 'v2', 'v3', 'v4'}
    assert {a['model']: a['runs_ok'] for a in summary['aggregate'] if a['pass'] == 'masked'} == {
        'hron': 2, 'tsagris': 2, 'micoda': 0}
    config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    assert config['data_sha256'] == digest and config['seeds'] == [1, 2]
    assert config['models'] == {'hron': {'k': 3}, 'tsagris': {'k': 2}, 'micoda': {}}

    # Every model of a seed sees the same mask, and only the visible cells.
    data = load_csv(source, write(tmp_path, 'schema.toml', COVID_SCHEMA))
    observed = np.isfinite(data.counts)
    assert len(fitted) == 3
    assert np.isfinite(fitted[0][:, :-1]).sum() == observed.sum()
    for seed, values in zip((1, 2), fitted[1:]):
        hidden = np.load(out / f'mask_whole_feature_seed{seed}.npy')
        np.testing.assert_array_equal(np.isfinite(values[:, :-1]), observed & ~hidden)
        masked = summary['masked'][str(seed)]
        assert masked['mask']['hidden_cells'] == hidden.sum()
        assert masked['models']['hron']['errors']['all']['cells'] == hidden.sum()

    # Completed CSV keeps every observed text cell and the extra columns; the input is untouched.
    original = read_rows(source)
    for name in ('hron', 'tsagris'):
        completed = load_csv(out / f'{name}_completed.csv', write(tmp_path, 'schema.toml', COVID_SCHEMA))
        assert np.isfinite(completed.counts).all()
        for i, row in enumerate(read_rows(out / f'{name}_completed.csv')):
            assert row['note'] == 'keep'
            for j, target in enumerate(data.targets):
                if observed[i, j]:
                    assert row[target] == original[i][target]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_error_log_translates_row_context_to_entity_and_csv_line(tmp_path):
    code, out = call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA,
                     '[hron]\n[tsagris]\nzero_anchor = "error"\n', '--mask', 'none')
    assert code == 1
    assert (out / 'hron_completed.csv').exists() and not (out / 'tsagris_completed.csv').exists()
    [line] = (out / 'errors.log').read_text(encoding='utf-8').splitlines()
    assert line == ("tsagris | native | - | BaselineError | row 13: observed parts sum to 0, JSD is undefined "
                    "[row=13, reason=zero_anchor, entity=Brill, csv_line=15]")
    assert not list(out.glob('mask_*'))


def test_microbiome_without_total_column_runs_clean(tmp_path):
    source = FIXTURES / 'microbiome_small.csv'
    code, out = call(tmp_path, source, MICROBIOME_SCHEMA, '[hron]\n[tsagris]\n[micoda]\nW = 0.5\n',
                     '--mask', 'random_cell', '--mask-rate', '.1', '--seeds', '3')
    assert code == 0 and not (out / 'errors.log').exists()
    rows = read_rows(out / 'summary.csv')
    assert [(r['model'], r['pass'], r['status']) for r in rows] == [
        (m, p, 'ok') for p in ('native', 'masked') for m in ('hron', 'tsagris', 'micoda')]
    # No NaN: the native completed tables are the input itself.
    assert all(r['imputed_cells'] == '0' for r in rows if r['pass'] == 'native')
    assert read_rows(out / 'hron_completed.csv') == read_rows(source)
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    errors = summary['masked']['3']['models']['tsagris']['errors']
    assert errors['all']['cells'] == np.load(out / 'mask_random_cell_seed3.npy').sum() > 0
    assert errors['forced_zero']['cells'] == 0


def test_refuses_existing_output_and_bad_model_config(tmp_path):
    data = FIXTURES / 'covid_small.csv'
    (tmp_path / 'out').mkdir()
    with pytest.raises(FileExistsError):
        call(tmp_path, data, COVID_SCHEMA, '[hron]\n')
    (tmp_path / 'out').rmdir()
    with pytest.raises(ValueError, match='declare one or more'):
        call(tmp_path, data, COVID_SCHEMA, '[knn]\n')
    with pytest.raises(ValueError, match=r'\[hron\].*neighbours'):
        call(tmp_path, data, COVID_SCHEMA, '[hron]\nneighbours = 3\n')
    with pytest.raises(ValueError):
        call(tmp_path, data, COVID_SCHEMA, '[tsagris]\nk = 0\n')
    assert not (tmp_path / 'out').exists()


def test_malformed_csv_stops_before_any_output(tmp_path):
    source = write(tmp_path, 'bad.csv', 'location,date,total,a,b,c\nL,2020-01-01,100,10,20,70\n'
                                        'L,2020-01-02,100,20,30,50\nL,2020-01-03,100,20,,,EXTRA\n')
    schema = '[dataset]\nentity="location"\ntime="date"\ntotal="total"\ntargets=["a","b","c"]\n'
    with pytest.raises(ValueError, match='CSV line 4 has 1 more field'):
        call(tmp_path, source, schema, '[tsagris]\n')
    assert not (tmp_path / 'out').exists()


def test_interrupted_completed_csv_is_not_published(tmp_path, monkeypatch):
    original = csv.DictWriter.writerow
    calls = []

    def fail_after_header(self, row):
        if 'note' in self.fieldnames:             # only the completed CSV has the extra column
            calls.append(row)
            if len(calls) == 3:                   # header and one row written, then the write breaks
                raise ValueError('write interrupted (simulated)')
        return original(self, row)

    monkeypatch.setattr(csv.DictWriter, 'writerow', fail_after_header)
    code, out = call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA, '[hron]\n', '--mask', 'none')
    monkeypatch.undo()
    assert code == 1
    assert not list(out.glob('*_completed.csv')) and not list(out.glob('.*tmp'))
    [line] = (out / 'errors.log').read_text(encoding='utf-8').splitlines()
    assert line.startswith('hron | native | - | ValueError | write interrupted')
