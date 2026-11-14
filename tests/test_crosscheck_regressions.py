"""Small, rerunnable counterexamples from CROSSCHECK_vi; no research data required."""
import json

import numpy as np
import pytest

from longitudinal_impute import run
from longitudinal_impute.baselines.micoda import CovidResult, MICoDaCovid, prepare, restore
from test_run import call, write, FIXTURES, COVID_SCHEMA


def test_r1_allocates_other_to_single_hidden_target():
    # Truth: [20, 30], N=100, Other=50. Hiding 30 makes R1 allocate 80.
    prepared = prepare([[20., 30.], [30., 20.], [20., 30.]], [100., 100., 100.],
                       np.array([[False, False], [False, False], [False, True]]))
    result = restore(prepared, np.ones((3, 2)))
    assert result.values[2, 1] == 80.
    assert result.values[2, 1] - 30. == 50.


def test_real_adapter_failure_preserves_original_error_and_no_csv(tmp_path):
    source = write(tmp_path, 'insufficient.csv',
                   'location,date,total_sequence,v1,v2,v3,v4\nA,1,100,10,20,30,40\nB,1,100,10,,,\n')
    code, out = call(tmp_path, source, COVID_SCHEMA, '[micoda]\n')
    assert code == 1
    result = json.loads((out / 'summary.json').read_text())['native']['micoda']
    assert result['info']['error']['code'] == 'insufficient_fit'
    assert 'Complete-case initialization' in result['error']
    assert 'insufficient_fit' in (out / 'errors.log').read_text()
    assert not list(out.glob('micoda*.csv'))


@pytest.mark.parametrize('status', ['partial', 'failed'])
def test_adapter_terminal_status_native_and_masked(tmp_path, monkeypatch, status):
    def simulate(self, counts, totals, hidden):
        values = np.where(hidden, np.nan, counts).copy()
        missing = np.isnan(values)
        if status == 'partial':
            values[tuple(np.argwhere(missing)[0])] = 0.
        info = {'status': status, 'unresolved_cells': int(np.isnan(values).sum()),
                'error': {'code': 'test_failure', 'message': 'original cause', 'context': {'row': 1}}}
        return CovidResult(values, status, info, totals - np.nansum(values, axis=1))

    monkeypatch.setattr(MICoDaCovid, 'impute_counts', simulate)
    code, out = call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA, '[micoda]\n',
                     '--mask', 'random_cell', '--mask-rate', '.2', '--seeds', '7')
    assert code == 1
    summary = json.loads((out / 'summary.json').read_text())
    for result in (summary['native']['micoda'], summary['masked']['7']['models']['micoda']):
        assert result['status'] == status
        assert result['info']['error']['message'] == 'original cause'
        assert 'unassigned' in result['info']
        assert not result.get('errors')
    assert (out / 'micoda_partial.csv').exists() == (status == 'partial')
    assert not (out / 'micoda.csv').exists()
    assert not (out / 'micoda_completed.csv').exists()


def test_legacy_impute_does_not_return_failed_raw_counts(tmp_path):
    from longitudinal_impute.data import load_csv
    source = write(tmp_path, 'insufficient.csv',
                   'location,date,total_sequence,v1,v2,v3,v4\nA,1,100,10,,,\n')
    data = load_csv(source, write(tmp_path, 'schema.toml', COVID_SCHEMA))
    with pytest.raises(ValueError, match='insufficient_fit'):
        run.impute(data, MICoDaCovid(), np.zeros_like(data.counts, dtype=bool))
