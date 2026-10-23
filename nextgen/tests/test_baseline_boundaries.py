"""Additional scoped-fit/export properties; no filesystem cleanup fixtures."""
import copy
import csv
import io
import json
from pathlib import Path
import numpy as np
import pytest
from longitudinal_impute.baselines.adapters import CountBaselineAdapter
from longitudinal_impute.baselines.hron import HronKNN
from longitudinal_impute.baselines.micoda import MICoDa
from longitudinal_impute.baselines.tsagris import TsagrisKNN
from longitudinal_impute.baselines.common import BaselineError
from longitudinal_impute.baselines.csv_io import schema_from_config, load_csv, write_completed_csv
from longitudinal_impute.contracts import FitScope, ImputationResult
from test_general_baselines import panel, positive_data


@pytest.mark.parametrize('core', [HronKNN(2), MICoDa(W=1e-5), TsagrisKNN(2)])
def test_hidden_training_truth_does_not_change_fit_state(core):
    train = panel(positive_data(n=24, d=4) * 70)
    train.counts[:2, 0] = np.nan; train.conditioning_mask[:2, 0] = False
    scope = FitScope('inductive', train.row_ids, 's', train.mask_id)
    query = panel([[10., 20., np.nan, np.nan]])
    a = CountBaselineAdapter(core).fit(train, scope).predict(query)
    poisoned = copy.deepcopy(train)
    poisoned.counts[~poisoned.conditioning_mask] = 1e12
    b = CountBaselineAdapter(core).fit(poisoned, scope).predict(query)
    np.testing.assert_array_equal(a, b)


def test_future_observations_are_allowed_donor_context():
    # No triangular time filter is hidden in a general-imputation adapter.
    train = panel([[10., 20., 30.], [20., 10., 5.]])
    train.timestamps[:] = [100, 200]  # both later than the query
    scope = FitScope('inductive', train.row_ids, 's', train.mask_id)
    query = panel([[10., 20., np.nan]])
    query.timestamps[:] = [1]
    model = CountBaselineAdapter(HronKNN(1)).fit(train, scope)
    assert model.predict(query)[0, 2] > 0
    changed = copy.deepcopy(train); changed.counts[0, 2] = 50.
    alternate = CountBaselineAdapter(HronKNN(1)).fit(changed, scope)
    assert alternate.predict(query)[0, 2] != pytest.approx(model.predict(query)[0, 2])


def test_csv_schema_and_export_preserve_observed_text(monkeypatch):
    schema = schema_from_config({'dataset': {'entity': 'location', 'time': 'date',
                                             'total': 'N', 'targets': ['A', 'B'],
                                             'metadata_candidates': ['continent']}})
    text = 'location,date,N,A,B,continent\nx,2020-01-01,100,00020,,Asia\ny,2020-01-02,100,0,30,Europe\n'
    original_open = Path.open
    def read_mock(path, *args, **kwargs):
        if str(path) == 'test_input.csv': return io.StringIO(text)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', read_mock)
    raw, fields, rows = load_csv('test_input.csv', schema)
    # Metadata categorical strings did not become target features.
    assert raw.counts.shape == (2, 2)
    assert np.isnan(raw.counts[0, 1]) and raw.counts[1, 0] == 0
    result = ImputationResult(raw.row_ids, np.array([[20., 40.], [0., 30.]]), {})
    class RetainedString(io.StringIO):
        def close(self): self.saved = self.getvalue()
    output = RetainedString()
    def write_mock(path, *args, **kwargs):
        if str(path) == 'test_output.csv':
            assert args[0] == 'x'
            return output
        return read_mock(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', write_mock)
    write_completed_csv('test_output.csv', raw, result, fields, rows)
    values = list(csv.DictReader(io.StringIO(output.saved)))
    assert values[0]['A'] == '00020' and values[0]['B'] == '40'
    assert values[1]['A'] == '0' and values[1]['continent'] == 'Europe'
    changed = ImputationResult(raw.row_ids, np.array([[21., 40.], [0., 30.]]), {})
    with pytest.raises(BaselineError): write_completed_csv('test_output.csv', raw, changed, fields, rows)
