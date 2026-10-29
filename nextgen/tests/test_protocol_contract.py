"""G1 contract tests for the unified 3-baseline protocol.

The contract is defined in reports/pipeline_protocol_3_baselines_2026-10-29.md
(sections 2.3-2.8, decisions D-01..D-05). At G1 they documented the unmet
contract points of PR head `edfb0a2` (7 red / 5 green). In G2 the loader
strictness, atomic publish and mask preflight were ported, so those tests now
assert the ported behavior; the remaining reds are the MICoDa adapter (G3) and
the summary `space` label (G3/G4). No production code is touched in this file
after the port; PR author source semantics are preserved everywhere else.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from longitudinal_impute import run
from longitudinal_impute.data import load_csv, write_completed_csv

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
COVID = FIXTURES / 'covid_small.csv'
SCHEMA = ('[dataset]\nentity="location"\ntime="date"\ntotal="total_sequence"\n'
          'targets=["v1","v2","v3","v4"]\n')


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def hand_imputed(data):
    """A valid full completion: the residual is split uniformly over missing cells."""
    counts = data.counts.copy()
    missing = ~np.isfinite(counts)
    residual = np.maximum(data.totals - np.nansum(counts, axis=1), 0.)[:, None]
    per_cell = residual / np.maximum(missing.sum(axis=1, keepdims=True), 1)
    fill = np.where(missing, per_cell, 0.)
    counts[missing] = fill[missing]
    return counts


def test_contract_covid_header():
    text = COVID.read_text()
    assert 'total_sequence' in text


def test_models_config_validates_before_any_output(tmp_path):
    out = tmp_path / 'out'
    schema = write(tmp_path, 'schema.toml', SCHEMA)
    with pytest.raises(ValueError, match='neighbours'):
        run.main(['--data', str(COVID), '--schema', str(schema),
                  '--models', str(write(tmp_path, 'models.toml', '[hron]\nneighbours = 2\n')),
                  '--output', str(out)])
    assert not out.exists()


def test_preflight_rejects_bad_mask_config_before_any_run(tmp_path):
    """Bad mask rate must be rejected before any run or directory side effect (D-04)."""
    out = tmp_path / 'out'
    schema = write(tmp_path, 'schema.toml', SCHEMA)
    with pytest.raises(ValueError, match='rate'):
        run.main(['--data', str(COVID), '--schema', str(schema),
                  '--models', str(write(tmp_path, 'models.toml', '[hron]\nk = 3\n')),
                  '--output', str(out), '--mask', 'random_cell', '--mask-rate', '1.0'])
    assert not out.exists()
    assert not list(tmp_path.glob('*.tmp'))


def test_loader_rejects_record_with_extra_fields(tmp_path):
    source = COVID.read_text()
    lines = source.splitlines()
    bad = lines[:]
    bad[9] = bad[9] + ',EXTRA'
    source_path = write(tmp_path, 'extra.csv', '\n'.join(bad) + '\n')
    with pytest.raises(ValueError, match='fields'):
        load_csv(source_path, write(tmp_path, 'schema.toml', SCHEMA))


def test_loader_rejects_record_with_missing_fields(tmp_path):
    lines = COVID.read_text().splitlines()
    lines[9] = ','.join(lines[9].split(',')[:-1])
    source_path = write(tmp_path, 'short.csv', '\n'.join(lines) + '\n')
    with pytest.raises(ValueError, match='fields'):
        load_csv(source_path, write(tmp_path, 'schema.toml', SCHEMA))


def test_completed_csv_never_published_when_run_failed(tmp_path):
    """A record with one extra field must fail at the boundary, before publish.

    Contract point (TM-001): the run must not leave a truncated `<model>_completed.csv`.
    """
    lines = COVID.read_text().splitlines()
    lines[9] = lines[9] + ',EXTRA'
    extra = write(tmp_path, 'extra.csv', '\n'.join(lines) + '\n')
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='fields'):
        run.main(['--data', str(extra), '--schema', str(write(tmp_path, 'schema.toml', SCHEMA)),
                  '--models', str(write(tmp_path, 'models.toml', '[hron]\nk = 3\n')),
                  '--output', str(out), '--mask', 'none'])
    assert not out.exists()


def test_completed_writer_never_overwrites(tmp_path):
    data = load_csv(COVID, write(tmp_path, 'schema.toml', SCHEMA))
    destination = tmp_path / 'out.csv'
    destination.write_text('do-not-touch\n')
    with pytest.raises(FileExistsError):
        write_completed_csv(destination, data, hand_imputed(data))
    assert destination.read_text() == 'do-not-touch\n'


def test_micoda_model_accepts_protocol_params_and_rejects_W(tmp_path):
    model = run.MODELS['micoda'](max_iter=2, tol=1e-8, reference=0)
    assert model is not None
    with pytest.raises((TypeError, ValueError), match='W'):
        run.MODELS['micoda'](**{'W': 1e-6, 'max_iter': 2, 'tol': 1e-8})


def test_micoda_native_pass_fills_nan_on_covid_fixture(tmp_path):
    out = tmp_path / 'out'
    code = run.main(['--data', str(COVID), '--schema', str(write(tmp_path, 'schema.toml', SCHEMA)),
                     '--models', str(write(tmp_path, 'models.toml', '[micoda]\nmax_iter = 40\ntol = 1e-8\n')),
                     '--output', str(out), '--mask', 'none'])
    assert code == 0
    data = load_csv(out / 'micoda_completed.csv', write(tmp_path, 'schema.toml', SCHEMA))
    raw = load_csv(COVID, write(tmp_path, 'schema2.toml', SCHEMA))
    assert data.counts.shape == raw.counts.shape
    observed = np.isfinite(raw.counts)
    np.testing.assert_array_equal(data.counts[observed], raw.counts[observed])
    assert np.isfinite(data.counts).all()


def test_summary_labels_model_space(tmp_path):
    out = tmp_path / 'out'
    run.main(['--data', str(COVID), '--schema', str(write(tmp_path, 'schema.toml', SCHEMA)),
              '--models', str(write(tmp_path, 'models.toml', '[hron]\n[tsagris]\n')),
              '--output', str(out), '--mask', 'none'])
    with (out / 'summary.csv').open(newline='', encoding='utf-8') as stream:
        header = stream.readline().strip().split(',')
    assert 'space' in header
    with (out / 'summary.json').open(encoding='utf-8') as stream:
        aggregate = json.load(stream)['aggregate']
    spaces = {row['model']: row['space'] for row in aggregate}
    assert spaces == {'hron': '18_with_other', 'tsagris': '18_with_other'}


def test_summary_reports_runs_ok_and_failed(tmp_path):
    out = tmp_path / 'out'
    run.main(['--data', str(COVID), '--schema', str(write(tmp_path, 'schema.toml', SCHEMA)),
              '--models', str(write(tmp_path, 'models.toml', '[hron]\n[tsagris]\n')),
              '--output', str(out), '--mask', 'none'])
    with (out / 'summary.json').open(encoding='utf-8') as stream:
        aggregate = json.load(stream)['aggregate']
    for row in aggregate:
        assert row['runs_ok'] == 1 and row['runs_failed'] == 0


def test_exit_zero_when_all_runs_complete(tmp_path):
    out = tmp_path / 'out'
    code = run.main(['--data', str(COVID), '--schema', str(write(tmp_path, 'schema.toml', SCHEMA)),
                     '--models', str(write(tmp_path, 'models.toml', '[hron]\n[tsagris]\n')),
                     '--output', str(out), '--mask', 'none'])
    assert code == 0
    assert (out / 'hron_completed.csv').exists()
    assert (out / 'tsagris_completed.csv').exists()
