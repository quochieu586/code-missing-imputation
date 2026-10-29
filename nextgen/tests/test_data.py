"""Generic CSV loading, evaluation masks and the hidden-truth firewall."""
import csv as csv_module
from pathlib import Path

import numpy as np
import pytest

from longitudinal_impute.data import load_csv, make_mask, model_input, write_completed_csv

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
COVID_SCHEMA = '[dataset]\nentity="location"\ntime="date"\ntotal="total_sequence"\ntargets=["v1","v2","v3","v4"]\n'
MICROBIOME_SCHEMA = ('[dataset]\nentity="subject"\ntime="visit"\ntargets=["Bacteroides","Prevotella",'
                     '"Faecalibacterium","Akkermansia","Bifidobacterium"]\n')


def schema(tmp_path, text):
    path = tmp_path / 'schema.toml'
    path.write_text(text)
    return path


@pytest.fixture
def covid(tmp_path):
    return load_csv(FIXTURES / 'covid_small.csv', schema(tmp_path, COVID_SCHEMA))


@pytest.fixture
def microbiome(tmp_path):
    return load_csv(FIXTURES / 'microbiome_small.csv', schema(tmp_path, MICROBIOME_SCHEMA))


def test_without_total_column_the_total_is_the_row_sum(microbiome):
    assert np.isfinite(microbiome.counts).all()
    np.testing.assert_array_equal(microbiome.totals, microbiome.counts.sum(axis=1))
    assert microbiome.entities[:6] == ('S01',) * 5 + ('S02',)
    assert microbiome.lines[0] == 2
    other = model_input(microbiome, np.zeros(microbiome.counts.shape, bool))[:, -1]
    np.testing.assert_array_equal(other, 0.)


def test_missing_target_without_total_column_names_line_and_entity(tmp_path):
    csv = tmp_path / 'data.csv'
    csv.write_text('subject,a,b\nS1,1,2\nS2,,3\n')
    with pytest.raises(ValueError, match=r"CSV line 3 \(entity 'S2'\): targets \['a'\] are missing.*no total"):
        load_csv(csv, schema(tmp_path, '[dataset]\nentity="subject"\ntargets=["a","b"]\n'))


@pytest.mark.parametrize('body,message', [
    ('L,d1,10,x,1\n', r"CSV line 2 \(entity 'L'\): column 'a' = 'x' is not a number"),
    ('L,d1,10,-1,1\n', r"CSV line 2 \(entity 'L'\): column 'a' = '-1' must be finite and nonnegative"),
    ('L,d1,10,1,1\nL,d1,10,1,1\n', r"CSV line 3 \(entity 'L'\): empty or duplicate 'date'"),
    ('L,d1,1,1,1\n', r"CSV line 2 \(entity 'L'\): column 'n' = 1 is not positive or below"),
    ('L,d1,,1,1\n', r"CSV line 2 \(entity 'L'\): column 'n' is empty"),
])
def test_load_errors_name_line_entity_and_column(tmp_path, body, message):
    csv = tmp_path / 'data.csv'
    csv.write_text('loc,date,n,a,b\n' + body)
    with pytest.raises(ValueError, match=message):
        load_csv(csv, schema(tmp_path, '[dataset]\nentity="loc"\ntime="date"\ntotal="n"\ntargets=["a","b"]\n'))


def test_schema_needs_entity_and_rejects_unknown_keys(tmp_path):
    csv = tmp_path / 'data.csv'
    csv.write_text('loc,a,b\nL,1,1\n')
    with pytest.raises(ValueError, match='needs entity and targets'):
        load_csv(csv, schema(tmp_path, '[dataset]\ntargets=["a","b"]\n'))
    with pytest.raises(ValueError, match='unknown'):
        load_csv(csv, schema(tmp_path, '[dataset]\nentity="loc"\ntargets=["a","b"]\ncovariates=["x"]\n'))


@pytest.mark.parametrize('kind', ['random_cell', 'whole_feature'])
def test_mask_is_a_reproducible_subset_that_keeps_a_visible_cell(covid, kind):
    observed = np.isfinite(covid.counts)
    hidden, stats = make_mask(covid, kind, .3, 4)
    again, _ = make_mask(covid, kind, .3, 4)
    np.testing.assert_array_equal(hidden, again)
    assert hidden.any() and not (hidden & ~observed).any()
    assert (observed & ~hidden).any(axis=1)[observed.any(axis=1)].all()
    assert stats['hidden_cells'] == hidden.sum()
    assert stats['actual_rate'] == hidden.sum() / observed.sum()
    assert stats['rows_affected'] == hidden.any(axis=1).sum()
    assert stats['complete_rows_after'] == (observed.all(axis=1) & ~hidden.any(axis=1)).sum()
    assert not np.array_equal(hidden, make_mask(covid, kind, .3, 5)[0])


def test_whole_feature_hides_entire_series_and_reports_skips(covid):
    observed = np.isfinite(covid.counts)
    entities = np.asarray(covid.entities)
    hidden, stats = make_mask(covid, 'whole_feature', .3, 4)
    for e in np.unique(entities):
        for j in range(observed.shape[1]):
            series = (entities == e) & observed[:, j]
            assert not hidden[series, j].any() or hidden[series, j].all()
    pairs = {(e, j) for e, j in zip(entities[np.nonzero(hidden)[0]], np.nonzero(hidden)[1])}
    assert stats['pairs_hidden'] == len(pairs)
    assert stats['hidden_cells'] >= .3 * observed.sum()
    # Corin 2021-05-04 (CSV line 22) only shows v3, so the pair (Corin, v3) is never hidden.
    row = covid.lines.index(22)
    assert np.isfinite(covid.counts[row]).tolist() == [False, False, True, False]
    assert make_mask(covid, 'whole_feature', .9, 0)[1]['pairs_skipped'] > 0
    for seed in range(20):
        assert not make_mask(covid, 'whole_feature', .9, seed)[0][row, 2]


def test_mask_rejects_unknown_kind_and_rate(covid):
    with pytest.raises(ValueError, match='kind'):
        make_mask(covid, 'block', .1, 0)
    with pytest.raises(ValueError, match='rate'):
        make_mask(covid, 'random_cell', 1., 0)


@pytest.mark.parametrize('kind', ['random_cell', 'whole_feature'])
@pytest.mark.parametrize('positive', [False, True])
def test_hidden_truth_never_reaches_model_input_not_even_through_other(covid, kind, positive):
    hidden, _ = make_mask(covid, kind, .3, 2)
    before = model_input(covid, hidden, positive=positive)
    covid.counts[hidden] = 1e9 * (1 + np.arange(hidden.sum()))
    np.testing.assert_array_equal(before, model_input(covid, hidden, positive=positive))
    assert np.isnan(before[:, :-1][hidden]).all()
    assert np.isnan(before[hidden.any(axis=1), -1]).all()


MALFORMED_SCHEMA = '[dataset]\nentity="location"\ntime="date"\ntotal="total"\ntargets=["a","b","c"]\n'
HEADER = 'location,date,total,a,b,c,note\nL,2020-01-01,100,10,20,70,x\nL,2020-01-02,100,20,30,50,x\n'


@pytest.mark.parametrize('last,message', [
    ('L,2020-01-03,100,20,,,x,EXTRA', r"CSV line 4 has 1 more field\(s\) than the 7 header columns, after column 'note'"),
    ('L,2020-01-03,100,20', r"CSV line 4 has 4 of 7 fields; columns \['b', 'c', 'note'\] are missing"),
    ('L,2020-01-03,100,20,10,70', r"CSV line 4 has 6 of 7 fields; columns \['note'\] are missing"),
])
def test_rows_with_surplus_or_absent_fields_are_rejected_with_line_and_columns(tmp_path, last, message):
    csv = tmp_path / 'data.csv'
    csv.write_text(HEADER + last + '\n')
    with pytest.raises(ValueError, match=message):
        load_csv(csv, schema(tmp_path, MALFORMED_SCHEMA))


def completed_table(covid):
    return np.where(np.isfinite(covid.counts), covid.counts, 0.)


def test_writer_publishes_only_complete_files_and_never_overwrites(tmp_path, covid):
    target = tmp_path / 'out' / 'm_completed.csv'
    target.parent.mkdir()
    write_completed_csv(target, covid, completed_table(covid))
    assert [p.name for p in target.parent.iterdir()] == ['m_completed.csv']
    assert len(target.read_text(encoding='utf-8').splitlines()) == len(covid.rows) + 1
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_completed_csv(target, covid, completed_table(covid))
    assert target.read_bytes() == before
    assert [p.name for p in target.parent.iterdir()] == ['m_completed.csv']


def test_interrupted_write_leaves_neither_the_final_nor_a_temporary_file(tmp_path, covid, monkeypatch):
    written = []
    original = csv_module.DictWriter.writerow

    def fail_on_third_row(self, row):
        if len(written) == 3:
            raise OSError('disk full (simulated)')
        written.append(row)
        return original(self, row)

    monkeypatch.setattr(csv_module.DictWriter, 'writerow', fail_on_third_row)
    target = tmp_path / 'out' / 'm_completed.csv'
    target.parent.mkdir()
    with pytest.raises(OSError, match='disk full'):
        write_completed_csv(target, covid, completed_table(covid))
    assert len(written) == 3                        # header and two rows reached the temporary file
    assert list(target.parent.iterdir()) == []
