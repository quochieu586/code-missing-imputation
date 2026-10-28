"""Generic CSV loading, evaluation masks and the hidden-truth firewall."""
from pathlib import Path

import numpy as np
import pytest

from longitudinal_impute.data import load_csv, make_mask, model_input

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
